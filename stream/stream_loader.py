"""Standalone stream loader for video streams and files.

This module provides the StreamLoader class for loading video streams
from various sources (RTSP, RTMP, HTTP, TCP) and local video files.
It is completely standalone with no dependencies on ultralytics modules.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from threading import Thread
from typing import Iterator

import cv2
import numpy as np

from .constants import DEFAULT_BUFFER_SIZE, DEFAULT_FPS_FALLBACK
from .exceptions import StreamConnectionError, StreamReadError
from .utils import (
    clean_str,
    format_stream_info,
    get_stream_file_logger,
    is_streams_file,
    parse_sources_file,
    setup_logger,
)

# Console logger (for backward compatibility with stream info output)
logger = setup_logger("stream_loader")

# Stream file logger (writes to logs/streams.log, no console output)
stream_log = get_stream_file_logger()

# Recovery delay in seconds (5 minutes)
STREAM_RECOVERY_DELAY = 300


class StreamLoader:
    """Standalone stream loader for RTSP, RTMP, HTTP, TCP streams and video files.

    This class handles the loading and processing of multiple video streams
    simultaneously, making it suitable for real-time video analysis tasks.

    Attributes:
        sources (list[str]): Clean source identifiers for each stream.
        raw_sources (list[str]): Original source URLs/paths.
        vid_stride (int): Video frame-rate stride (process every Nth frame).
        buffer (bool): Whether to buffer input streams (FIFO) or keep latest only.
        running (bool): Flag indicating if reader threads are active.
        mode (str): Always 'stream' for this loader.
        imgs (list[list[np.ndarray]]): Frame buffers for each stream.
        fps (list[float]): FPS for each stream.
        frames (list[int]): Total frame count for each stream.
        threads (list[Thread]): Reader threads for each stream.
        shape (list[tuple]): Frame shapes (H, W, C) for each stream.
        caps (list[cv2.VideoCapture]): OpenCV capture objects.
        bs (int): Batch size (number of streams).
        cv2_flag (int): OpenCV image read flag (grayscale or color).

    Examples:
        >>> loader = StreamLoader("rtsp://192.168.1.100:554/stream")
        >>> for sources, imgs, info in loader:
        ...     # Process the images
        ...     pass
        >>> loader.close()

        >>> # Using context manager
        >>> with StreamLoader("video.mp4") as loader:
        ...     for sources, imgs, info in loader:
        ...         process(imgs)

        >>> # Multiple streams
        >>> loader = StreamLoader(["rtsp://cam1/stream", "rtsp://cam2/stream"])
    """

    def __init__(
        self,
        sources: str | list[str] = "file.streams",
        vid_stride: int = 1,
        buffer: bool = False,
        channels: int = 3,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        reconnect: bool = True,
    ):
        """Initialize stream loader for multiple video sources.

        Args:
            sources: Path to streams file, single stream URL, video file path,
                or list of sources.
            vid_stride: Video frame-rate stride (process every Nth frame).
            buffer: Whether to buffer input streams. If True, keeps up to
                buffer_size frames in FIFO order. If False, keeps only the
                latest frame.
            channels: Number of image channels (1 for grayscale, 3 for RGB).
            buffer_size: Maximum frames to buffer per stream (default 30).
            reconnect: Whether to auto-reconnect on stream loss.
        """
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.running = True
        self.mode = "stream"
        self.vid_stride = vid_stride
        self.reconnect = reconnect
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

        # Parse sources
        self.raw_sources = self._parse_sources(sources)
        n = len(self.raw_sources)
        self.bs = n

        # Initialize per-stream attributes
        self.fps: list[float] = [0.0] * n
        self.frames: list[int | float] = [0] * n
        self.threads: list[Thread | None] = [None] * n
        self.caps: list[cv2.VideoCapture | None] = [None] * n
        self.imgs: list[list[np.ndarray]] = [[] for _ in range(n)]
        self.shape: list[tuple] = [() for _ in range(n)]
        self.sources: list[str] = [
            clean_str(x).replace(os.sep, "_") for x in self.raw_sources
        ]

        # Open streams and start reader threads
        for i, s in enumerate(self.raw_sources):
            self._open_stream(i, s)

        logger.info("")  # newline after stream info

    def _parse_sources(self, sources: str | list[str]) -> list[str]:
        """Parse source input to list of stream URLs/paths.

        Args:
            sources: Source input (file path, URL, or list).

        Returns:
            List of source strings.
        """
        if isinstance(sources, list):
            return sources

        # Check if it's a .streams file
        if is_streams_file(sources):
            return parse_sources_file(sources)

        # Check if it's a regular file containing stream URLs
        if os.path.isfile(sources) and not self._is_media_file(sources):
            return Path(sources).read_text().strip().split()

        # Single source
        return [sources]

    def _is_media_file(self, path: str) -> bool:
        """Check if path is a media file (video/image).

        Args:
            path: File path to check.

        Returns:
            True if path is a media file.
        """
        from .constants import IMG_FORMATS, VID_FORMATS

        ext = Path(path).suffix.lower().lstrip(".")
        return ext in VID_FORMATS or ext in IMG_FORMATS

    def _open_stream(self, index: int, source: str) -> None:
        """Open a single stream and start reader thread.

        Args:
            index: Stream index.
            source: Stream URL or file path.

        Raises:
            StreamConnectionError: If unable to open stream.
            StreamReadError: If unable to read first frame.
        """
        st = f"{index + 1}/{self.bs}: {source}... "

        # Convert numeric string to int for webcam
        s = int(source) if source.isnumeric() else source

        # Open video capture
        cap = cv2.VideoCapture(s)
        if not cap.isOpened():
            raise StreamConnectionError(source, "Failed to open stream")

        self.caps[index] = cap

        # Get stream properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Handle invalid FPS values
        self.fps[index] = max((fps if math.isfinite(fps) else 0) % 100, 0) or DEFAULT_FPS_FALLBACK

        # Get frame count (infinite for live streams)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames[index] = max(frame_count, 0) or float("inf")

        # Read first frame to verify connection
        success, im = cap.read()
        if self.cv2_flag == cv2.IMREAD_GRAYSCALE:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]

        if not success or im is None:
            raise StreamReadError(source, "Failed to read first frame")

        self.imgs[index].append(im)
        self.shape[index] = im.shape

        # Start reader thread
        self.threads[index] = Thread(
            target=self.update,
            args=(index, cap, s),
            daemon=True
        )
        logger.info(format_stream_info(
            index, self.bs, source, w, h, self.fps[index], self.frames[index]
        ))
        self.threads[index].start()

    def update(self, index: int, cap: cv2.VideoCapture, stream: str | int) -> None:
        """Read stream frames in daemon thread and update image buffer.

        This method runs continuously in a background thread, reading frames
        from the video capture and storing them in the buffer.

        Implements 5-minute recovery delay on stream disconnection:
        - Task keeps running with placeholder (black) frames
        - Reconnect attempt after 5 minutes
        - Seamless resume after successful reconnect

        Args:
            index: Stream index.
            cap: OpenCV VideoCapture object.
            stream: Stream URL/path or webcam index.
        """
        n = 0  # frame number
        f = self.frames[index]  # total frames

        # Recovery state tracking
        if not hasattr(self, '_reconnect_scheduled'):
            self._reconnect_scheduled = {}
        if not hasattr(self, '_disconnect_logged'):
            self._disconnect_logged = {}

        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[index]) < self.buffer_size:
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()

                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()

                    # Convert to grayscale if needed
                    if self.cv2_flag == cv2.IMREAD_GRAYSCALE:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None]

                    if not success:
                        # Create placeholder frame (black) - task keeps processing
                        im = np.zeros(self.shape[index], dtype=np.uint8)

                        # Log disconnection (only once per disconnect event)
                        if index not in self._disconnect_logged:
                            stream_log(f"Stream {index} disconnected - using placeholder frames")
                            self._disconnect_logged[index] = True

                        # Schedule reconnect with 5-minute delay
                        if self.reconnect:
                            if index not in self._reconnect_scheduled:
                                self._reconnect_scheduled[index] = time.time()
                                stream_log(f"Stream {index}: scheduling reconnect in {STREAM_RECOVERY_DELAY // 60} minutes")

                            # Check if recovery delay has passed
                            elapsed = time.time() - self._reconnect_scheduled[index]
                            if elapsed >= STREAM_RECOVERY_DELAY:
                                stream_log(f"Stream {index}: attempting reconnect after {elapsed:.0f}s")
                                reconnect_success = cap.open(stream)
                                if reconnect_success and cap.isOpened():
                                    # Verify with a test read
                                    test_success, _ = cap.read()
                                    if test_success:
                                        stream_log(f"Stream {index}: reconnected successfully")
                                        # Clear recovery state
                                        del self._reconnect_scheduled[index]
                                        if index in self._disconnect_logged:
                                            del self._disconnect_logged[index]
                                        continue  # Skip this frame, use fresh one next iteration
                                    else:
                                        stream_log(f"Stream {index}: reconnect failed (read test failed)")
                                else:
                                    stream_log(f"Stream {index}: reconnect failed (cap.open failed)")

                                # Reset timer for next attempt
                                self._reconnect_scheduled[index] = time.time()
                                stream_log(f"Stream {index}: retrying in {STREAM_RECOVERY_DELAY // 60} minutes")
                    else:
                        # Successful read - clear any recovery state
                        if index in self._reconnect_scheduled:
                            del self._reconnect_scheduled[index]
                        if index in self._disconnect_logged:
                            stream_log(f"Stream {index}: recovered, resuming normal operation")
                            del self._disconnect_logged[index]

                    if self.buffer:
                        self.imgs[index].append(im)
                    else:
                        self.imgs[index] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer has space

    def close(self) -> None:
        """Terminate stream loader, stop threads, and release video capture resources."""
        self.running = False

        # Stop all reader threads
        for thread in self.threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)

        # Release all video captures
        for cap in self.caps:
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.warning(f"Could not release VideoCapture object: {e}")

    def __iter__(self) -> Iterator:
        """Return iterator object for the stream loader."""
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """Return the next batch of frames from video streams.

        Returns:
            Tuple containing:
                - sources: List of source identifiers
                - images: List of numpy arrays (frames)
                - info: List of info strings (empty)

        Raises:
            StopIteration: When all streams have ended.
        """
        self.count += 1

        # Rate limiting for "Waiting for stream" message (log max once per 60 seconds)
        if not hasattr(self, '_wait_logged'):
            self._wait_logged = {}

        images = []
        for i, x in enumerate(self.imgs):
            # Wait until a frame is available in each buffer
            while not x:
                if self.threads[i] is None or not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    # Rate-limited logging to streams.log (max once per 60 seconds per stream)
                    current_time = time.time()
                    if i not in self._wait_logged or current_time - self._wait_logged[i] > 60:
                        stream_log(f"Waiting for stream {i} - buffer empty")
                        self._wait_logged[i] = current_time

            # Get frame based on buffer mode
            if self.buffer:
                # FIFO: get and remove the first frame
                images.append(x.pop(0))
            else:
                # Latest only: get last frame, clear rest
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self) -> int:
        """Return the number of video streams."""
        return self.bs

    def __enter__(self) -> "StreamLoader":
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def get_frame(self, index: int = 0, timeout: float | None = None) -> np.ndarray | None:
        """Get a single frame from specified stream.

        Args:
            index: Stream index (default 0).
            timeout: Maximum time to wait for a frame in seconds.

        Returns:
            Frame as numpy array, or None if timeout.
        """
        if index >= self.bs:
            raise IndexError(f"Stream index {index} out of range (0-{self.bs - 1})")

        start = time.time()
        while True:
            if self.imgs[index]:
                if self.buffer:
                    return self.imgs[index].pop(0)
                else:
                    frame = self.imgs[index].pop(-1) if self.imgs[index] else None
                    self.imgs[index].clear()
                    return frame

            if timeout is not None and (time.time() - start) > timeout:
                return None

            if self.threads[index] is None or not self.threads[index].is_alive():
                return None

            time.sleep(0.01)

    def get_stream_properties(self, index: int = 0) -> dict:
        """Get actual stream properties for a given stream index.

        Returns the actual resolution and FPS as detected from the video stream
        via OpenCV VideoCapture. These values may differ from configured values.

        Args:
            index: Stream index (default 0).

        Returns:
            Dictionary with:
                - width: Actual stream width in pixels
                - height: Actual stream height in pixels
                - fps: Actual frames per second
        """
        if index >= len(self.shape):
            return {"width": 0, "height": 0, "fps": 0.0}

        # shape is (H, W, C) from OpenCV
        h, w = self.shape[index][:2]
        fps = self.fps[index]

        return {
            "width": w,
            "height": h,
            "fps": fps,
        }

    @property
    def is_alive(self) -> bool:
        """Check if any stream thread is still running."""
        return any(t is not None and t.is_alive() for t in self.threads)
