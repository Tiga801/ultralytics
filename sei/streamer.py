# -*- coding: utf-8 -*-
"""
RTMP Streamer Module

This module provides RTMP streaming functionality for pushing H.264 encoded
video to RTMP servers. Uses FFmpeg for protocol handling and data transmission.

Features:
- FFmpeg-based RTMP streaming
- Automatic reconnection on failure
- Health monitoring and statistics
- Low-latency configuration
- Thread-safe operations

Usage:
    >>> from sei.streamer import RtmpStreamer
    >>> from sei.config import StreamConfig
    >>> config = StreamConfig(width=1920, height=1080, fps=10)
    >>> streamer = RtmpStreamer(config)
    >>> streamer.start("rtmp://server/live/stream")
    >>> streamer.push_frame(h264_data)
    >>> streamer.stop()
"""

import subprocess
import threading
import time
from typing import Optional

from .config import StreamConfig
from .interfaces import StreamerInterface, LogFunc


class RtmpStreamer(StreamerInterface):
    """
    RTMP Streaming Implementation.

    Streams H.264 encoded video to RTMP servers using FFmpeg.
    Handles connection management, reconnection, and health monitoring.

    FFmpeg Command:
        ffmpeg -f h264 -i - [low-latency options] -f flv rtmp://...

    Attributes:
        config: Stream configuration
        log: Logging function

    Example:
        >>> from sei.streamer import RtmpStreamer
        >>> from sei.config import StreamConfig
        >>> config = StreamConfig()
        >>> streamer = RtmpStreamer(config)
        >>> streamer.start("rtmp://server/live/stream")
        >>> for frame in frames:
        ...     streamer.push_frame(frame.h264_data)
        >>> streamer.stop()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize RTMP streamer.

        Args:
            config: Stream configuration
            log_func: Logging function
        """
        self._config = config or StreamConfig()
        self._log = log_func or self._default_log

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._streaming = False
        self._rtmp_url = ""

        # Statistics
        self._bytes_pushed = 0
        self._frames_pushed = 0
        self._frames_failed = 0
        self._last_push_time: Optional[float] = None
        self._start_time: Optional[float] = None

        # Health monitoring
        self._reconnect_count = 0
        self._last_error: Optional[str] = None

    @staticmethod
    def _default_log(message: str) -> None:
        """Default logging to stdout with timestamp."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def start(self, rtmp_url: str) -> bool:
        """
        Start streaming to RTMP server.

        Args:
            rtmp_url: RTMP server URL (e.g., rtmp://server/live/stream)

        Returns:
            True if streaming started successfully
        """
        with self._lock:
            if self._streaming:
                if self._rtmp_url == rtmp_url:
                    return True
                else:
                    # URL changed, need to restart
                    self._stop_process()

            try:
                self._rtmp_url = rtmp_url
                self._process = self._create_streamer_process(rtmp_url)
                self._streaming = True
                self._bytes_pushed = 0
                self._frames_pushed = 0
                self._frames_failed = 0
                self._start_time = time.time()
                self._log(f"RTMP streaming started: {rtmp_url}")
                return True

            except Exception as e:
                self._last_error = str(e)
                self._log(f"Failed to start streaming: {e}")
                return False

    def stop(self) -> None:
        """Stop streaming and close connection."""
        with self._lock:
            self._streaming = False
            self._stop_process()
            self._log("RTMP streaming stopped")

    def push_frame(self, frame_data: bytes) -> bool:
        """
        Push H.264 frame data to stream.

        Args:
            frame_data: H.264 encoded frame bytes

        Returns:
            True if pushed successfully
        """
        if not self._streaming or not self._process:
            return False

        try:
            self._process.stdin.write(frame_data)
            self._process.stdin.flush()

            self._bytes_pushed += len(frame_data)
            self._frames_pushed += 1
            self._last_push_time = time.time()

            return True

        except BrokenPipeError:
            self._frames_failed += 1
            self._last_error = "Broken pipe"
            self._log("Stream pipe broken, attempting reconnection...")
            self._handle_reconnect()
            return False

        except Exception as e:
            self._frames_failed += 1
            self._last_error = str(e)
            self._log(f"Push error: {e}")
            return False

    def is_streaming(self) -> bool:
        """Check if actively streaming."""
        return self._streaming and self._process is not None

    def is_healthy(self) -> bool:
        """
        Check streaming health.

        Returns:
            True if streaming and no recent errors
        """
        if not self._streaming or not self._process:
            return False

        # Check if process is still running
        if self._process.poll() is not None:
            return False

        # Check for recent activity
        if self._last_push_time:
            idle_time = time.time() - self._last_push_time
            if idle_time > 30:  # 30 seconds idle threshold
                return False

        return True

    @property
    def bytes_pushed(self) -> int:
        """Get total bytes pushed."""
        return self._bytes_pushed

    @property
    def frames_pushed(self) -> int:
        """Get total frames pushed."""
        return self._frames_pushed

    @property
    def frames_failed(self) -> int:
        """Get total failed frames."""
        return self._frames_failed

    @property
    def rtmp_url(self) -> str:
        """Get current RTMP URL."""
        return self._rtmp_url

    @property
    def reconnect_count(self) -> int:
        """Get reconnection attempt count."""
        return self._reconnect_count

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error

    @property
    def uptime(self) -> float:
        """Get streaming uptime in seconds."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def get_statistics(self) -> dict:
        """
        Get streaming statistics.

        Returns:
            Dictionary with streaming metrics
        """
        return {
            "streaming": self._streaming,
            "rtmp_url": self._rtmp_url,
            "frames_pushed": self._frames_pushed,
            "frames_failed": self._frames_failed,
            "bytes_pushed": self._bytes_pushed,
            "reconnect_count": self._reconnect_count,
            "uptime": round(self.uptime, 2),
            "healthy": self.is_healthy(),
            "last_error": self._last_error
        }

    def _create_streamer_process(self, rtmp_url: str) -> subprocess.Popen:
        """
        Create FFmpeg streamer process.

        Uses low-latency configuration for real-time streaming.

        Args:
            rtmp_url: RTMP server URL

        Returns:
            FFmpeg subprocess
        """
        cmd = [
            self._config.ffmpeg_path,
            '-thread_queue_size', '512',  # Input option - must be before -i
            '-f', 'h264',           # Input format
            '-i', '-',              # Read from stdin
            '-c:v', 'copy',         # Copy video (no re-encoding)
            '-fflags', 'nobuffer',  # No buffering
            '-flags', 'low_delay',  # Low delay mode
            '-avioflags', 'direct', # Direct I/O
            '-flush_packets', '1',  # Flush after each packet
            '-use_wallclock_as_timestamps', '1',  # Use wall clock
            '-muxdelay', '0',
            '-muxpreload', '0',
            '-flvflags', 'no_duration_filesize',
            '-rtmp_live', 'live',
            '-rtmp_buffer', '0',
            '-f', 'flv',
            '-an',                  # No audio
            rtmp_url
        ]

        self._log(f"FFmpeg command: {' '.join(cmd)}")

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

    def _stop_process(self) -> None:
        """Stop FFmpeg process."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception as e:
                self._log(f"Error stopping process: {e}")
                try:
                    self._process.kill()
                except:
                    pass
            finally:
                self._process = None

    def _handle_reconnect(self) -> None:
        """Handle reconnection attempt."""
        if self._reconnect_count >= self._config.max_retries:
            self._log("Max reconnection attempts reached")
            self._streaming = False
            return

        self._stop_process()
        time.sleep(self._config.retry_interval)

        if self._rtmp_url:
            try:
                self._reconnect_count += 1
                self._process = self._create_streamer_process(self._rtmp_url)
                self._log(f"Reconnected (attempt {self._reconnect_count})")
            except Exception as e:
                self._log(f"Reconnection failed: {e}")
                self._streaming = False
