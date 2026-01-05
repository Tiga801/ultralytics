# -*- coding: utf-8 -*-
"""
Event-Triggered MP4 Recorder Module

This module provides MP4 recording functionality for event-triggered video
saving. When an event occurs, it captures pre-event frames from the buffer
and continues recording post-event frames until completion.

Features:
- H.264 to MP4 muxing via FFmpeg
- Pre-event frame injection from buffer
- Post-event frame countdown
- Configurable output patterns
- Multiple concurrent recordings support
- Automatic file naming with timestamps

Recording Workflow:
1. Event triggered -> retrieve pre-event frames from buffer
2. Start FFmpeg muxing process
3. Write pre-event frames immediately
4. Continue writing new frames for post_event_frames count
5. Finalize and close file
"""

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from .config import RecorderConfig
from .interfaces import (
    RecorderInterface,
    EncodedFrame,
    RecordingStatistics,
    LogFunc
)


@dataclass
class RecordingSession:
    """
    Active Recording Session State.

    Tracks state for a single recording operation.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of triggering event
        output_path: Target file path
        process: FFmpeg muxer process
        start_time: Recording start timestamp
        frames_written: Count of frames written
        post_event_countdown: Remaining post-event frames
        task_id: Associated task ID
    """
    event_id: str
    event_type: str
    output_path: str
    process: Optional[subprocess.Popen] = None
    start_time: float = 0.0
    frames_written: int = 0
    post_event_countdown: int = 0
    task_id: Optional[str] = None
    width: int = 1920
    height: int = 1080
    fps: float = 10.0


class Mp4Recorder(RecorderInterface):
    """
    Event-Triggered MP4 Recorder.

    Records H.264 encoded frames to MP4 files when events are triggered.
    Uses FFmpeg for H.264 to MP4 container muxing.

    Recording Flow:
        Event Trigger
            │
            ▼
        Retrieve pre-event frames from buffer
            │
            ▼
        Start FFmpeg muxer process (H.264 -> MP4)
            │
            ▼
        Write pre-event frames
            │
            ▼
        Continue writing new frames (post-event countdown)
            │
            ▼
        Finalize and close file

    Attributes:
        config: Recorder configuration
        log: Logging function

    Example:
        >>> from sei.recorder import Mp4Recorder
        >>> from sei.config import RecorderConfig
        >>> config = RecorderConfig(output_dir="recordings")
        >>> recorder = Mp4Recorder(config)
        >>> recorder.start_recording("evt001", "cross_line", pre_event_frames)
        >>> for frame in new_frames:
        ...     if not recorder.write_frame(frame):
        ...         break  # Recording complete
        >>> output_path = recorder.stop_recording("evt001")
    """

    def __init__(
        self,
        config: Optional[RecorderConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize MP4 recorder.

        Args:
            config: Recorder configuration (optional)
            log_func: Logging function (optional)
        """
        self._config = config or RecorderConfig()
        self._log = log_func or (lambda x: None)

        # Active recording sessions
        self._sessions: Dict[str, RecordingSession] = {}
        self._lock = threading.RLock()

        # Statistics
        self._stats = RecordingStatistics()

        # Ensure output directory exists
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)

        self._log(f"MP4 recorder initialized, output_dir={self._config.output_dir}")

    def start_recording(
        self,
        event_id: str,
        event_type: str,
        pre_event_frames: Optional[List[EncodedFrame]] = None,
        task_id: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        fps: float = 10.0
    ) -> bool:
        """
        Start a new recording session.

        Creates an FFmpeg muxer process and writes any pre-event frames.

        Args:
            event_id: Unique event identifier
            event_type: Type of event (for filename)
            pre_event_frames: Buffered frames from before event
            task_id: Associated task ID (for filename)
            width: Video width
            height: Video height
            fps: Frame rate

        Returns:
            True if recording started successfully
        """
        with self._lock:
            # Check concurrent recording limit
            if len(self._sessions) >= self._config.max_concurrent_recordings:
                self._log(f"Max concurrent recordings reached ({self._config.max_concurrent_recordings})")
                return False

            # Check if event already being recorded
            if event_id in self._sessions:
                self._log(f"Recording already active for event {event_id}")
                return False

            try:
                # Generate output path
                output_path = self._generate_output_path(event_id, event_type, task_id)

                # Create muxer process
                process = self._create_muxer_process(output_path, width, height, fps)

                # Create session
                session = RecordingSession(
                    event_id=event_id,
                    event_type=event_type,
                    output_path=output_path,
                    process=process,
                    start_time=time.time(),
                    frames_written=0,
                    post_event_countdown=self._config.post_event_frames,
                    task_id=task_id,
                    width=width,
                    height=height,
                    fps=fps
                )

                self._sessions[event_id] = session
                self._stats.recordings_started += 1

                self._log(f"Started recording: {output_path}")

                # Write pre-event frames if provided
                if pre_event_frames:
                    for frame in pre_event_frames:
                        self._write_frame_to_session(session, frame)
                    self._log(f"Wrote {len(pre_event_frames)} pre-event frames")

                return True

            except Exception as e:
                self._log(f"Failed to start recording: {e}")
                self._stats.recordings_failed += 1
                return False

    def stop_recording(self, event_id: str) -> Optional[str]:
        """
        Stop recording and finalize file.

        Args:
            event_id: Event identifier to stop

        Returns:
            Path to recorded file, or None on failure
        """
        with self._lock:
            session = self._sessions.pop(event_id, None)

            if not session:
                self._log(f"No active recording for event {event_id}")
                return None

            try:
                # Close FFmpeg process
                if session.process:
                    try:
                        session.process.stdin.close()
                        session.process.wait(timeout=10)
                    except Exception as e:
                        self._log(f"Error closing muxer: {e}")
                        try:
                            session.process.kill()
                        except:
                            pass

                # Update statistics
                duration = time.time() - session.start_time
                self._stats.recordings_completed += 1
                self._stats.frames_recorded += session.frames_written
                self._stats.total_duration += duration

                self._log(f"Completed recording: {session.output_path} "
                         f"({session.frames_written} frames, {duration:.1f}s)")

                return session.output_path

            except Exception as e:
                self._log(f"Error stopping recording: {e}")
                self._stats.recordings_failed += 1
                return None

    def write_frame(self, frame: EncodedFrame) -> bool:
        """
        Write frame to all active recordings.

        Decrements post-event countdown for each recording.
        Automatically stops recordings when countdown reaches zero.

        Args:
            frame: Encoded frame to write

        Returns:
            True if any recordings are still active
        """
        with self._lock:
            if not self._sessions:
                return False

            # Track completed recordings for removal
            completed = []

            for event_id, session in self._sessions.items():
                # Write frame
                if self._write_frame_to_session(session, frame):
                    # Decrement countdown
                    session.post_event_countdown -= 1

                    # Check if recording complete
                    if session.post_event_countdown <= 0:
                        completed.append(event_id)

                    # Check max duration
                    elapsed = time.time() - session.start_time
                    if elapsed >= self._config.max_recording_duration:
                        completed.append(event_id)

            # Stop completed recordings
            for event_id in completed:
                self.stop_recording(event_id)

            return len(self._sessions) > 0

    def write_frame_to_event(self, event_id: str, frame: EncodedFrame) -> bool:
        """
        Write frame to specific recording.

        Args:
            event_id: Target event recording
            frame: Encoded frame to write

        Returns:
            True if frame was written
        """
        with self._lock:
            session = self._sessions.get(event_id)
            if not session:
                return False

            return self._write_frame_to_session(session, frame)

    def is_recording(self, event_id: Optional[str] = None) -> bool:
        """
        Check recording state.

        Args:
            event_id: Specific event to check (None = any active)

        Returns:
            True if recording is active
        """
        with self._lock:
            if event_id:
                return event_id in self._sessions
            return len(self._sessions) > 0

    def get_active_recordings(self) -> List[str]:
        """Get list of active recording event IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def get_statistics(self) -> RecordingStatistics:
        """Get recording statistics."""
        return RecordingStatistics(
            recordings_started=self._stats.recordings_started,
            recordings_completed=self._stats.recordings_completed,
            recordings_failed=self._stats.recordings_failed,
            frames_recorded=self._stats.frames_recorded,
            bytes_recorded=self._stats.bytes_recorded,
            total_duration=self._stats.total_duration
        )

    def stop_all(self) -> List[str]:
        """
        Stop all active recordings.

        Returns:
            List of completed recording paths
        """
        with self._lock:
            paths = []
            event_ids = list(self._sessions.keys())
            for event_id in event_ids:
                path = self.stop_recording(event_id)
                if path:
                    paths.append(path)
            return paths

    def _generate_output_path(
        self,
        event_id: str,
        event_type: str,
        task_id: Optional[str] = None
    ) -> str:
        """Generate output file path from pattern."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = self._config.filename_pattern.format(
            event_id=event_id,
            event_type=event_type,
            task_id=task_id or "unknown",
            timestamp=timestamp
        )

        # Ensure correct extension
        if not filename.endswith(f".{self._config.container}"):
            filename = f"{filename}.{self._config.container}"

        return os.path.join(self._config.output_dir, filename)

    def _create_muxer_process(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float
    ) -> subprocess.Popen:
        """
        Create FFmpeg muxer process for H.264 -> MP4.

        Args:
            output_path: Output file path
            width: Video width
            height: Video height
            fps: Frame rate

        Returns:
            FFmpeg subprocess
        """
        cmd = [
            self._config.ffmpeg_path,
            '-y',  # Overwrite output
            '-f', 'h264',  # Input format
            '-framerate', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'copy',  # Copy video stream (no re-encoding)
            '-movflags', '+faststart',  # Enable fast start for streaming
            '-f', self._config.container,
            output_path
        ]

        self._log(f"Muxer command: {' '.join(cmd)}")

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

    def _write_frame_to_session(
        self,
        session: RecordingSession,
        frame: EncodedFrame
    ) -> bool:
        """Write frame to recording session."""
        if not session.process or not session.process.stdin:
            return False

        try:
            session.process.stdin.write(frame.h264_data)
            session.process.stdin.flush()
            session.frames_written += 1
            self._stats.bytes_recorded += len(frame.h264_data)
            return True

        except BrokenPipeError:
            self._log(f"Broken pipe for recording {session.event_id}")
            return False

        except Exception as e:
            self._log(f"Error writing frame: {e}")
            return False
