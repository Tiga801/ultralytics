# -*- coding: utf-8 -*-
"""
Output Manager Module

This module provides the OutputManager class that coordinates dual output:
RTMP streaming and event-triggered MP4 recording.

Features:
- Unified output interface for encoded frames
- Automatic routing to streaming and recording
- Event-triggered recording coordination
- Buffer management for pre-event capture
- Combined statistics collection

Data Flow:
    EncodedFrame
        │
        ▼
    OutputManager
        │
        ├──▶ FrameRingBuffer (always)
        │
        ├──▶ RtmpStreamer (if streaming enabled)
        │
        └──▶ Mp4Recorder (when event active)

Usage:
    >>> from sei.output_manager import OutputManager
    >>> from sei.config import OutputConfig
    >>> config = OutputConfig(rtmp_enabled=True, recording_enabled=True)
    >>> manager = OutputManager(config)
    >>> manager.start(rtmp_url="rtmp://server/live/stream")
    >>> for frame in frames:
    ...     result = manager.output_frame(frame)
    >>> manager.stop()
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from .config import OutputConfig, StreamConfig, RecorderConfig, BufferConfig
from .interfaces import (
    EncodedFrame,
    OutputResult,
    StreamingStatistics,
    RecordingStatistics,
    EventHandlerInterface,
    DefaultEventHandler,
    LogFunc
)
from .buffer import FrameRingBuffer
from .recorder import Mp4Recorder
from .events import SeiEvent


@dataclass
class OutputStatistics:
    """
    Combined Output Statistics.

    Aggregates streaming and recording statistics.
    """
    streaming: StreamingStatistics = field(default_factory=StreamingStatistics)
    recording: RecordingStatistics = field(default_factory=RecordingStatistics)
    buffer_size: int = 0
    active_recordings: int = 0
    total_frames_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "streaming": self.streaming.to_dict(),
            "recording": self.recording.to_dict(),
            "buffer_size": self.buffer_size,
            "active_recordings": self.active_recordings,
            "total_frames_processed": self.total_frames_processed
        }


class OutputManager:
    """
    Dual Output Coordinator.

    Coordinates RTMP streaming and event-triggered recording,
    managing frame routing, buffering, and event handling.

    Components:
    - FrameRingBuffer: Always active, stores recent frames
    - RtmpStreamer: Real-time streaming (Mode 1)
    - Mp4Recorder: Event-triggered recording (Mode 2)

    Attributes:
        config: Output configuration
        event_handler: Event lifecycle callbacks

    Example:
        >>> config = OutputConfig(
        ...     rtmp_enabled=True,
        ...     recording_enabled=True
        ... )
        >>> manager = OutputManager(config, log_func=print)
        >>> manager.start("rtmp://server/live/stream")
        >>>
        >>> # Output frames
        >>> for frame in encoded_frames:
        ...     result = manager.output_frame(frame)
        ...     if not result.success:
        ...         print(f"Output failed: {result.error}")
        >>>
        >>> # Trigger event recording
        >>> event = SeiEvent.create("cross_line", "task001")
        >>> manager.trigger_event(event)
        >>>
        >>> # Continue outputting frames (will be recorded)
        >>> for frame in more_frames:
        ...     manager.output_frame(frame)
        >>>
        >>> manager.stop()
    """

    def __init__(
        self,
        config: Optional[OutputConfig] = None,
        event_handler: Optional[EventHandlerInterface] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize output manager.

        Args:
            config: Output configuration
            event_handler: Event lifecycle callbacks
            log_func: Logging function
        """
        self._config = config or OutputConfig()
        self._event_handler = event_handler or DefaultEventHandler()
        self._log = log_func or (lambda x: None)

        # Initialize components
        self._buffer = FrameRingBuffer(
            self._config.buffer_config,
            log_func
        )

        self._recorder = Mp4Recorder(
            self._config.recorder_config,
            log_func
        )

        # Streamer is imported dynamically to avoid circular dependency
        self._streamer = None
        self._streamer_class = None

        # State
        self._running = False
        self._rtmp_url: Optional[str] = None
        self._lock = threading.RLock()

        # Statistics
        self._streaming_stats = StreamingStatistics()
        self._total_frames = 0

        # Active events
        self._active_events: Dict[str, SeiEvent] = {}

        self._log("Output manager initialized")

    def _get_streamer(self):
        """Lazy load streamer to avoid circular imports."""
        if self._streamer is None:
            from .streamer import RtmpStreamer
            self._streamer = RtmpStreamer(
                self._config.stream_config,
                self._log
            )
        return self._streamer

    def start(self, rtmp_url: Optional[str] = None) -> bool:
        """
        Start output manager.

        Args:
            rtmp_url: RTMP server URL (optional if streaming disabled)

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._running:
                return True

            self._running = True
            self._rtmp_url = rtmp_url

            # Start streaming if enabled and URL provided
            if self._config.rtmp_enabled and rtmp_url:
                streamer = self._get_streamer()
                if not streamer.start(rtmp_url):
                    self._log("Failed to start RTMP streaming")
                    # Continue anyway - recording may still work

            # Update statistics
            self._streaming_stats.start_time = time.time()

            self._log(f"Output manager started (streaming={self._config.rtmp_enabled}, "
                     f"recording={self._config.recording_enabled})")

            return True

    def stop(self) -> None:
        """Stop output manager and all components."""
        with self._lock:
            self._running = False

            # Stop all active recordings
            if self._config.recording_enabled:
                self._recorder.stop_all()

            # Stop streaming
            if self._streamer is not None:
                self._streamer.stop()

            # Clear buffer
            self._buffer.clear()

            self._log("Output manager stopped")

    def output_frame(self, frame: EncodedFrame) -> OutputResult:
        """
        Route encoded frame to all outputs.

        Always buffers the frame. Routes to streaming and active
        recordings based on configuration and state.

        Args:
            frame: Encoded frame to output

        Returns:
            OutputResult indicating success/failure for each output
        """
        if not self._running:
            return OutputResult(success=False, error="Not running")

        result = OutputResult(success=True)

        with self._lock:
            self._total_frames += 1

            # Always buffer frame (for pre-event capture)
            self._buffer.push(frame)
            result.buffered = True

            # Stream if enabled
            if self._config.rtmp_enabled and self._streamer is not None:
                if self._streamer.is_streaming():
                    if self._streamer.push_frame(frame.h264_data):
                        result.streamed = True
                        self._streaming_stats.frames_pushed += 1
                        self._streaming_stats.bytes_pushed += len(frame.h264_data)
                        self._streaming_stats.last_frame_time = time.time()
                        if frame.has_sei():
                            self._streaming_stats.frames_with_sei += 1
                    else:
                        self._streaming_stats.frames_failed += 1

            # Write to active recordings
            if self._config.recording_enabled and self._recorder.is_recording():
                if self._recorder.write_frame(frame):
                    result.recording = True

        return result

    def trigger_event(
        self,
        event: SeiEvent,
        width: int = 1920,
        height: int = 1080,
        fps: float = 10.0
    ) -> bool:
        """
        Trigger event recording.

        Retrieves pre-event frames from buffer and starts recording.

        Args:
            event: Event that triggered recording
            width: Video width for recording
            height: Video height for recording
            fps: Frame rate for recording

        Returns:
            True if recording started successfully
        """
        if not self._config.recording_enabled:
            self._log("Recording not enabled")
            return False

        with self._lock:
            # Check if already recording this event
            if event.event_id in self._active_events:
                return True

            # Get pre-event frames from buffer
            pre_event_frames = self._buffer.get_pre_event_frames()

            # Start recording
            success = self._recorder.start_recording(
                event_id=event.event_id,
                event_type=event.event_type,
                pre_event_frames=pre_event_frames,
                task_id=event.task_id,
                width=width,
                height=height,
                fps=fps
            )

            if success:
                self._active_events[event.event_id] = event
                self._event_handler.on_event_triggered(event)
                self._log(f"Event triggered: {event.event_type} ({event.event_id})")
            else:
                self._log(f"Failed to start recording for event {event.event_id}")

            return success

    def end_event(self, event_id: str) -> Optional[str]:
        """
        End event and stop recording.

        Args:
            event_id: Event identifier to end

        Returns:
            Path to recorded file, or None if not found
        """
        with self._lock:
            event = self._active_events.pop(event_id, None)

            output_path = self._recorder.stop_recording(event_id)

            if event:
                self._event_handler.on_event_ended(event)
                if output_path:
                    self._event_handler.on_recording_completed(event_id, output_path)

            return output_path

    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    def is_streaming(self) -> bool:
        """Check if actively streaming."""
        if not self._config.rtmp_enabled or self._streamer is None:
            return False
        return self._streamer.is_streaming()

    def is_recording(self, event_id: Optional[str] = None) -> bool:
        """Check if recording is active."""
        if not self._config.recording_enabled:
            return False
        return self._recorder.is_recording(event_id)

    def get_active_events(self) -> List[str]:
        """Get list of active event IDs."""
        with self._lock:
            return list(self._active_events.keys())

    def get_statistics(self) -> OutputStatistics:
        """Get combined output statistics."""
        with self._lock:
            return OutputStatistics(
                streaming=StreamingStatistics(
                    frames_encoded=self._streaming_stats.frames_encoded,
                    frames_pushed=self._streaming_stats.frames_pushed,
                    frames_with_sei=self._streaming_stats.frames_with_sei,
                    frames_failed=self._streaming_stats.frames_failed,
                    bytes_pushed=self._streaming_stats.bytes_pushed,
                    start_time=self._streaming_stats.start_time,
                    last_frame_time=self._streaming_stats.last_frame_time
                ),
                recording=self._recorder.get_statistics(),
                buffer_size=len(self._buffer),
                active_recordings=len(self._recorder.get_active_recordings()),
                total_frames_processed=self._total_frames
            )

    def set_streaming_enabled(self, enabled: bool) -> None:
        """Enable or disable streaming."""
        self._config.rtmp_enabled = enabled
        if not enabled and self._streamer is not None:
            self._streamer.stop()

    def set_recording_enabled(self, enabled: bool) -> None:
        """Enable or disable recording."""
        self._config.recording_enabled = enabled
        if not enabled:
            self._recorder.stop_all()

    @property
    def buffer(self) -> FrameRingBuffer:
        """Get buffer instance."""
        return self._buffer

    @property
    def recorder(self) -> Mp4Recorder:
        """Get recorder instance."""
        return self._recorder
