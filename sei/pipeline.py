# -*- coding: utf-8 -*-
"""
SEI Streaming Pipeline Module

This module provides the complete SEI video streaming pipeline, integrating
encoding, SEI injection, and dual output (RTMP streaming + event recording).

Processing Flow:
    Raw Video Frame
        │
        ▼
    H.264 Encoder
        │
        ▼
    SEI Injector (adds metadata)
        │
        ▼
    Output Manager
        │
        ├──▶ Buffer (always)
        ├──▶ RTMP Streamer (Mode 1)
        └──▶ MP4 Recorder (Mode 2, on event)

Features:
- Complete encode-inject-stream pipeline
- Dual output: real-time streaming + event recording
- Configurable SEI injection with inference data
- Event trigger evaluation
- Async frame queue for non-blocking operation
- Comprehensive statistics

Usage:
    >>> from sei import SeiStreamingPipeline, SeiConfig, StreamConfig
    >>> sei_config = SeiConfig(enable=True)
    >>> stream_config = StreamConfig(width=1920, height=1080, fps=10)
    >>> pipeline = SeiStreamingPipeline(sei_config=sei_config, stream_config=stream_config)
    >>> pipeline.start("rtmp://server/live/stream")
    >>> for frame, inference_data in inference_loop:
    ...     pipeline.push_frame(frame, inference_data)
    >>> pipeline.stop()
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .config import (
    SeiConfig,
    StreamConfig,
    SeiPipelineConfig,
    OutputConfig,
    BufferConfig,
    RecorderConfig
)
from .interfaces import (
    SeiPayload,
    EncodedFrame,
    SeiStreamCallback,
    DefaultSeiStreamCallback,
    EventHandlerInterface,
    DefaultEventHandler,
    StreamingStatistics,
    LogFunc
)
from .encoder import H264Encoder, SimpleH264Encoder
from .injector import SeiInjector
from .output_manager import OutputManager, OutputStatistics
from .events import SeiEvent, EventTrigger
from .nalutils import get_nal_type, split_nalus, NAL_TYPE_IDR


@dataclass
class PushResult:
    """
    Result of push_frame operation.

    Attributes:
        success: Overall success
        encoded: Frame was encoded
        injected: SEI was injected
        streamed: Frame was streamed
        recorded: Frame was recorded
        event_triggered: Event was triggered
        error: Error message if failed
    """
    success: bool = True
    encoded: bool = False
    injected: bool = False
    streamed: bool = False
    recorded: bool = False
    event_triggered: bool = False
    error: Optional[str] = None


@dataclass
class PipelineStatistics:
    """
    Pipeline statistics container.

    Aggregates all pipeline component statistics.
    """
    frames_received: int = 0
    frames_encoded: int = 0
    frames_with_sei: int = 0
    frames_pushed: int = 0
    frames_failed: int = 0
    events_triggered: int = 0
    recordings_completed: int = 0
    start_time: Optional[float] = None
    last_frame_time: Optional[float] = None

    @property
    def fps(self) -> float:
        """Calculate actual FPS."""
        if not self.start_time or not self.last_frame_time:
            return 0.0
        duration = self.last_frame_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.frames_pushed / duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_received": self.frames_received,
            "frames_encoded": self.frames_encoded,
            "frames_with_sei": self.frames_with_sei,
            "frames_pushed": self.frames_pushed,
            "frames_failed": self.frames_failed,
            "events_triggered": self.events_triggered,
            "recordings_completed": self.recordings_completed,
            "fps": round(self.fps, 2)
        }


class SeiStreamingPipeline:
    """
    Complete SEI Video Streaming Pipeline.

    Integrates encoder, SEI injector, and dual output management
    into a complete processing pipeline.

    Components:
    - H264Encoder: Encodes raw frames to H.264
    - SeiInjector: Injects metadata into H.264 stream
    - OutputManager: Handles streaming and recording

    Attributes:
        sei_config: SEI injection configuration
        stream_config: Streaming configuration
        event_callback: Stream event callbacks
        event_handler: Recording event callbacks

    Example:
        >>> from sei import SeiStreamingPipeline, SeiConfig, StreamConfig
        >>>
        >>> # Configure pipeline
        >>> sei_config = SeiConfig(enable=True)
        >>> stream_config = StreamConfig(width=1920, height=1080, fps=10)
        >>>
        >>> # Create and start pipeline
        >>> pipeline = SeiStreamingPipeline(
        ...     sei_config=sei_config,
        ...     stream_config=stream_config,
        ...     log_func=print
        ... )
        >>> pipeline.start("rtmp://server/live/stream")
        >>>
        >>> # Process frames
        >>> for frame, inference_data in inference_loop:
        ...     result = pipeline.push_frame(frame, inference_data)
        ...     if result.event_triggered:
        ...         print("Event recorded!")
        >>>
        >>> # Stop pipeline
        >>> pipeline.stop()
    """

    def __init__(
        self,
        sei_config: Optional[SeiConfig] = None,
        stream_config: Optional[StreamConfig] = None,
        output_config: Optional[OutputConfig] = None,
        event_callback: Optional[SeiStreamCallback] = None,
        event_handler: Optional[EventHandlerInterface] = None,
        event_trigger_config: Optional[Dict[str, Any]] = None,
        log_func: Optional[LogFunc] = None,
        use_simple_encoder: bool = False
    ):
        """
        Initialize SEI streaming pipeline.

        Args:
            sei_config: SEI injection configuration
            stream_config: Streaming configuration
            output_config: Output (stream + record) configuration
            event_callback: Stream lifecycle callbacks
            event_handler: Recording lifecycle callbacks
            event_trigger_config: Event trigger configuration
            log_func: Logging function
            use_simple_encoder: Use simple encoder (per-frame process)
        """
        # Initialize logger
        self._log = log_func or self._default_log
        self._init_sei_logger()

        # Configuration
        self._sei_config = sei_config or SeiConfig()
        self._stream_config = stream_config or StreamConfig()

        # Build output config if not provided
        if output_config is None:
            output_config = OutputConfig(
                rtmp_enabled=True,
                recording_enabled=True,
                stream_config=self._stream_config,
                recorder_config=RecorderConfig(),
                buffer_config=BufferConfig()
            )
        self._output_config = output_config

        # Callbacks
        self._event_callback = event_callback or DefaultSeiStreamCallback()
        self._event_handler = event_handler or DefaultEventHandler()

        # Create encoder
        if use_simple_encoder:
            self._encoder = SimpleH264Encoder(self._stream_config, self._log)
        else:
            self._encoder = H264Encoder(self._stream_config, self._log)

        # Create injector
        self._injector = SeiInjector(self._sei_config, self._log)

        # Create output manager
        self._output_manager = OutputManager(
            self._output_config,
            self._event_handler,
            self._log
        )

        # Create event trigger
        self._event_trigger = EventTrigger(
            event_trigger_config or {},
            self._log
        )

        # Frame queue for async operation
        self._frame_queue: queue.Queue = queue.Queue(
            maxsize=self._stream_config.buffer_size
        )

        # Thread control
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics
        self._stats = PipelineStatistics()
        self._stats_lock = threading.Lock()

        # State
        self._rtmp_url: Optional[str] = None
        self._frame_index = 0

        self._log("SEI pipeline initialized")

    @staticmethod
    def _default_log(message: str) -> None:
        """Default logging to stdout."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [SEI] {message}")

    def _init_sei_logger(self) -> None:
        """Initialize SEI-specific logger (logs/sei.log)."""
        try:
            # Try to integrate with centralized Logger
            from utils.logger import Logger
            Logger.register_component_logger("sei", "sei.log")
            self._log = Logger.get_logging_method("SEI", "sei")
        except ImportError:
            # Fall back to default if Logger not available
            pass

    def start(self, rtmp_url: Optional[str] = None) -> bool:
        """
        Start the streaming pipeline.

        Args:
            rtmp_url: RTMP server URL (optional if only recording)

        Returns:
            True if started successfully
        """
        if self._running:
            self._log("Pipeline already running")
            return True

        try:
            # Start encoder
            if not self._encoder.start():
                self._log("Encoder start failed")
                return False

            # Start output manager
            if not self._output_manager.start(rtmp_url):
                self._log("Output manager start failed")
                self._encoder.stop()
                return False

            # Start processing thread
            self._stop_event.clear()
            self._running = True
            self._rtmp_url = rtmp_url
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="SeiPipelineThread"
            )
            self._processing_thread.start()

            # Update statistics
            with self._stats_lock:
                self._stats.start_time = time.time()

            self._event_callback.on_stream_started(rtmp_url or "local")
            self._log(f"Pipeline started (rtmp={rtmp_url or 'disabled'})")
            return True

        except Exception as e:
            self._log(f"Pipeline start failed: {e}")
            self._event_callback.on_stream_error(str(e))
            return False

    def stop(self) -> None:
        """Stop the streaming pipeline."""
        self._log("Stopping pipeline...")

        self._running = False
        self._stop_event.set()

        # Clear queue
        try:
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
        except:
            pass

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)

        # Stop components
        self._encoder.stop()
        self._output_manager.stop()

        self._event_callback.on_stream_stopped("Manual stop")
        self._log("Pipeline stopped")

    def push_frame(
        self,
        frame,
        inference_data: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> PushResult:
        """
        Push frame through pipeline (synchronous).

        Encodes frame, injects SEI, and outputs to stream/recording.

        Args:
            frame: OpenCV BGR image (numpy.ndarray)
            inference_data: Model inference results
            custom_data: Additional metadata
            task_id: Task identifier for events

        Returns:
            PushResult with operation status
        """
        if not self._running:
            return PushResult(success=False, error="Pipeline not running")

        result = PushResult()

        try:
            with self._stats_lock:
                self._stats.frames_received += 1

            # Encode frame
            h264_data, is_keyframe = self._encoder.encode_with_keyframe_info(frame)
            if not h264_data:
                with self._stats_lock:
                    self._stats.frames_failed += 1
                return PushResult(success=False, error="Encoding failed")

            result.encoded = True
            with self._stats_lock:
                self._stats.frames_encoded += 1

            # Create SEI payload and inject
            sei_payload = None
            if self._sei_config.enable and (inference_data or custom_data):
                sei_payload = SeiPayload(
                    frame_timestamp=time.time(),
                    inference_data=inference_data,
                    custom_data=custom_data,
                    frame_count=self._frame_index
                )
                h264_data = self._injector.inject(h264_data, sei_payload)
                result.injected = True

                with self._stats_lock:
                    self._stats.frames_with_sei += 1

            # Create encoded frame container
            encoded_frame = EncodedFrame(
                h264_data=h264_data,
                timestamp=time.time(),
                frame_index=self._frame_index,
                sei_payload=sei_payload,
                is_keyframe=is_keyframe,
                width=self._stream_config.width,
                height=self._stream_config.height
            )

            # Output frame
            output_result = self._output_manager.output_frame(encoded_frame)
            result.streamed = output_result.streamed
            result.recorded = output_result.recording

            # Check for event triggers
            if inference_data and task_id:
                event = self._event_trigger.evaluate(inference_data, task_id)
                if event:
                    self._output_manager.trigger_event(
                        event,
                        width=self._stream_config.width,
                        height=self._stream_config.height,
                        fps=self._stream_config.fps
                    )
                    result.event_triggered = True

                    with self._stats_lock:
                        self._stats.events_triggered += 1

            # Update statistics
            with self._stats_lock:
                self._stats.frames_pushed += 1
                self._stats.last_frame_time = time.time()

            self._frame_index += 1

            # Fire callback
            self._event_callback.on_frame_pushed(
                self._frame_index,
                result.injected
            )

            return result

        except Exception as e:
            self._log(f"Push frame error: {e}")
            self._event_callback.on_stream_error(str(e))

            with self._stats_lock:
                self._stats.frames_failed += 1

            return PushResult(success=False, error=str(e))

    def push_frame_async(
        self,
        frame,
        inference_data: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> bool:
        """
        Push frame to async queue for processing.

        Non-blocking operation. Frame will be processed by background thread.

        Args:
            frame: OpenCV BGR image
            inference_data: Model inference results
            custom_data: Additional metadata
            task_id: Task identifier

        Returns:
            True if queued successfully
        """
        if not self._running:
            return False

        try:
            self._frame_queue.put_nowait((frame, inference_data, custom_data, task_id))
            return True
        except queue.Full:
            self._log("Frame queue full")
            return False

    def trigger_event(
        self,
        event_type: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Manually trigger event recording.

        Args:
            event_type: Type of event
            task_id: Task identifier
            metadata: Event metadata

        Returns:
            True if recording started
        """
        event = SeiEvent.create(
            event_type=event_type,
            task_id=task_id,
            metadata=metadata,
            source="manual"
        )

        success = self._output_manager.trigger_event(
            event,
            width=self._stream_config.width,
            height=self._stream_config.height,
            fps=self._stream_config.fps
        )

        if success:
            with self._stats_lock:
                self._stats.events_triggered += 1

        return success

    def end_event(self, event_id: str) -> Optional[str]:
        """
        End event and stop recording.

        Args:
            event_id: Event identifier

        Returns:
            Path to recorded file, or None
        """
        path = self._output_manager.end_event(event_id)
        if path:
            with self._stats_lock:
                self._stats.recordings_completed += 1
        return path

    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    def is_streaming(self) -> bool:
        """Check if actively streaming."""
        return self._output_manager.is_streaming()

    def is_recording(self, event_id: Optional[str] = None) -> bool:
        """Check if recording is active."""
        return self._output_manager.is_recording(event_id)

    def get_statistics(self) -> PipelineStatistics:
        """Get pipeline statistics."""
        with self._stats_lock:
            return PipelineStatistics(
                frames_received=self._stats.frames_received,
                frames_encoded=self._stats.frames_encoded,
                frames_with_sei=self._stats.frames_with_sei,
                frames_pushed=self._stats.frames_pushed,
                frames_failed=self._stats.frames_failed,
                events_triggered=self._stats.events_triggered,
                recordings_completed=self._stats.recordings_completed,
                start_time=self._stats.start_time,
                last_frame_time=self._stats.last_frame_time
            )

    def get_output_statistics(self) -> OutputStatistics:
        """Get output manager statistics."""
        return self._output_manager.get_statistics()

    def set_resolution(self, width: int, height: int) -> None:
        """Update encoding resolution."""
        self._stream_config.width = width
        self._stream_config.height = height
        self._encoder.set_resolution(width, height)

    def enable_sei(self) -> None:
        """Enable SEI injection."""
        self._sei_config.enable = True

    def disable_sei(self) -> None:
        """Disable SEI injection."""
        self._sei_config.enable = False

    def enable_recording(self) -> None:
        """Enable event recording."""
        self._output_manager.set_recording_enabled(True)

    def disable_recording(self) -> None:
        """Disable event recording."""
        self._output_manager.set_recording_enabled(False)

    def _processing_loop(self) -> None:
        """Background processing loop for async frames."""
        self._log("Processing loop started")

        frame_interval = self._stream_config.frame_interval
        last_frame_time = time.time()

        while self._running and not self._stop_event.is_set():
            try:
                # Get frame from queue
                try:
                    frame, inference_data, custom_data, task_id = \
                        self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Rate limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                # Process frame
                self.push_frame(frame, inference_data, custom_data, task_id)
                last_frame_time = time.time()

            except Exception as e:
                self._log(f"Processing loop error: {e}")

        self._log("Processing loop stopped")


def create_pipeline_from_config(
    config: SeiPipelineConfig,
    event_callback: Optional[SeiStreamCallback] = None,
    event_handler: Optional[EventHandlerInterface] = None,
    log_func: Optional[LogFunc] = None
) -> SeiStreamingPipeline:
    """
    Create pipeline from configuration object.

    Args:
        config: Complete pipeline configuration
        event_callback: Stream event callbacks
        event_handler: Recording event callbacks
        log_func: Logging function

    Returns:
        Configured SeiStreamingPipeline instance
    """
    return SeiStreamingPipeline(
        sei_config=config.sei_config,
        stream_config=config.output_config.stream_config,
        output_config=config.output_config,
        event_callback=event_callback,
        event_handler=event_handler,
        log_func=log_func
    )
