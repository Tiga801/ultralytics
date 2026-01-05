# -*- coding: utf-8 -*-
"""
SEI Video Streaming Interfaces Module

This module defines abstract interfaces for the SEI video streaming pipeline,
enabling modular design and easy component replacement.

Interfaces:
- H264EncoderInterface: Video encoding
- SeiInjectorInterface: SEI data injection
- StreamerInterface: RTMP streaming
- RecorderInterface: Event-triggered recording
- BufferInterface: Frame buffer operations
- EventHandlerInterface: Event callback handling

Data Classes:
- SeiPayload: SEI data payload container
- EncodedFrame: Encoded frame with metadata
- StreamingStatistics: Streaming statistics
- OutputResult: Frame output operation result
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
import json


# Type alias for logging function
LogFunc = Callable[[str], None]


@dataclass
class SeiPayload:
    """
    SEI Data Payload Container.

    Contains metadata to be embedded in SEI NAL units, including
    timestamps, inference results, and custom data.

    Attributes:
        frame_timestamp: Frame capture timestamp
        inference_data: Detection/tracking results from model
        custom_data: Additional user-defined metadata
        frame_count: Frame sequence number
        injection_time: Time when SEI was injected
    """
    frame_timestamp: float
    inference_data: Optional[Dict[str, Any]] = None
    custom_data: Optional[Dict[str, Any]] = None
    frame_count: int = 0
    injection_time: Optional[float] = None

    def __post_init__(self):
        """Set injection time if not provided."""
        if self.injection_time is None:
            self.injection_time = datetime.now().timestamp()

    def to_json(self) -> str:
        """
        Convert payload to JSON string.

        Returns compact JSON with abbreviated keys for efficiency.
        """
        data = {
            "ts": self.frame_timestamp,
            "it": self.injection_time,
            "fc": self.frame_count
        }

        if self.inference_data:
            data["inf"] = self.inference_data

        if self.custom_data:
            data["cus"] = self.custom_data

        return json.dumps(data, separators=(',', ':'))

    def to_bytes(self) -> bytes:
        """Convert payload to UTF-8 encoded bytes."""
        return self.to_json().encode('utf-8')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary representation."""
        return {
            "frame_timestamp": self.frame_timestamp,
            "injection_time": self.injection_time,
            "frame_count": self.frame_count,
            "inference_data": self.inference_data,
            "custom_data": self.custom_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeiPayload":
        """Create payload from dictionary."""
        return cls(
            frame_timestamp=data.get("ts", data.get("frame_timestamp", 0.0)),
            inference_data=data.get("inf", data.get("inference_data")),
            custom_data=data.get("cus", data.get("custom_data")),
            frame_count=data.get("fc", data.get("frame_count", 0)),
            injection_time=data.get("it", data.get("injection_time"))
        )


@dataclass
class EncodedFrame:
    """
    Encoded Frame Container.

    Stores H.264 encoded frame data along with metadata for
    streaming and recording operations.

    Attributes:
        h264_data: Raw H.264 encoded bytes
        timestamp: Frame timestamp
        frame_index: Sequential frame number
        sei_payload: Associated SEI payload (if any)
        is_keyframe: Whether this is an IDR frame
        width: Frame width (for validation)
        height: Frame height (for validation)
    """
    h264_data: bytes
    timestamp: float
    frame_index: int
    sei_payload: Optional[SeiPayload] = None
    is_keyframe: bool = False
    width: int = 0
    height: int = 0

    @property
    def size(self) -> int:
        """Get encoded data size in bytes."""
        return len(self.h264_data)

    def has_sei(self) -> bool:
        """Check if frame has SEI data."""
        return self.sei_payload is not None


@dataclass
class StreamingStatistics:
    """
    Streaming Statistics Container.

    Tracks streaming performance metrics for monitoring and debugging.

    Attributes:
        frames_encoded: Total frames successfully encoded
        frames_pushed: Total frames pushed to stream
        frames_with_sei: Frames containing SEI data
        frames_failed: Failed frame operations
        bytes_pushed: Total bytes transmitted
        start_time: Streaming start timestamp
        last_frame_time: Most recent frame timestamp
    """
    frames_encoded: int = 0
    frames_pushed: int = 0
    frames_with_sei: int = 0
    frames_failed: int = 0
    bytes_pushed: int = 0
    start_time: Optional[float] = None
    last_frame_time: Optional[float] = None

    @property
    def fps(self) -> float:
        """Calculate actual frames per second."""
        if not self.start_time or not self.last_frame_time:
            return 0.0
        duration = self.last_frame_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.frames_pushed / duration

    @property
    def success_rate(self) -> float:
        """Calculate frame push success rate."""
        total = self.frames_pushed + self.frames_failed
        if total == 0:
            return 1.0
        return self.frames_pushed / total

    @property
    def duration(self) -> float:
        """Get streaming duration in seconds."""
        if not self.start_time or not self.last_frame_time:
            return 0.0
        return self.last_frame_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_encoded": self.frames_encoded,
            "frames_pushed": self.frames_pushed,
            "frames_with_sei": self.frames_with_sei,
            "frames_failed": self.frames_failed,
            "bytes_pushed": self.bytes_pushed,
            "fps": round(self.fps, 2),
            "success_rate": round(self.success_rate, 4),
            "duration": round(self.duration, 2)
        }


@dataclass
class RecordingStatistics:
    """
    Recording Statistics Container.

    Tracks recording metrics for event-triggered video saving.

    Attributes:
        recordings_started: Total recordings initiated
        recordings_completed: Successfully completed recordings
        recordings_failed: Failed recordings
        frames_recorded: Total frames written to files
        bytes_recorded: Total bytes written
        total_duration: Combined recording duration
    """
    recordings_started: int = 0
    recordings_completed: int = 0
    recordings_failed: int = 0
    frames_recorded: int = 0
    bytes_recorded: int = 0
    total_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recordings_started": self.recordings_started,
            "recordings_completed": self.recordings_completed,
            "recordings_failed": self.recordings_failed,
            "frames_recorded": self.frames_recorded,
            "bytes_recorded": self.bytes_recorded,
            "total_duration": round(self.total_duration, 2)
        }


@dataclass
class OutputResult:
    """
    Frame Output Operation Result.

    Reports the outcome of pushing a frame through the output pipeline.

    Attributes:
        success: Overall operation success
        streamed: Frame was pushed to RTMP stream
        buffered: Frame was added to buffer
        recording: Frame was written to active recording
        error: Error message if operation failed
    """
    success: bool = True
    streamed: bool = False
    buffered: bool = False
    recording: bool = False
    error: Optional[str] = None


class H264EncoderInterface(ABC):
    """
    H.264 Encoder Interface.

    Defines the contract for video frame encoding implementations.
    """

    @abstractmethod
    def encode(self, frame: Any) -> Optional[bytes]:
        """
        Encode a single video frame.

        Args:
            frame: OpenCV image (numpy.ndarray, BGR format)

        Returns:
            H.264 encoded bytes, or None on failure
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """
        Start the encoder.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the encoder and release resources."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check encoder running state.

        Returns:
            True if encoder is active
        """
        pass

    @abstractmethod
    def set_resolution(self, width: int, height: int) -> None:
        """
        Set output resolution.

        May trigger encoder restart if resolution changes.

        Args:
            width: Video width in pixels
            height: Video height in pixels
        """
        pass


class SeiInjectorInterface(ABC):
    """
    SEI Injector Interface.

    Defines the contract for SEI data injection into H.264 streams.
    """

    @abstractmethod
    def inject(self, h264_data: bytes, payload: SeiPayload) -> bytes:
        """
        Inject SEI data into H.264 stream.

        Args:
            h264_data: Original H.264 encoded data
            payload: SEI payload to inject

        Returns:
            H.264 data with SEI injected
        """
        pass

    @abstractmethod
    def create_sei_nal_unit(self, payload: SeiPayload) -> bytes:
        """
        Create standalone SEI NAL unit.

        Args:
            payload: SEI payload

        Returns:
            SEI NAL unit bytes
        """
        pass

    @abstractmethod
    def set_uuid(self, uuid_bytes: bytes) -> None:
        """
        Set SEI UUID identifier.

        Args:
            uuid_bytes: 16-byte UUID
        """
        pass


class StreamerInterface(ABC):
    """
    RTMP Streamer Interface.

    Defines the contract for RTMP streaming implementations.
    """

    @abstractmethod
    def start(self, rtmp_url: str) -> bool:
        """
        Start streaming to RTMP server.

        Args:
            rtmp_url: RTMP server URL

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and close connection."""
        pass

    @abstractmethod
    def push_frame(self, frame_data: bytes) -> bool:
        """
        Push H.264 frame to stream.

        Args:
            frame_data: H.264 encoded frame

        Returns:
            True if pushed successfully
        """
        pass

    @abstractmethod
    def is_streaming(self) -> bool:
        """
        Check streaming state.

        Returns:
            True if actively streaming
        """
        pass


class RecorderInterface(ABC):
    """
    Event-Triggered Recorder Interface.

    Defines the contract for MP4 recording implementations.
    """

    @abstractmethod
    def start_recording(
        self,
        event_id: str,
        event_type: str,
        pre_event_frames: Optional[List[EncodedFrame]] = None
    ) -> bool:
        """
        Start a new recording session.

        Args:
            event_id: Unique event identifier
            event_type: Type of triggering event
            pre_event_frames: Buffered frames from before event

        Returns:
            True if recording started successfully
        """
        pass

    @abstractmethod
    def stop_recording(self, event_id: str) -> Optional[str]:
        """
        Stop recording and finalize file.

        Args:
            event_id: Event identifier to stop

        Returns:
            Path to recorded file, or None on failure
        """
        pass

    @abstractmethod
    def write_frame(self, frame: EncodedFrame) -> bool:
        """
        Write encoded frame to active recording.

        Args:
            frame: Encoded frame to write

        Returns:
            True if written successfully
        """
        pass

    @abstractmethod
    def is_recording(self, event_id: Optional[str] = None) -> bool:
        """
        Check recording state.

        Args:
            event_id: Optional specific event to check

        Returns:
            True if recording is active
        """
        pass


class BufferInterface(ABC):
    """
    Ring Buffer Interface.

    Defines the contract for frame buffer implementations.
    """

    @abstractmethod
    def push(self, item: Any) -> bool:
        """
        Push item to buffer.

        Args:
            item: Item to store

        Returns:
            True if stored successfully
        """
        pass

    @abstractmethod
    def pop(self) -> Optional[Any]:
        """
        Pop oldest item from buffer.

        Returns:
            Oldest item, or None if empty
        """
        pass

    @abstractmethod
    def get_recent(self, count: int) -> List[Any]:
        """
        Get N most recent items without removing.

        Args:
            count: Number of items to retrieve

        Returns:
            List of recent items (may be fewer than requested)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from buffer."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get current buffer size."""
        pass


class SeiStreamCallback(ABC):
    """
    SEI Stream Event Callback Interface.

    Defines callbacks for streaming lifecycle events.
    """

    @abstractmethod
    def on_stream_started(self, rtmp_url: str) -> None:
        """
        Called when streaming starts.

        Args:
            rtmp_url: RTMP server URL
        """
        pass

    @abstractmethod
    def on_stream_stopped(self, reason: str) -> None:
        """
        Called when streaming stops.

        Args:
            reason: Stop reason description
        """
        pass

    @abstractmethod
    def on_frame_pushed(self, frame_index: int, with_sei: bool) -> None:
        """
        Called after each frame is pushed.

        Args:
            frame_index: Frame sequence number
            with_sei: Whether frame contained SEI
        """
        pass

    @abstractmethod
    def on_stream_error(self, error: str) -> None:
        """
        Called when a streaming error occurs.

        Args:
            error: Error description
        """
        pass


class EventHandlerInterface(ABC):
    """
    Event Handler Interface.

    Defines callbacks for event-triggered recording lifecycle.
    """

    @abstractmethod
    def on_event_triggered(self, event: Any) -> None:
        """
        Called when an event is triggered.

        Args:
            event: Event object
        """
        pass

    @abstractmethod
    def on_event_ended(self, event: Any) -> None:
        """
        Called when an event ends.

        Args:
            event: Event object
        """
        pass

    @abstractmethod
    def on_recording_started(self, event_id: str, output_path: str) -> None:
        """
        Called when recording starts.

        Args:
            event_id: Event identifier
            output_path: Recording file path
        """
        pass

    @abstractmethod
    def on_recording_completed(self, event_id: str, output_path: str) -> None:
        """
        Called when recording completes successfully.

        Args:
            event_id: Event identifier
            output_path: Final recording file path
        """
        pass


class DefaultSeiStreamCallback(SeiStreamCallback):
    """
    Default SEI Stream Callback Implementation.

    Provides empty implementations for all callbacks.
    Can be subclassed to override specific methods.
    """

    def on_stream_started(self, rtmp_url: str) -> None:
        """No-op implementation."""
        pass

    def on_stream_stopped(self, reason: str) -> None:
        """No-op implementation."""
        pass

    def on_frame_pushed(self, frame_index: int, with_sei: bool) -> None:
        """No-op implementation."""
        pass

    def on_stream_error(self, error: str) -> None:
        """No-op implementation."""
        pass


class DefaultEventHandler(EventHandlerInterface):
    """
    Default Event Handler Implementation.

    Provides empty implementations for all callbacks.
    Can be subclassed to override specific methods.
    """

    def on_event_triggered(self, event: Any) -> None:
        """No-op implementation."""
        pass

    def on_event_ended(self, event: Any) -> None:
        """No-op implementation."""
        pass

    def on_recording_started(self, event_id: str, output_path: str) -> None:
        """No-op implementation."""
        pass

    def on_recording_completed(self, event_id: str, output_path: str) -> None:
        """No-op implementation."""
        pass
