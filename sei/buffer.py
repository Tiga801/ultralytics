# -*- coding: utf-8 -*-
"""
Frame Ring Buffer Module

This module provides thread-safe ring buffer implementations for storing
encoded video frames. The buffer supports event-triggered recording by
maintaining a sliding window of recent frames.

Features:
- Thread-safe operations with RLock
- Automatic overflow handling (drop oldest)
- Pre-event frame retrieval for recording
- Timestamp-based frame queries
- Memory-efficient deque-based storage

Usage:
    >>> from sei.buffer import FrameRingBuffer
    >>> from sei.config import BufferConfig
    >>> config = BufferConfig(capacity=150, pre_event_frames=50)
    >>> buffer = FrameRingBuffer(config)
    >>> buffer.push(encoded_frame)
    >>> pre_event = buffer.get_pre_event_frames()
"""

import threading
from collections import deque
from typing import List, Optional, Any, Iterator

from .config import BufferConfig
from .interfaces import BufferInterface, EncodedFrame, LogFunc


class FrameRingBuffer(BufferInterface):
    """
    Thread-Safe Ring Buffer for Encoded Frames.

    Implements a fixed-capacity circular buffer optimized for video
    frame storage. Automatically discards oldest frames when capacity
    is reached.

    Memory Layout:
        [oldest...][pre-event frames][newest]
                   |<- pre_event_size ->|
                   ^
                   Retrieved on event trigger

    Attributes:
        capacity: Maximum buffer size
        pre_event_size: Frames to retain for pre-event capture

    Example:
        >>> config = BufferConfig(capacity=150, pre_event_frames=50)
        >>> buffer = FrameRingBuffer(config)
        >>> for frame in encoded_frames:
        ...     buffer.push(frame)
        >>> pre_event = buffer.get_pre_event_frames()
        >>> len(pre_event) <= 50
        True
    """

    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize ring buffer.

        Args:
            config: Buffer configuration (optional, uses defaults)
            log_func: Logging function (optional)
        """
        self._config = config or BufferConfig()
        self._log = log_func or (lambda x: None)

        self._capacity = self._config.capacity
        self._pre_event_size = self._config.pre_event_frames

        # Use deque with maxlen for automatic overflow handling
        self._buffer: deque = deque(maxlen=self._capacity)
        self._lock = threading.RLock()

        # Statistics
        self._total_pushed = 0
        self._total_dropped = 0

    def push(self, item: EncodedFrame) -> bool:
        """
        Push encoded frame to buffer.

        If buffer is at capacity, oldest frame is automatically removed.

        Args:
            item: Encoded frame to store

        Returns:
            True (always succeeds with drop_oldest strategy)
        """
        with self._lock:
            was_full = len(self._buffer) >= self._capacity

            # deque with maxlen handles overflow automatically
            self._buffer.append(item)

            self._total_pushed += 1
            if was_full:
                self._total_dropped += 1

            return True

    def pop(self) -> Optional[EncodedFrame]:
        """
        Pop oldest frame from buffer.

        Returns:
            Oldest frame, or None if buffer is empty
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def get_recent(self, count: int) -> List[EncodedFrame]:
        """
        Get N most recent frames without removing.

        Args:
            count: Number of frames to retrieve

        Returns:
            List of recent frames (newest last), may be fewer than requested
        """
        with self._lock:
            if count <= 0:
                return []

            # Get last 'count' items
            if count >= len(self._buffer):
                return list(self._buffer)

            return list(self._buffer)[-count:]

    def get_pre_event_frames(self) -> List[EncodedFrame]:
        """
        Get frames for pre-event capture.

        Returns the configured number of most recent frames,
        suitable for prepending to event-triggered recordings.

        Returns:
            List of pre-event frames (oldest first)
        """
        return self.get_recent(self._pre_event_size)

    def get_since_timestamp(self, timestamp: float) -> List[EncodedFrame]:
        """
        Get all frames since a given timestamp.

        Useful for retrieving frames from a specific point in time.

        Args:
            timestamp: Cutoff timestamp (frames after this are returned)

        Returns:
            List of frames with timestamp > given timestamp
        """
        with self._lock:
            result = []
            for frame in self._buffer:
                if frame.timestamp > timestamp:
                    result.append(frame)
            return result

    def get_since_index(self, frame_index: int) -> List[EncodedFrame]:
        """
        Get all frames since a given frame index.

        Args:
            frame_index: Cutoff frame index (frames after this are returned)

        Returns:
            List of frames with frame_index > given index
        """
        with self._lock:
            result = []
            for frame in self._buffer:
                if frame.frame_index > frame_index:
                    result.append(frame)
            return result

    def get_keyframe_and_following(self) -> List[EncodedFrame]:
        """
        Get frames starting from most recent keyframe.

        Useful for starting recordings from a valid decode point.

        Returns:
            List of frames starting from most recent keyframe
        """
        with self._lock:
            # Find most recent keyframe
            frames = list(self._buffer)
            keyframe_idx = -1

            for i in range(len(frames) - 1, -1, -1):
                if frames[i].is_keyframe:
                    keyframe_idx = i
                    break

            if keyframe_idx < 0:
                # No keyframe found, return all (decoder may handle)
                return frames

            return frames[keyframe_idx:]

    def peek_oldest(self) -> Optional[EncodedFrame]:
        """
        Peek at oldest frame without removing.

        Returns:
            Oldest frame, or None if empty
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[0]

    def peek_newest(self) -> Optional[EncodedFrame]:
        """
        Peek at newest frame without removing.

        Returns:
            Newest frame, or None if empty
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def clear(self) -> None:
        """Remove all frames from buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[EncodedFrame]:
        """Iterate over frames (oldest to newest)."""
        with self._lock:
            # Return copy to avoid iteration issues
            return iter(list(self._buffer))

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def pre_event_size(self) -> int:
        """Get pre-event frame count."""
        return self._pre_event_size

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self._capacity

    @property
    def total_pushed(self) -> int:
        """Get total frames pushed (including dropped)."""
        return self._total_pushed

    @property
    def total_dropped(self) -> int:
        """Get total frames dropped due to overflow."""
        return self._total_dropped

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer metrics
        """
        with self._lock:
            oldest = self._buffer[0] if self._buffer else None
            newest = self._buffer[-1] if self._buffer else None

            return {
                "capacity": self._capacity,
                "current_size": len(self._buffer),
                "pre_event_size": self._pre_event_size,
                "total_pushed": self._total_pushed,
                "total_dropped": self._total_dropped,
                "oldest_timestamp": oldest.timestamp if oldest else None,
                "newest_timestamp": newest.timestamp if newest else None,
                "oldest_frame_index": oldest.frame_index if oldest else None,
                "newest_frame_index": newest.frame_index if newest else None
            }


class RawFrameBuffer:
    """
    Simple Buffer for Raw (Unencoded) Frames.

    Stores raw numpy arrays for cases where encoding happens
    at recording time rather than streaming time.

    Note: Uses more memory than EncodedFrame buffer.
    """

    def __init__(
        self,
        capacity: int = 150,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize raw frame buffer.

        Args:
            capacity: Maximum frames to store
            log_func: Logging function
        """
        self._capacity = capacity
        self._log = log_func or (lambda x: None)
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()

    def push(self, frame: Any, timestamp: float, frame_index: int) -> bool:
        """
        Push raw frame to buffer.

        Args:
            frame: Raw frame (numpy.ndarray)
            timestamp: Frame timestamp
            frame_index: Frame sequence number

        Returns:
            True on success
        """
        with self._lock:
            self._buffer.append({
                "frame": frame,
                "timestamp": timestamp,
                "frame_index": frame_index
            })
            return True

    def get_recent(self, count: int) -> List[dict]:
        """Get N most recent frames."""
        with self._lock:
            if count >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-count:]

    def clear(self) -> None:
        """Clear all frames."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Get current size."""
        with self._lock:
            return len(self._buffer)
