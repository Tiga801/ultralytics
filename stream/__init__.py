"""Standalone video stream module for ultralytics.

This module provides stream loading functionality that is completely
independent of the ultralytics package, while maintaining interface
compatibility with the original LoadStreams class.

Features:
    - Load video streams from RTSP, RTMP, HTTP, HTTPS, TCP sources
    - Load local video files
    - Multi-stream support with threading
    - Configurable frame buffering
    - Auto-reconnection on stream loss
    - Context manager support

Example:
    >>> from stream import StreamLoader
    >>> loader = StreamLoader("rtsp://192.168.2.71:8554/mystream7")
    >>> for sources, images, info in loader:
    ...     # Process frames
    ...     for img in images:
    ...         process(img)
    >>> loader.close()

    # With context manager:
    >>> with StreamLoader("video.mp4") as loader:
    ...     for sources, images, info in loader:
    ...         process(images)

    # Multiple streams:
    >>> loader = StreamLoader([
    ...     "rtsp://192.168.2.71:8554/mystream7",
    ...     "rtsp://192.168.2.71:8554/mystream8"
    ... ])
"""

from .constants import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_FPS_FALLBACK,
    IMG_FORMATS,
    STREAM_PROTOCOLS,
    VID_FORMATS,
)
from .exceptions import (
    InvalidSourceError,
    StreamConnectionError,
    StreamError,
    StreamReadError,
)
from .stream_loader import StreamLoader
from .utils import clean_str, is_stream_url, is_video_file, is_webcam

__version__ = "1.0.0"

__all__ = [
    # Main class
    "StreamLoader",
    # Exceptions
    "StreamError",
    "StreamConnectionError",
    "StreamReadError",
    "InvalidSourceError",
    # Constants
    "VID_FORMATS",
    "IMG_FORMATS",
    "STREAM_PROTOCOLS",
    "DEFAULT_BUFFER_SIZE",
    "DEFAULT_FPS_FALLBACK",
    # Utilities
    "clean_str",
    "is_stream_url",
    "is_video_file",
    "is_webcam",
]
