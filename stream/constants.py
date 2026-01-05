"""Standalone constants for the stream module.

This module contains format definitions and configuration constants
used throughout the stream loading functionality.
"""

from __future__ import annotations

# Supported video formats (file extensions)
VID_FORMATS = frozenset({
    "asf", "avi", "gif", "m4v", "mkv", "mov",
    "mp4", "mpeg", "mpg", "ts", "wmv", "webm"
})

# Supported image formats (for reference)
IMG_FORMATS = frozenset({
    "bmp", "dng", "jpeg", "jpg", "mpo", "png",
    "tif", "tiff", "webp", "pfm", "heic"
})

# Stream protocol prefixes
STREAM_PROTOCOLS = ("rtsp://", "rtmp://", "http://", "https://", "tcp://")

# Default configuration values
DEFAULT_BUFFER_SIZE = 30  # Maximum frames to buffer per stream
DEFAULT_VID_STRIDE = 1  # Process every frame by default
DEFAULT_RECONNECT_DELAY = 1.0  # Seconds to wait before reconnect attempt
DEFAULT_FPS_FALLBACK = 30.0  # FPS to use when stream doesn't report it

# Special characters pattern for cleaning source names
SPECIAL_CHARS_PATTERN = r"[|@#!¡·$€%&()=?¿^*;:,¨´><+]"
