"""Standalone utility functions for the stream module.

This module provides utility functions for string cleaning,
logging setup, and source type detection.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from .constants import SPECIAL_CHARS_PATTERN, STREAM_PROTOCOLS, VID_FORMATS

# Module logger (console only for backward compatibility)
logger = logging.getLogger("stream_loader")

# Stream file logger (set by get_stream_file_logger)
_stream_file_logger: Optional[Callable[[str], None]] = None


def clean_str(s: str) -> str:
    """Clean a string by replacing special characters with '_' character.

    Args:
        s: A string needing special characters replaced.

    Returns:
        A string with special characters replaced by an underscore _.
    """
    return re.sub(pattern=SPECIAL_CHARS_PATTERN, repl="_", string=s)


def setup_logger(
    name: str = "stream_loader",
    level: int = logging.INFO,
    format_str: str = "%(message)s"
) -> logging.Logger:
    """Setup and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level.
        format_str: Log message format.

    Returns:
        Configured logger instance.
    """
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_str))
        log.addHandler(handler)
    log.setLevel(level)
    return log


def get_stream_file_logger() -> Callable[[str], None]:
    """Get the stream file logger.

    This function returns a logger that writes to logs/streams.log.
    It uses the centralized Logger class from utils module.

    Returns:
        A callable that logs stream messages to file.
    """
    global _stream_file_logger

    if _stream_file_logger is not None:
        return _stream_file_logger

    try:
        from utils import Logger
        _stream_file_logger = Logger.get_stream_logger()
    except ImportError:
        # Fallback to console logger if utils not available
        def fallback_log(message: str) -> None:
            logger.info(f"[STREAM] {message}")
        _stream_file_logger = fallback_log

    return _stream_file_logger


def is_stream_url(source: str) -> bool:
    """Check if source is a stream URL (RTSP, RTMP, HTTP, TCP).

    Args:
        source: Source string to check.

    Returns:
        True if source is a stream URL, False otherwise.
    """
    if not isinstance(source, str):
        return False
    source_lower = source.lower()
    return source_lower.startswith(STREAM_PROTOCOLS)


def is_video_file(source: str) -> bool:
    """Check if source is a video file based on extension.

    Args:
        source: Source string (path or URL) to check.

    Returns:
        True if source has a video file extension, False otherwise.
    """
    if not isinstance(source, str):
        return False
    # Get the file extension from the path
    parsed = urlparse(source)
    path = parsed.path if parsed.scheme else source
    ext = Path(path).suffix.lower().lstrip(".")
    return ext in VID_FORMATS


def is_webcam(source: str | int) -> bool:
    """Check if source is a webcam index.

    Args:
        source: Source to check (string or int).

    Returns:
        True if source represents a webcam (numeric), False otherwise.
    """
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.isnumeric()
    return False


def is_streams_file(source: str) -> bool:
    """Check if source is a .streams file containing multiple sources.

    Args:
        source: Source string to check.

    Returns:
        True if source is a .streams file, False otherwise.
    """
    if not isinstance(source, str):
        return False
    return source.endswith(".streams") and os.path.isfile(source)


def parse_sources_file(filepath: str) -> list[str]:
    """Parse a file containing stream sources (one per line).

    Args:
        filepath: Path to the sources file.

    Returns:
        List of source URLs/paths.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {filepath}")
    return path.read_text().strip().split()


def format_stream_info(
    index: int,
    total: int,
    source: str,
    width: int,
    height: int,
    fps: float,
    frames: int | float
) -> str:
    """Format stream information for logging.

    Args:
        index: Stream index (0-based).
        total: Total number of streams.
        source: Source URL/path.
        width: Frame width.
        height: Frame height.
        fps: Frames per second.
        frames: Total frames (inf for live streams).

    Returns:
        Formatted info string.
    """
    frames_str = "inf" if frames == float("inf") else str(int(frames))
    return f"{index + 1}/{total}: {source}... Success ({frames_str} frames of shape {width}x{height} at {fps:.2f} FPS)"
