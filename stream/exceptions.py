"""Custom exceptions for the stream module.

This module defines exception classes for handling various
stream-related error conditions.
"""

from __future__ import annotations


class StreamError(Exception):
    """Base exception for stream-related errors."""

    pass


class StreamConnectionError(StreamError):
    """Raised when unable to connect to a stream.

    Attributes:
        source: The stream source that failed to connect.
    """

    def __init__(self, source: str, message: str = ""):
        self.source = source
        msg = f"Failed to connect to {source}"
        if message:
            msg += f". {message}"
        super().__init__(msg)


class StreamReadError(StreamError):
    """Raised when unable to read from a stream.

    Attributes:
        source: The stream source that failed to read.
    """

    def __init__(self, source: str, message: str = ""):
        self.source = source
        msg = f"Failed to read from {source}"
        if message:
            msg += f". {message}"
        super().__init__(msg)


class InvalidSourceError(StreamError):
    """Raised when an invalid source is provided.

    Attributes:
        source: The invalid source string.
    """

    def __init__(self, source: str, message: str = ""):
        self.source = source
        msg = f"Invalid source: {source}"
        if message:
            msg += f". {message}"
        super().__init__(msg)
