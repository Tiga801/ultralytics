"""Time-related utility functions.

This module provides helper functions for time operations used across
the task management system.
"""

import time
from datetime import datetime, timezone
from typing import Optional


def get_timestamp() -> float:
    """Get the current Unix timestamp.

    Returns:
        Current time as Unix timestamp (seconds since epoch).
    """
    return time.time()


def get_timestamp_ms() -> int:
    """Get the current Unix timestamp in milliseconds.

    Returns:
        Current time as Unix timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def get_formatted_time(
    timestamp: Optional[float] = None,
    fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Get formatted time string.

    Args:
        timestamp: Unix timestamp. If None, uses current time.
        fmt: Format string for strftime.

    Returns:
        Formatted time string.

    Example:
        >>> get_formatted_time()
        '2024-01-01 12:00:00'
        >>> get_formatted_time(fmt="%H:%M:%S")
        '12:00:00'
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime(fmt)


def timezone_offset() -> int:
    """Get the local timezone offset in seconds.

    Returns:
        Timezone offset from UTC in seconds.
    """
    local_time = datetime.now()
    utc_time = datetime.now(timezone.utc).replace(tzinfo=None)
    offset = local_time - utc_time
    return int(offset.total_seconds())


def is_within_time_range(
    start_time: str,
    end_time: str,
    time_format: str = "%H:%M:%S",
    current_time: Optional[datetime] = None
) -> bool:
    """Check if current time is within a time range.

    Args:
        start_time: Start time string (e.g., "08:00:00").
        end_time: End time string (e.g., "18:00:00").
        time_format: Format of time strings.
        current_time: Optional current time. If None, uses now.

    Returns:
        True if current time is within the range.

    Example:
        >>> is_within_time_range("08:00:00", "18:00:00")
        True  # if current time is between 8 AM and 6 PM
    """
    if current_time is None:
        current_time = datetime.now()

    current = current_time.strftime(time_format)
    start = datetime.strptime(start_time, time_format).strftime(time_format)
    end = datetime.strptime(end_time, time_format).strftime(time_format)

    # Handle overnight ranges (e.g., 22:00 to 06:00)
    if start <= end:
        return start <= current <= end
    else:
        return current >= start or current <= end


def is_weekday_active(weekdays: str, current_day: Optional[int] = None) -> bool:
    """Check if the current weekday is in the active list.

    Args:
        weekdays: Comma-separated weekday numbers (1=Monday, 7=Sunday).
                  Example: "1,2,3,4,5" for weekdays only.
        current_day: Optional day number. If None, uses today.

    Returns:
        True if current day is in the active list.

    Example:
        >>> is_weekday_active("1,2,3,4,5")  # Weekdays only
        True  # if today is Monday-Friday
    """
    if current_day is None:
        current_day = datetime.now().isoweekday()  # 1=Monday, 7=Sunday

    active_days = [int(d.strip()) for d in weekdays.split(",") if d.strip()]
    return current_day in active_days


class Timer:
    """Simple timer for measuring execution time.

    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do something ...
        >>> elapsed = timer.stop()
        >>> print(f"Elapsed: {elapsed:.3f}s")
    """

    def __init__(self):
        """Initialize timer."""
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> "Timer":
        """Start the timer.

        Returns:
            Self for method chaining.
        """
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds.
        """
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time
            self._start_time = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Get elapsed time without stopping.

        Returns:
            Elapsed time in seconds.
        """
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed

    def __enter__(self) -> "Timer":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
