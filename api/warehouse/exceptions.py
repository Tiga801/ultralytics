"""Algorithm Warehouse Exception Classes.

This module defines custom exception classes for the Algorithm Warehouse
integration. These exceptions provide structured error handling and enable
callers to catch and handle specific failure scenarios.

Exception Hierarchy:
    WarehouseError (base)
    ├── WarehouseConnectionError
    ├── WarehouseRegistrationError
    ├── WarehouseSyncError
    ├── WarehouseTimeoutError
    └── WarehouseResponseError
"""

from typing import Any, Dict, Optional


class WarehouseError(Exception):
    """Base exception for all warehouse-related errors.

    This is the parent class for all warehouse exceptions. Catching this
    exception will catch any warehouse-related error.
    """

    pass


class WarehouseConnectionError(WarehouseError):
    """Raised when connection to the warehouse server fails.

    This exception indicates network-level failures such as DNS resolution
    errors, connection refused, or network unreachable conditions.
    """

    pass


class WarehouseRegistrationError(WarehouseError):
    """Raised when engine registration with the warehouse fails.

    This exception indicates that the registration request was sent but
    the warehouse server rejected it or returned an error response.
    """

    pass


class WarehouseSyncError(WarehouseError):
    """Raised when capability or task synchronization fails.

    This exception indicates that a sync request was sent but the warehouse
    server rejected it or returned an error response.
    """

    pass


class WarehouseTimeoutError(WarehouseError):
    """Raised when a warehouse request times out.

    This exception indicates that the request was sent but no response was
    received within the configured timeout period.
    """

    pass


class WarehouseResponseError(WarehouseError):
    """Raised when the warehouse returns an unexpected or error response.

    This exception includes additional context about the response received,
    including the error code and full response data when available.

    Attributes:
        code: The error code from the response, or None if not available.
        response: The full response dictionary, or None if not available.
    """

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the response error with context.

        Args:
            message: Human-readable error description.
            code: Error code from the warehouse response.
            response: Full response dictionary for debugging.
        """
        super().__init__(message)
        self.code = code
        self.response = response

    def __str__(self) -> str:
        """Return string representation including error code if present."""
        if self.code is not None:
            return f"{self.args[0]} (code: {self.code})"
        return self.args[0]
