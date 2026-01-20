"""Algorithm Warehouse Interfaces Module.

This module defines abstract interfaces and data structures for the Algorithm
Warehouse integration. These interfaces enable loose coupling between the
warehouse service and concrete implementations of task management and engine
information providers.

Classes:
    TaskInfo: Data structure for task status reporting.
    ServiceStatus: Data structure for warehouse service status.
    TaskConnectorInterface: Abstract interface for task information retrieval.
    EngineInfoProviderInterface: Abstract interface for engine identification.
    WarehouseEventCallback: Abstract interface for event notifications.
    DefaultWarehouseEventCallback: No-op implementation of event callbacks.
    LoggingEventCallback: Event callback that logs to warehouse.log.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .config import EngineCapabilities


# Type alias for logging functions
LogFunc = Callable[[str], None]


class TaskStatus:
    """Task status code constants.

    These status codes are used for reporting task states to the Algorithm
    Warehouse server.
    """

    SCHEDULED = 3    # Task has been scheduled but not started
    RUNNING = 4      # Task is currently running
    ERROR = 6        # Task encountered an error
    COMPLETED = 7    # Task has completed successfully

    @classmethod
    def get_status_text(cls, status: int) -> str:
        """Get human-readable text for a status code.

        Args:
            status: The status code to translate.

        Returns:
            Human-readable status description.
        """
        status_map = {
            cls.SCHEDULED: "Scheduled",
            cls.RUNNING: "Running",
            cls.ERROR: "Error",
            cls.COMPLETED: "Completed",
        }
        return status_map.get(status, "Unknown")


@dataclass
class TaskInfo:
    """Task information structure for warehouse reporting.

    This dataclass represents the state of a single task for synchronization
    with the Algorithm Warehouse server.

    Attributes:
        task_id: Unique task identifier.
        task_status: Task status code (see TaskStatus constants).
        task_err_code: Error code, 0 indicates no error.
        task_err_msg: Error or status message.
        resolution: Actual video stream width in pixels (from TaskConfig.actual_width).
        task_name: Optional human-readable task name.
    """

    task_id: str
    task_status: int  # TaskStatus.SCHEDULED, RUNNING, ERROR, or COMPLETED
    task_err_code: int = 0
    task_err_msg: str = "OK"
    resolution: int = 0  # Actual stream width, set from TaskConfig
    task_name: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API format dictionary.

        Note: The API uses 'resloution' (misspelled) for compatibility with
        the existing warehouse server protocol.

        Returns:
            Dictionary with camelCase keys for API requests.
        """
        return {
            "taskID": self.task_id,
            "taskStatus": self.task_status,
            "taskErrCode": self.task_err_code,
            "taskErrMsg": self.task_err_msg,
            "resloution": self.resolution,  # API typo preserved for compatibility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        """Create a TaskInfo instance from a dictionary.

        Supports both camelCase (API format) and snake_case keys.

        Args:
            data: Dictionary containing task information.

        Returns:
            New TaskInfo instance.
        """
        return cls(
            task_id=data.get("taskID", data.get("task_id", "")),
            task_status=data.get("taskStatus", data.get("task_status", TaskStatus.RUNNING)),
            task_err_code=data.get("taskErrCode", data.get("task_err_code", 0)),
            task_err_msg=data.get("taskErrMsg", data.get("task_err_msg", "OK")),
            resolution=data.get("resloution", data.get("resolution", 0)),
            task_name=data.get("taskName", data.get("task_name")),
        )


@dataclass
class ServiceStatus:
    """Warehouse service status information.

    This dataclass provides a snapshot of the warehouse service state,
    useful for monitoring and debugging.

    Attributes:
        enabled: Whether the warehouse service is enabled.
        registered: Whether the engine is registered with the warehouse.
        running: Whether the sync thread is running.
        sync_interval: Current sync interval in seconds.
        task_count: Current number of active tasks.
        last_sync_time: Unix timestamp of last successful sync, or None.
        connector_name: Name of the registered task connector, or None.
    """

    enabled: bool = True
    registered: bool = False
    running: bool = False
    sync_interval: int = 60
    task_count: int = 0
    last_sync_time: Optional[float] = None
    connector_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all status fields.
        """
        return {
            "enabled": self.enabled,
            "registered": self.registered,
            "running": self.running,
            "sync_interval": self.sync_interval,
            "task_count": self.task_count,
            "last_sync_time": self.last_sync_time,
            "connector_name": self.connector_name,
        }


class TaskConnectorInterface(ABC):
    """Abstract interface for retrieving task information.

    Implementations of this interface provide task data from various sources
    such as MainEngine, TaskConfigManager, or external systems. This enables
    the warehouse service to remain decoupled from specific task management
    implementations.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Get the connector name for identification.

        Returns:
            Unique name identifying this connector instance.
        """
        pass

    @abstractmethod
    def get_task_count(self) -> int:
        """Get the current number of active tasks.

        Returns:
            Number of tasks currently managed by this connector.
        """
        pass

    @abstractmethod
    def get_tasks(self) -> List[TaskInfo]:
        """Get information for all active tasks.

        Returns:
            List of TaskInfo objects for all current tasks.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """Get current engine capabilities.

        Returns:
            Current EngineCapabilities snapshot reflecting resource usage.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the connector is available and healthy.

        Returns:
            True if the connector can provide valid data.
        """
        pass


class EngineInfoProviderInterface(ABC):
    """Abstract interface for engine identification information.

    Implementations provide the engine identity used for registration and
    synchronization with the Algorithm Warehouse server.
    """

    @abstractmethod
    def get_engine_id(self) -> str:
        """Get the unique engine identifier.

        Returns:
            Unique identifier string for this engine instance.
        """
        pass

    @abstractmethod
    def get_engine_addr(self) -> str:
        """Get the engine network address.

        Returns:
            IP address or hostname where this engine can be reached.
        """
        pass

    @abstractmethod
    def get_engine_port(self) -> int:
        """Get the engine API port.

        Returns:
            Port number for the engine's API endpoint.
        """
        pass


class WarehouseEventCallback(ABC):
    """Abstract interface for warehouse lifecycle event callbacks.

    Implementations receive notifications about key events in the warehouse
    service lifecycle, enabling logging, monitoring, or custom actions.
    """

    @abstractmethod
    def on_register_success(self, engine_id: str) -> None:
        """Called when engine registration succeeds.

        Args:
            engine_id: The registered engine identifier.
        """
        pass

    @abstractmethod
    def on_register_failed(self, engine_id: str, error: str) -> None:
        """Called when engine registration fails.

        Args:
            engine_id: The engine identifier that failed to register.
            error: Description of the failure reason.
        """
        pass

    @abstractmethod
    def on_sync_completed(self, task_count: int) -> None:
        """Called when capability/task synchronization succeeds.

        Args:
            task_count: Number of tasks synchronized.
        """
        pass

    @abstractmethod
    def on_sync_failed(self, error: str) -> None:
        """Called when synchronization fails.

        Args:
            error: Description of the failure reason.
        """
        pass

    @abstractmethod
    def on_service_started(self) -> None:
        """Called when the warehouse service starts successfully."""
        pass

    @abstractmethod
    def on_service_stopped(self) -> None:
        """Called when the warehouse service stops."""
        pass


class DefaultWarehouseEventCallback(WarehouseEventCallback):
    """Default no-op implementation of warehouse event callbacks.

    This class provides empty implementations for all callback methods,
    suitable for use as a base class or when event handling is not needed.
    """

    def on_register_success(self, engine_id: str) -> None:
        """No-op implementation."""
        pass

    def on_register_failed(self, engine_id: str, error: str) -> None:
        """No-op implementation."""
        pass

    def on_sync_completed(self, task_count: int) -> None:
        """No-op implementation."""
        pass

    def on_sync_failed(self, error: str) -> None:
        """No-op implementation."""
        pass

    def on_service_started(self) -> None:
        """No-op implementation."""
        pass

    def on_service_stopped(self) -> None:
        """No-op implementation."""
        pass


class LoggingEventCallback(WarehouseEventCallback):
    """Event callback implementation that logs all events.

    This callback writes all warehouse events to the provided log function,
    intended for use with the dedicated warehouse logger.
    """

    def __init__(self, log_func: LogFunc):
        """Initialize the logging callback.

        Args:
            log_func: Function to call for logging messages.
        """
        self._log = log_func

    def on_register_success(self, engine_id: str) -> None:
        """Log registration success."""
        self._log(f"Engine registered successfully: {engine_id}")

    def on_register_failed(self, engine_id: str, error: str) -> None:
        """Log registration failure."""
        self._log(f"Engine registration failed for {engine_id}: {error}")

    def on_sync_completed(self, task_count: int) -> None:
        """Log sync completion."""
        self._log(f"Sync completed: {task_count} tasks synchronized")

    def on_sync_failed(self, error: str) -> None:
        """Log sync failure."""
        self._log(f"Sync failed: {error}")

    def on_service_started(self) -> None:
        """Log service start."""
        self._log("Warehouse service started")

    def on_service_stopped(self) -> None:
        """Log service stop."""
        self._log("Warehouse service stopped")
