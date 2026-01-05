"""Algorithm Warehouse Connector Implementations.

This module provides concrete implementations of the TaskConnectorInterface
and EngineInfoProviderInterface for integrating with the main engine system.

Classes:
    MainEngineConnector: Connector that retrieves task info from MainEngine.
    DefaultEngineInfoProvider: Simple engine info provider with static values.
"""

import socket
from typing import List, Optional

from .config import EngineCapabilities, WarehouseConfig
from .interfaces import (
    EngineInfoProviderInterface,
    TaskConnectorInterface,
    TaskInfo,
    TaskStatus,
)


def get_local_ip() -> str:
    """Get the local IP address of this machine.

    Attempts to determine the IP address by connecting to an external host.
    Falls back to 127.0.0.1 if detection fails.

    Returns:
        The local IP address as a string.
    """
    try:
        # Create a socket and connect to external host to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        # Using Google DNS - doesn't actually send data
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback: try to get hostname-based IP
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def get_hostname() -> str:
    """Get the hostname of this machine.

    Returns:
        The hostname as a string, or 'unknown' if detection fails.
    """
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def generate_engine_id(engine_name: str, engine_version: str) -> str:
    """Generate a unique engine identifier.

    Creates an ID in the format: {engine_name}#{engine_version}#{hostname}

    Args:
        engine_name: Name of the engine application.
        engine_version: Version string of the engine.

    Returns:
        Unique engine identifier string.
    """
    hostname = get_hostname()
    return f"{engine_name}#{engine_version}#{hostname}"


class DefaultEngineInfoProvider(EngineInfoProviderInterface):
    """Simple engine info provider with static configuration.

    This provider returns fixed values provided at initialization time,
    suitable for simple deployments or testing.
    """

    def __init__(
        self,
        engine_id: Optional[str] = None,
        engine_addr: Optional[str] = None,
        engine_port: int = 8555,
<<<<<<< HEAD
        engine_name: str = "easyair-mta",
        engine_version: str = "1.0.0",
=======
        engine_name: str = "easyair-terminal",
        engine_version: str = "2.0.0.0",
>>>>>>> 07331326 (feat: build video analytics task management system)
    ):
        """Initialize the engine info provider.

        Args:
            engine_id: Explicit engine ID, or None to auto-generate.
            engine_addr: Explicit engine address, or None to auto-detect.
            engine_port: Engine API port number.
            engine_name: Engine name for ID generation.
            engine_version: Engine version for ID generation.
        """
        self._engine_id = engine_id or generate_engine_id(engine_name, engine_version)
        self._engine_addr = engine_addr or get_local_ip()
        self._engine_port = engine_port

    def get_engine_id(self) -> str:
        """Get the engine identifier."""
        return self._engine_id

    def get_engine_addr(self) -> str:
        """Get the engine network address."""
        return self._engine_addr

    def get_engine_port(self) -> int:
        """Get the engine API port."""
        return self._engine_port


class MainEngineConnector(TaskConnectorInterface, EngineInfoProviderInterface):
    """Connector that retrieves task information from MainEngine.

    This connector bridges the warehouse service with the MainEngine singleton,
    providing real-time task information and engine capabilities. It implements
    both TaskConnectorInterface and EngineInfoProviderInterface for convenience.

    The connector lazily imports MainEngine to avoid circular dependencies
    and only accesses it when methods are called.
    """

    def __init__(
        self,
        config: Optional[WarehouseConfig] = None,
        engine_id: Optional[str] = None,
        engine_addr: Optional[str] = None,
    ):
        """Initialize the MainEngine connector.

        Args:
            config: Optional warehouse configuration for engine info defaults.
            engine_id: Explicit engine ID, or None to auto-generate.
            engine_addr: Explicit engine address, or None to auto-detect.
        """
        self._config = config or WarehouseConfig()
        self._engine_id = engine_id
        self._engine_addr = engine_addr
        self._engine = None
        self._engine_config = None

    def _get_engine(self):
        """Lazily get the MainEngine singleton.

        Returns:
            The MainEngine instance, or None if not available.
        """
        if self._engine is None:
            try:
                from engine import MainEngine
                self._engine = MainEngine()
            except ImportError:
                pass
        return self._engine

    def _get_engine_config(self):
        """Lazily get the EngineConfig singleton.

        Returns:
            The EngineConfig instance, or None if not available.
        """
        if self._engine_config is None:
            try:
                from engine.config import get_engine_config
                self._engine_config = get_engine_config()
            except ImportError:
                pass
        return self._engine_config

    # === TaskConnectorInterface implementation ===

    def get_name(self) -> str:
        """Get the connector name for identification."""
        return "MainEngineConnector"

    def get_task_count(self) -> int:
        """Get the current number of active tasks.

        Returns:
            Number of tasks, or 0 if MainEngine is not available.
        """
        engine = self._get_engine()
        if engine is None:
            return 0

        try:
            status = engine.get_engine_status()
            return status.get("task_count", 0)
        except Exception:
            return 0

    def get_tasks(self) -> List[TaskInfo]:
        """Get information for all active tasks.

        Converts MainEngine task dictionaries to TaskInfo objects.

        Returns:
            List of TaskInfo objects for all current tasks.
        """
        engine = self._get_engine()
        if engine is None:
            return []

        tasks = []
        try:
            all_tasks = engine.get_all_tasks()
            for task_dict in all_tasks:
                # Determine task status based on state
                is_running = task_dict.get("is_running", False)
                is_paused = task_dict.get("is_paused", False)
                has_error = task_dict.get("has_error", False)

                if has_error:
                    status = TaskStatus.ERROR
                elif is_paused:
                    status = TaskStatus.SCHEDULED  # Paused tasks treated as scheduled
                elif is_running:
                    status = TaskStatus.RUNNING
                else:
                    status = TaskStatus.SCHEDULED

                # Get actual stream width from TaskConfig
                task_id = task_dict.get("task_id", "")
                resolution = 0  # Default to 0 if not available
                if task_id:
                    from task import TaskConfigManager
                    config = TaskConfigManager().get_config(task_id)
                    if config and config.actual_width:
                        resolution = config.actual_width

                tasks.append(TaskInfo(
                    task_id=task_dict.get("task_id", ""),
                    task_status=status,
                    task_err_code=task_dict.get("error_code", 0),
                    task_err_msg=task_dict.get("error_msg", "OK"),
                    resolution=resolution,
                    task_name=task_dict.get("task_name"),
                ))
        except Exception:
            pass

        return tasks

    def get_capabilities(self) -> EngineCapabilities:
        """Get current engine capabilities.

        Retrieves capability information from MainEngine.get_capabilities().

        Returns:
            Current EngineCapabilities snapshot.
        """
        engine = self._get_engine()
        if engine is None:
            return EngineCapabilities()

        try:
            caps = engine.get_capabilities()
<<<<<<< HEAD
            resolution = caps.get("resolutionCap", [300, 500])
=======
            resolution = caps.get("resolutionCap", [300, 5000])
>>>>>>> 07331326 (feat: build video analytics task management system)
            if isinstance(resolution, list):
                resolution = tuple(resolution)

            return EngineCapabilities(
                task_cur_num=caps.get("taskCurNum", 0),
                task_total_num=caps.get("taskTotalNum", 8),
                total_capability=caps.get("totalCapability", 1000),
                cur_capability=caps.get("curCapability", 500),
                resolution_cap=resolution,
            )
        except Exception:
            return EngineCapabilities()

    def is_available(self) -> bool:
        """Check if the MainEngine is available and initialized.

        Returns:
            True if MainEngine can be accessed and is initialized.
        """
        engine = self._get_engine()
        if engine is None:
            return False

        try:
            status = engine.get_engine_status()
            return status.get("initialized", False)
        except Exception:
            return False

    # === EngineInfoProviderInterface implementation ===

    def get_engine_id(self) -> str:
        """Get the unique engine identifier.

        Uses explicitly set ID, or generates one from configuration.

        Returns:
            Unique engine identifier string.
        """
        if self._engine_id:
            return self._engine_id

        engine_config = self._get_engine_config()
        if engine_config:
            name = getattr(engine_config, "service_name", self._config.engine_name)
            version = self._config.engine_version
        else:
            name = self._config.engine_name
            version = self._config.engine_version

        return generate_engine_id(name, version)

    def get_engine_addr(self) -> str:
        """Get the engine network address.

        Uses explicitly set address, or auto-detects local IP.

        Returns:
            IP address or hostname string.
        """
        if self._engine_addr:
            return self._engine_addr
        return get_local_ip()

    def get_engine_port(self) -> int:
        """Get the engine API port.

        Uses engine config if available, otherwise falls back to
        warehouse config default.

        Returns:
            Port number for the engine's API.
        """
        engine_config = self._get_engine_config()
        if engine_config:
            return getattr(engine_config, "service_port", self._config.engine_port)
        return self._config.engine_port
