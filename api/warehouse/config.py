"""Algorithm Warehouse Configuration Module.

This module provides immutable configuration classes for the Algorithm Warehouse
service integration. All configuration classes use frozen dataclasses to ensure
thread safety and prevent accidental modifications after initialization.

Classes:
    EngineCapabilities: Engine resource and capacity information.
    WarehouseConfig: Configuration for warehouse server connection and sync behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class EngineCapabilities:
    """Immutable engine capabilities snapshot for warehouse reporting.

    This dataclass represents the current state of engine resources and capacity,
    used for periodic synchronization with the Algorithm Warehouse server.

    Attributes:
        task_cur_num: Current number of active tasks.
        task_total_num: Maximum task capacity of the engine.
        total_capability: Total processing capability units available.
        cur_capability: Current available capability units.
        resolution_cap: Supported video resolution range as (min, max) tuple.
    """

    task_cur_num: int = 0
    task_total_num: int = 8
    total_capability: int = 1000
    cur_capability: int = 500
    resolution_cap: Tuple[int, int] = (300, 500)

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to camelCase format for API requests.

        Returns:
            Dictionary with camelCase keys matching the warehouse API format.
        """
        return {
            "taskCurNum": self.task_cur_num,
            "taskTotalNum": self.task_total_num,
            "totalCapability": self.total_capability,
            "curCapability": self.cur_capability,
            "resolutionCap": list(self.resolution_cap),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineCapabilities":
        """Create an EngineCapabilities instance from a dictionary.

        Supports both camelCase (API format) and snake_case keys.

        Args:
            data: Dictionary containing capability values.

        Returns:
            New EngineCapabilities instance.
        """
        resolution = data.get("resolutionCap", data.get("resolution_cap", [300, 500]))
        if isinstance(resolution, list):
            resolution = tuple(resolution)

        return cls(
            task_cur_num=data.get("taskCurNum", data.get("task_cur_num", 0)),
            task_total_num=data.get("taskTotalNum", data.get("task_total_num", 8)),
            total_capability=data.get("totalCapability", data.get("total_capability", 1000)),
            cur_capability=data.get("curCapability", data.get("cur_capability", 500)),
            resolution_cap=resolution,
        )


@dataclass(frozen=True)
class WarehouseConfig:
    """Immutable configuration for Algorithm Warehouse connection.

    This dataclass contains all settings required for connecting to and
    communicating with the Algorithm Warehouse server, including server
    address, retry behavior, and engine identification.

    Attributes:
        host: Warehouse server hostname or IP address.
        port: Warehouse server port number.
        enabled: Whether warehouse integration is enabled.
        sync_interval: Seconds between capability sync calls.
        connect_timeout: HTTP connection timeout in seconds.
        read_timeout: HTTP read timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
        retry_base_delay: Base delay in seconds for exponential backoff.
        retry_max_delay: Maximum delay in seconds between retries.
        engine_name: Name of this engine for identification.
        engine_version: Version string for this engine.
        engine_port: Local API port this engine listens on.
        algo_category: Algorithm category for task classification.
    """

    # Server connection settings
    host: str = "easyair-algo-warehouse"
    port: int = 30000

    # Feature toggle
    enabled: bool = True

    # Sync settings
    sync_interval: int = 60

    # HTTP settings
    connect_timeout: int = 10
    read_timeout: int = 30
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Engine identification
    engine_name: str = "easyair-mta"
    engine_version: str = "1.0.0"
    engine_port: int = 8555

    # Algorithm classification
    algo_category: str = "behavior"

    @property
    def base_url(self) -> str:
        """Get the base URL for warehouse API requests.

        Returns:
            Complete base URL including protocol, host, and port.
        """
        return f"http://{self.host}:{self.port}"

    def validate(self) -> bool:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.

        Returns:
            True if all values are valid.
        """
        if not self.host:
            raise ValueError("host cannot be empty")

        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")

        if self.sync_interval <= 0:
            raise ValueError(f"sync_interval must be positive, got {self.sync_interval}")

        if self.connect_timeout <= 0:
            raise ValueError(f"connect_timeout must be positive, got {self.connect_timeout}")

        if self.read_timeout <= 0:
            raise ValueError(f"read_timeout must be positive, got {self.read_timeout}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")

        if self.retry_base_delay < 0:
            raise ValueError(f"retry_base_delay cannot be negative, got {self.retry_base_delay}")

        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError(
                f"retry_max_delay ({self.retry_max_delay}) must be >= "
                f"retry_base_delay ({self.retry_base_delay})"
            )

        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarehouseConfig":
        """Create a WarehouseConfig instance from a dictionary.

        Supports both camelCase and snake_case keys. Missing values use defaults.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            New WarehouseConfig instance.
        """
        return cls(
            host=data.get("host", data.get("server_addr", "easyair-algo-warehouse")),
            port=int(data.get("port", 30000)),
            enabled=data.get("enabled", data.get("enable", True)),
            sync_interval=int(data.get("sync_interval", data.get("syncInterval", 60))),
            connect_timeout=int(data.get("connect_timeout", data.get("connectTimeout", 10))),
            read_timeout=int(data.get("read_timeout", data.get("readTimeout", 30))),
            max_retries=int(data.get("max_retries", data.get("maxRetries", 3))),
            retry_base_delay=float(data.get("retry_base_delay", data.get("retryBaseDelay", 1.0))),
            retry_max_delay=float(data.get("retry_max_delay", data.get("retryMaxDelay", 30.0))),
            engine_name=data.get("engine_name", data.get("engineName", "easyair-mta")),
            engine_version=data.get("engine_version", data.get("engineVersion", "1.0.0")),
            engine_port=int(data.get("engine_port", data.get("enginePort", 8555))),
            algo_category=data.get("algo_category", data.get("algoCategory", "behavior")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary with snake_case keys containing all configuration values.
        """
        return {
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "sync_interval": self.sync_interval,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "engine_port": self.engine_port,
            "algo_category": self.algo_category,
        }
