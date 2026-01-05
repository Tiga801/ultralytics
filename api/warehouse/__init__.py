"""Algorithm Warehouse Integration Module.

This module provides integration with the external Algorithm Warehouse service
for engine registration, capability reporting, and task synchronization.

The warehouse service runs in the main process only and uses a singleton
pattern to ensure a globally shared instance across the application.

Quick Start:
    >>> from api.warehouse import (
    ...     WarehouseService,
    ...     WarehouseConfig,
    ...     MainEngineConnector,
    ... )
    >>>
    >>> # Create configuration
    >>> config = WarehouseConfig(
    ...     host="easyair-algo-warehouse",
    ...     port=30000,
    ... )
    >>>
    >>> # Create connector to MainEngine
    >>> connector = MainEngineConnector()
    >>>
    >>> # Initialize and start service
    >>> service = WarehouseService()
    >>> service.init(config=config, connector=connector)
    >>> service.start()
    >>>
    >>> # ... later ...
    >>> service.stop()

Default Configuration:
    - Host: easyair-algo-warehouse
    - Port: 30000
    - Sync Interval: 60 seconds
    - Max Retries: 3 with exponential backoff

Logging:
    All warehouse-related logs are written to logs/warehouse.log,
    independent from the main service logging.

Classes:
    WarehouseConfig: Immutable configuration for warehouse connection.
    EngineCapabilities: Immutable engine capability snapshot.
    TaskInfo: Task information for warehouse reporting.
    ServiceStatus: Warehouse service status information.
    TaskConnectorInterface: Abstract interface for task retrieval.
    EngineInfoProviderInterface: Abstract interface for engine identification.
    WarehouseEventCallback: Abstract interface for event notifications.
    DefaultWarehouseEventCallback: No-op callback implementation.
    LoggingEventCallback: Callback that logs to warehouse.log.
    MainEngineConnector: Connector for MainEngine integration.
    DefaultEngineInfoProvider: Simple static engine info provider.
    WarehouseService: Main singleton service for warehouse integration.
    WarehouseHttpClient: HTTP client with connection pooling.

Exceptions:
    WarehouseError: Base exception for warehouse errors.
    WarehouseConnectionError: Connection to warehouse failed.
    WarehouseRegistrationError: Engine registration failed.
    WarehouseSyncError: Capability or task sync failed.
    WarehouseTimeoutError: Request timed out.
    WarehouseResponseError: Unexpected response from warehouse.

Functions:
    get_warehouse_service: Get the singleton WarehouseService instance.
    setup_warehouse_logger: Configure the warehouse logger.
    get_warehouse_logger: Get the warehouse logger function.
"""

# Configuration
from .config import EngineCapabilities, WarehouseConfig

# Interfaces and data structures
from .interfaces import (
    DefaultWarehouseEventCallback,
    EngineInfoProviderInterface,
    LogFunc,
    LoggingEventCallback,
    ServiceStatus,
    TaskConnectorInterface,
    TaskInfo,
    TaskStatus,
    WarehouseEventCallback,
)

# Exceptions
from .exceptions import (
    WarehouseConnectionError,
    WarehouseError,
    WarehouseRegistrationError,
    WarehouseResponseError,
    WarehouseSyncError,
    WarehouseTimeoutError,
)

# Connectors
from .connectors import (
    DefaultEngineInfoProvider,
    MainEngineConnector,
    generate_engine_id,
    get_hostname,
    get_local_ip,
)

# HTTP client
from .http_client import WarehouseHttpClient

# Service
from .service import WarehouseService, get_warehouse_service

# Logger
from .logger import get_warehouse_logger, reset_warehouse_logger, setup_warehouse_logger

__all__ = [
    # Configuration
    "WarehouseConfig",
    "EngineCapabilities",
    # Interfaces
    "TaskInfo",
    "ServiceStatus",
    "TaskStatus",
    "TaskConnectorInterface",
    "EngineInfoProviderInterface",
    "WarehouseEventCallback",
    "DefaultWarehouseEventCallback",
    "LoggingEventCallback",
    "LogFunc",
    # Exceptions
    "WarehouseError",
    "WarehouseConnectionError",
    "WarehouseRegistrationError",
    "WarehouseSyncError",
    "WarehouseTimeoutError",
    "WarehouseResponseError",
    # Connectors
    "MainEngineConnector",
    "DefaultEngineInfoProvider",
    "get_local_ip",
    "get_hostname",
    "generate_engine_id",
    # HTTP Client
    "WarehouseHttpClient",
    # Service
    "WarehouseService",
    "get_warehouse_service",
    # Logger
    "setup_warehouse_logger",
    "get_warehouse_logger",
    "reset_warehouse_logger",
]
