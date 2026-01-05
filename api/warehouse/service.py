"""Algorithm Warehouse Service Module.

This module provides the main WarehouseService singleton that coordinates
all warehouse integration functionality including engine registration,
periodic capability synchronization, and task status reporting.

The service runs in the main process only. Task processes running in
separate processes communicate through the MainEngine, which aggregates
their state for the warehouse service to report.

Classes:
    WarehouseService: Singleton service for Algorithm Warehouse integration.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from utils.singleton import SingletonMeta

from .config import WarehouseConfig
from .connectors import DefaultEngineInfoProvider, MainEngineConnector
from .exceptions import (
    WarehouseConnectionError,
    WarehouseError,
    WarehouseRegistrationError,
)
from .http_client import WarehouseHttpClient
from .interfaces import (
    EngineInfoProviderInterface,
    LoggingEventCallback,
    ServiceStatus,
    TaskConnectorInterface,
    WarehouseEventCallback,
)
from .logger import get_warehouse_logger


class WarehouseService(metaclass=SingletonMeta):
    """Singleton service for Algorithm Warehouse integration.

    This service handles all communication with the external Algorithm
    Warehouse server, including:
    - Engine registration on startup
    - Periodic capability and load synchronization
    - Task status reporting

    The service runs a background thread for periodic synchronization
    and supports event callbacks for monitoring lifecycle events.

    IMPORTANT: This service is designed to run only in the main process.
    Task processes do NOT instantiate this service. Instead, task state
    flows through MainEngine, which this service reads from.

    Example:
        >>> from api.warehouse import WarehouseService, WarehouseConfig
        >>> service = WarehouseService()
        >>> service.init(config=WarehouseConfig())
        >>> service.start()
        >>> # ... later ...
        >>> service.stop()
    """

    def __init__(self):
        """Initialize the warehouse service.

        The service is not started until init() and start() are called.
        """
        # Configuration
        self._config: Optional[WarehouseConfig] = None

        # Components
        self._connector: Optional[TaskConnectorInterface] = None
        self._info_provider: Optional[EngineInfoProviderInterface] = None
        self._http_client: Optional[WarehouseHttpClient] = None

        # Callbacks
        self._callbacks: List[WarehouseEventCallback] = []
        self._callbacks_lock = threading.Lock()

        # Background sync thread
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # State
        self._initialized = False
        self._registered = False
        self._running = False
        self._last_sync_time: Optional[float] = None
        self._consecutive_failures = 0

        # Logging
        self._log: Optional[Callable[[str], None]] = None

        # Thread safety
        self._lock = threading.RLock()

    def init(
        self,
        config: Optional[WarehouseConfig] = None,
        connector: Optional[TaskConnectorInterface] = None,
        info_provider: Optional[EngineInfoProviderInterface] = None,
    ) -> None:
        """Initialize the service with configuration and components.

        This method sets up the service configuration and components but
        does not start the background sync thread. Call start() to begin
        synchronization.

        Args:
            config: Warehouse configuration. Uses defaults if not provided.
            connector: Task connector for retrieving task info. Uses
                MainEngineConnector if not provided.
            info_provider: Engine info provider. Uses DefaultEngineInfoProvider
                if not provided and connector doesn't implement it.
        """
        with self._lock:
            if self._initialized:
                return

            # Set configuration with defaults
            self._config = config or WarehouseConfig()

            # Validate configuration
            self._config.validate()

            # Set up logging
            self._log = get_warehouse_logger()

            # Set up connector
            if connector is not None:
                self._connector = connector
            else:
                self._connector = MainEngineConnector(config=self._config)

            # Set up info provider
            if info_provider is not None:
                self._info_provider = info_provider
            elif isinstance(self._connector, EngineInfoProviderInterface):
                # Use connector as info provider if it implements the interface
                self._info_provider = self._connector
            else:
                self._info_provider = DefaultEngineInfoProvider(
                    engine_port=self._config.engine_port,
                    engine_name=self._config.engine_name,
                    engine_version=self._config.engine_version,
                )

            # Create HTTP client
            self._http_client = WarehouseHttpClient(
                config=self._config,
                log_func=self._log,
            )

            # Add logging callback by default
            self.add_callback(LoggingEventCallback(self._log))

            self._initialized = True
            self._log("Service initialized")
            self._log(f"Server: {self._config.base_url}")
            self._log(f"Engine ID: {self._info_provider.get_engine_id()}")
            self._log(f"Sync interval: {self._config.sync_interval}s")

    def start(self) -> bool:
        """Start the warehouse service.

        This method:
        1. Initializes the HTTP client
        2. Attempts to register the engine with the warehouse
        3. Starts the background sync thread

        Returns:
            True if the service started successfully, False otherwise.

        Raises:
            RuntimeError: If the service was not initialized.
        """
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Service not initialized. Call init() first.")

            if self._running:
                self._log("Service already running")
                return True

            if not self._config.enabled:
                self._log("Warehouse integration is disabled")
                return False

            # Start HTTP client
            self._http_client.start()

            # Attempt initial registration
            self._registered = self._register_engine()

            # Start sync thread
            self._stop_event.clear()
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                name="WarehouseSyncThread",
                daemon=True,
            )
            self._sync_thread.start()

            self._running = True
            self._dispatch_event("on_service_started")
            self._log("Service started")

            return True

    def stop(self) -> None:
        """Stop the warehouse service gracefully.

        This method signals the sync thread to stop and waits for it to
        complete before releasing resources.
        """
        with self._lock:
            if not self._running:
                return

            self._log("Stopping service...")

            # Signal stop
            self._stop_event.set()

            # Wait for sync thread
            if self._sync_thread is not None:
                self._sync_thread.join(timeout=5.0)
                self._sync_thread = None

            # Stop HTTP client
            if self._http_client is not None:
                self._http_client.stop()

            self._running = False
            self._registered = False
            self._dispatch_event("on_service_stopped")
            self._log("Service stopped")

    def add_callback(self, callback: WarehouseEventCallback) -> None:
        """Register an event callback.

        Args:
            callback: The callback to register.
        """
        with self._callbacks_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: WarehouseEventCallback) -> None:
        """Unregister an event callback.

        Args:
            callback: The callback to remove.
        """
        with self._callbacks_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def force_sync(self) -> bool:
        """Force an immediate synchronization.

        This method performs capability and task synchronization immediately,
        outside the normal sync interval.

        Returns:
            True if sync was successful, False otherwise.
        """
        if not self._running:
            return False

        try:
            cap_success = self._sync_capabilities()
            task_success = self._sync_tasks()
            return cap_success and task_success
        except Exception as e:
            self._log(f"Force sync failed: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if the service has been initialized."""
        return self._initialized

    def is_registered(self) -> bool:
        """Check if the engine is registered with the warehouse."""
        return self._registered

    def is_running(self) -> bool:
        """Check if the sync thread is running."""
        return self._running

    def get_status(self) -> ServiceStatus:
        """Get the current service status.

        Returns:
            ServiceStatus object with current state information.
        """
        connector_name = None
        task_count = 0

        if self._connector is not None:
            connector_name = self._connector.get_name()
            try:
                task_count = self._connector.get_task_count()
            except Exception:
                pass

        return ServiceStatus(
            enabled=self._config.enabled if self._config else True,
            registered=self._registered,
            running=self._running,
            sync_interval=self._config.sync_interval if self._config else 60,
            task_count=task_count,
            last_sync_time=self._last_sync_time,
            connector_name=connector_name,
        )

    def _dispatch_event(self, event_name: str, *args) -> None:
        """Dispatch an event to all registered callbacks.

        Args:
            event_name: Name of the callback method to invoke.
            *args: Arguments to pass to the callback.
        """
        with self._callbacks_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                handler = getattr(callback, event_name, None)
                if handler is not None:
                    handler(*args)
            except Exception as e:
                self._log(f"Callback error in {event_name}: {e}")

    def _register_engine(self) -> bool:
        """Register the engine with the Algorithm Warehouse.

        Returns:
            True if registration was successful, False otherwise.
        """
        try:
            engine_id = self._info_provider.get_engine_id()
            engine_addr = self._info_provider.get_engine_addr()
            engine_port = self._info_provider.get_engine_port()
            capabilities = self._connector.get_capabilities()

            result = self._http_client.register_engine(
                engine_id=engine_id,
                engine_addr=engine_addr,
                engine_port=engine_port,
                capabilities=capabilities,
            )

            if result.get("ok"):
                self._dispatch_event("on_register_success", engine_id)
                return True
            else:
                error_msg = result.get("msg", "Unknown error")
                self._dispatch_event("on_register_failed", engine_id, error_msg)
                return False

        except WarehouseError as e:
            engine_id = self._info_provider.get_engine_id()
            self._dispatch_event("on_register_failed", engine_id, str(e))
            return False

        except Exception as e:
            self._log(f"Registration error: {e}")
            engine_id = self._info_provider.get_engine_id()
            self._dispatch_event("on_register_failed", engine_id, str(e))
            return False

    def _sync_capabilities(self) -> bool:
        """Synchronize engine capabilities with the warehouse.

        Returns:
            True if sync was successful, False otherwise.
        """
        try:
            engine_id = self._info_provider.get_engine_id()
            engine_addr = self._info_provider.get_engine_addr()
            engine_port = self._info_provider.get_engine_port()
            capabilities = self._connector.get_capabilities()

            result = self._http_client.sync_capabilities(
                engine_id=engine_id,
                engine_addr=engine_addr,
                engine_port=engine_port,
                capabilities=capabilities,
            )

            return result.get("ok", False)

        except Exception as e:
            self._log(f"Capability sync error: {e}")
            return False

    def _sync_tasks(self) -> bool:
        """Synchronize task information with the warehouse.

        Returns:
            True if sync was successful, False otherwise.
        """
        try:
            engine_id = self._info_provider.get_engine_id()
            tasks = self._connector.get_tasks()

            result = self._http_client.sync_tasks(
                engine_id=engine_id,
                tasks=tasks,
                algo_category=self._config.algo_category,
            )

            return result.get("ok", False)

        except Exception as e:
            self._log(f"Task sync error: {e}")
            return False

    def _sync_loop(self) -> None:
        """Background thread loop for periodic synchronization.

        This method runs in a daemon thread and periodically synchronizes
        capabilities and task status with the warehouse server.
        """
        self._log("Sync thread started")

        while not self._stop_event.is_set():
            try:
                # Re-register if not registered
                if not self._registered:
                    self._registered = self._register_engine()
                    if not self._registered:
                        # Wait before retry
                        self._stop_event.wait(self._config.sync_interval)
                        continue

                # Sync capabilities
                cap_success = self._sync_capabilities()

                # Sync tasks
                task_success = self._sync_tasks()

                # Update state
                if cap_success and task_success:
                    self._last_sync_time = time.time()
                    self._consecutive_failures = 0
                    task_count = self._connector.get_task_count()
                    self._dispatch_event("on_sync_completed", task_count)
                else:
                    self._consecutive_failures += 1
                    error_msg = "Sync failed"
                    if not cap_success:
                        error_msg = "Capability sync failed"
                    elif not task_success:
                        error_msg = "Task sync failed"
                    self._dispatch_event("on_sync_failed", error_msg)

                    # If too many consecutive failures, mark as unregistered
                    # to trigger re-registration
                    if self._consecutive_failures >= 3:
                        self._registered = False
                        self._log("Too many failures, will attempt re-registration")

            except Exception as e:
                self._log(f"Sync loop error: {e}")
                self._consecutive_failures += 1
                self._dispatch_event("on_sync_failed", str(e))

            # Wait for next sync interval or stop signal
            self._stop_event.wait(self._config.sync_interval)

        self._log("Sync thread stopped")


def get_warehouse_service() -> WarehouseService:
    """Get the singleton WarehouseService instance.

    This is a convenience function for accessing the warehouse service.

    Returns:
        The WarehouseService singleton instance.
    """
    return WarehouseService()
