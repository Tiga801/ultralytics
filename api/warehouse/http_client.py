"""Algorithm Warehouse HTTP Client Module.

This module provides an HTTP client for communicating with the Algorithm
Warehouse server. The client includes connection pooling for efficiency,
exponential backoff retry for resilience, and proper timeout handling.

Classes:
    WarehouseHttpClient: HTTP client with connection pooling and retry logic.
"""

import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None
    HTTPAdapter = None
    Retry = None

from .config import EngineCapabilities, WarehouseConfig
from .exceptions import (
    WarehouseConnectionError,
    WarehouseResponseError,
    WarehouseTimeoutError,
)
from .interfaces import TaskInfo


class WarehouseHttpClient:
    """HTTP client for Algorithm Warehouse API communication.

    This client provides reliable HTTP communication with the warehouse server,
    featuring connection pooling for efficiency and exponential backoff retry
    for handling transient failures.

    Attributes:
        config: The warehouse configuration.
        is_started: Whether the client has been started.
    """

    # API endpoint paths
    ENDPOINT_REGISTER = "/algo/IAP/Cloud/engineRegister"
    ENDPOINT_CAP_SYNC = "/algo/IAP/Cloud/engineCapSync"
    ENDPOINT_TASK_SYNC = "/algo/IAP/runTaskSync"

    def __init__(
        self,
        config: WarehouseConfig,
        log_func: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the HTTP client.

        Args:
            config: Warehouse configuration containing server address and timeouts.
            log_func: Optional function for logging messages.

        Raises:
            ImportError: If the requests library is not installed.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is not installed. "
                "Please install it with: pip install requests"
            )

        self._config = config
        self._log = log_func or self._default_log
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()
        self._started = False

    @staticmethod
    def _default_log(message: str) -> None:
        """Default logging function using print."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WAREHOUSE] {message}")

    @property
    def is_started(self) -> bool:
        """Check if the client has been started."""
        return self._started

    @property
    def base_url(self) -> str:
        """Get the base URL for API requests."""
        return self._config.base_url

    def start(self) -> None:
        """Initialize the HTTP session with connection pooling.

        This method creates a requests Session with an HTTPAdapter configured
        for connection pooling. Call this before making any requests.
        """
        with self._lock:
            if self._started:
                return

            self._session = requests.Session()

            # Configure connection pooling
            adapter = HTTPAdapter(
                pool_connections=5,
                pool_maxsize=10,
                max_retries=0,  # We handle retries ourselves
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            self._started = True
            self._log("HTTP client started with connection pooling")

    def stop(self) -> None:
        """Close the HTTP session and release connections.

        This method should be called when the client is no longer needed
        to properly release resources.
        """
        with self._lock:
            if not self._started:
                return

            if self._session is not None:
                self._session.close()
                self._session = None

            self._started = False
            self._log("HTTP client stopped")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter.

        The formula is: min(base * 2^attempt * jitter, max_delay)
        where jitter is a random value between 0.75 and 1.25.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in seconds before the next retry.
        """
        base = self._config.retry_base_delay
        max_delay = self._config.retry_max_delay

        # Exponential component
        exponential = base * (2 ** attempt)

        # Add jitter: multiply by random value in [0.75, 1.25]
        jitter = 0.75 + random.random() * 0.5
        delay = exponential * jitter

        return min(delay, max_delay)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an HTTP request with retry logic.

        This method handles retries with exponential backoff for transient
        failures (connection errors, timeouts, 5xx errors). Client errors
        (4xx) are not retried.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path (e.g., "/algo/IAP/Cloud/engineRegister").
            json_data: Optional JSON payload for the request.

        Returns:
            Response data as a dictionary with 'ok' field added.

        Raises:
            WarehouseConnectionError: If connection fails after all retries.
            WarehouseTimeoutError: If request times out after all retries.
            WarehouseResponseError: If server returns an error response.
        """
        if not self._started:
            self.start()

        if not self._config.enabled:
            return {
                "code": -1,
                "msg": "Warehouse integration is disabled",
                "data": None,
                "ok": False,
            }

        url = f"{self.base_url}{endpoint}"
        max_attempts = self._config.max_retries + 1
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                self._log(f"Request: {method} {endpoint} (attempt {attempt + 1}/{max_attempts})")

                response = self._session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    timeout=(self._config.connect_timeout, self._config.read_timeout),
                )

                # Check for server errors (5xx) - these are retryable
                if response.status_code >= 500:
                    self._log(f"Server error: {response.status_code}")
                    if attempt < max_attempts - 1:
                        delay = self._calculate_backoff(attempt)
                        self._log(f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    raise WarehouseResponseError(
                        f"Server error: {response.status_code}",
                        code=response.status_code,
                    )

                # Client errors (4xx) are not retried
                if response.status_code >= 400:
                    raise WarehouseResponseError(
                        f"Client error: {response.status_code}",
                        code=response.status_code,
                    )

                # Parse successful response
                try:
                    result = response.json()
                except ValueError:
                    result = {"code": 0, "msg": "OK", "data": response.text}

                result["ok"] = result.get("code", 0) == 0
                return result

            except requests.exceptions.Timeout as e:
                last_exception = e
                self._log(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    delay = self._calculate_backoff(attempt)
                    self._log(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                self._log(f"Connection error: {e} (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    delay = self._calculate_backoff(attempt)
                    self._log(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

            except WarehouseResponseError:
                raise

            except Exception as e:
                last_exception = e
                self._log(f"Unexpected error: {e}")
                break

        # All retries exhausted
        if isinstance(last_exception, requests.exceptions.Timeout):
            raise WarehouseTimeoutError(
                f"Request timed out after {max_attempts} attempts"
            )
        elif isinstance(last_exception, requests.exceptions.ConnectionError):
            raise WarehouseConnectionError(
                f"Connection failed after {max_attempts} attempts: {last_exception}"
            )
        else:
            raise WarehouseConnectionError(f"Request failed: {last_exception}")

    def register_engine(
        self,
        engine_id: str,
        engine_addr: str,
        engine_port: int,
        capabilities: EngineCapabilities,
    ) -> Dict[str, Any]:
        """Register an engine with the Algorithm Warehouse.

        Args:
            engine_id: Unique identifier for the engine.
            engine_addr: Network address where the engine can be reached.
            engine_port: Port number for the engine's API.
            capabilities: Current engine capabilities.

        Returns:
            Response dictionary with 'ok' field indicating success.

        Raises:
            WarehouseConnectionError: If connection fails.
            WarehouseTimeoutError: If request times out.
            WarehouseResponseError: If server returns an error.
        """
        payload = {
            "engineID": engine_id,
            "engineAddr": engine_addr,
            "enginePort": engine_port,
            "capabilities": capabilities.to_api_dict(),
        }

        self._log(f"Registering engine: {engine_id} at {engine_addr}:{engine_port}")
        result = self._make_request("POST", self.ENDPOINT_REGISTER, payload)

        if result.get("ok"):
            self._log("Engine registration successful")
        else:
            self._log(f"Engine registration failed: {result.get('msg')}")

        return result

    def sync_capabilities(
        self,
        engine_id: str,
        engine_addr: str,
        engine_port: int,
        capabilities: EngineCapabilities,
    ) -> Dict[str, Any]:
        """Synchronize engine capabilities with the warehouse.

        Args:
            engine_id: Unique identifier for the engine.
            engine_addr: Network address where the engine can be reached.
            engine_port: Port number for the engine's API.
            capabilities: Current engine capabilities to report.

        Returns:
            Response dictionary with 'ok' field indicating success.

        Raises:
            WarehouseConnectionError: If connection fails.
            WarehouseTimeoutError: If request times out.
            WarehouseResponseError: If server returns an error.
        """
        payload = {
            "engineID": engine_id,
            "engineAddr": engine_addr,
            "enginePort": engine_port,
            "capabilities": capabilities.to_api_dict(),
        }

        self._log(f"Syncing capabilities: {capabilities.task_cur_num}/{capabilities.task_total_num} tasks")
        return self._make_request("POST", self.ENDPOINT_CAP_SYNC, payload)

    def sync_tasks(
        self,
        engine_id: str,
        tasks: List[TaskInfo],
        algo_category: str = "behavior",
    ) -> Dict[str, Any]:
        """Synchronize task information with the warehouse.

        Args:
            engine_id: Unique identifier for the engine.
            tasks: List of task information to report.
            algo_category: Category of algorithms for classification.

        Returns:
            Response dictionary with 'ok' field indicating success.

        Raises:
            WarehouseConnectionError: If connection fails.
            WarehouseTimeoutError: If request times out.
            WarehouseResponseError: If server returns an error.
        """
        payload = {
            "engineID": engine_id,
            "taskRunInfoList": [task.to_api_dict() for task in tasks],
            "algoCategory": algo_category,
        }

        self._log(f"Syncing {len(tasks)} task(s)")
        return self._make_request("POST", self.ENDPOINT_TASK_SYNC, payload)
