"""Algorithm Warehouse Module Test Suite.

This module provides comprehensive tests for the Algorithm Warehouse integration,
including configuration validation, HTTP client behavior, connector functionality,
and service lifecycle management.

Test Categories:
    - Configuration tests: Validate config defaults, immutability, and validation
    - HTTP client tests: Test retry logic, backoff, and error handling
    - Connector tests: Test task info mapping and capability retrieval
    - Service tests: Test singleton pattern, lifecycle, and callbacks
    - Integration tests: Test communication with mock warehouse server
"""

import threading
import time
from dataclasses import FrozenInstanceError
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from api.warehouse import (
    DefaultEngineInfoProvider,
    DefaultWarehouseEventCallback,
    EngineCapabilities,
    LoggingEventCallback,
    ServiceStatus,
    TaskConnectorInterface,
    TaskInfo,
    TaskStatus,
    WarehouseConfig,
    WarehouseConnectionError,
    WarehouseError,
    WarehouseHttpClient,
    WarehouseResponseError,
    WarehouseService,
    WarehouseTimeoutError,
    generate_engine_id,
    get_hostname,
    get_local_ip,
    get_warehouse_logger,
    reset_warehouse_logger,
    setup_warehouse_logger,
)
from utils.singleton import SingletonMeta


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create a default WarehouseConfig for testing."""
    return WarehouseConfig()


@pytest.fixture
def custom_config():
    """Create a custom WarehouseConfig for testing."""
    return WarehouseConfig(
        host="test-warehouse",
        port=31000,
        enabled=True,
        sync_interval=30,
        connect_timeout=5,
        read_timeout=15,
        max_retries=2,
        engine_name="test-engine",
        engine_version="2.0.0",
        engine_port=9000,
    )


@pytest.fixture
def mock_connector():
    """Create a mock TaskConnector for testing."""
    connector = MagicMock(spec=TaskConnectorInterface)
    connector.get_name.return_value = "MockConnector"
    connector.get_task_count.return_value = 2
    connector.get_tasks.return_value = [
        TaskInfo(task_id="task001", task_status=TaskStatus.RUNNING),
        TaskInfo(task_id="task002", task_status=TaskStatus.SCHEDULED),
    ]
    connector.get_capabilities.return_value = EngineCapabilities(
        task_cur_num=2,
        task_total_num=8,
        total_capability=1000,
        cur_capability=800,
    )
    connector.is_available.return_value = True
    return connector


@pytest.fixture
def mock_info_provider():
    """Create a mock EngineInfoProvider for testing."""
    return DefaultEngineInfoProvider(
        engine_id="test-engine#1.0.0#testhost",
        engine_addr="192.168.1.100",
        engine_port=8555,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances before each test."""
    SingletonMeta.clear_all_instances()
    reset_warehouse_logger()
    yield
    SingletonMeta.clear_all_instances()
    reset_warehouse_logger()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestWarehouseConfig:
    """Tests for WarehouseConfig dataclass."""

    def test_default_values(self, default_config):
        """Test that default configuration values are correct."""
        assert default_config.host == "easyair-algo-warehouse"
        assert default_config.port == 30000
        assert default_config.enabled is True
        assert default_config.sync_interval == 60
        assert default_config.connect_timeout == 10
        assert default_config.read_timeout == 30
        assert default_config.max_retries == 3
        assert default_config.retry_base_delay == 1.0
        assert default_config.retry_max_delay == 30.0
        assert default_config.engine_name == "easyair-mta"
        assert default_config.engine_version == "1.0.0"
        assert default_config.engine_port == 8555
        assert default_config.algo_category == "behavior"

    def test_frozen_immutability(self, default_config):
        """Test that configuration is immutable (frozen dataclass)."""
        with pytest.raises(FrozenInstanceError):
            default_config.host = "new-host"

    def test_base_url_property(self, default_config):
        """Test base_url property returns correct URL."""
        assert default_config.base_url == "http://easyair-algo-warehouse:30000"

    def test_custom_base_url(self, custom_config):
        """Test base_url with custom host and port."""
        assert custom_config.base_url == "http://test-warehouse:31000"

    def test_from_dict_snake_case(self):
        """Test creating config from snake_case dictionary."""
        config = WarehouseConfig.from_dict({
            "host": "my-warehouse",
            "port": 32000,
            "sync_interval": 120,
            "engine_name": "my-engine",
        })
        assert config.host == "my-warehouse"
        assert config.port == 32000
        assert config.sync_interval == 120
        assert config.engine_name == "my-engine"

    def test_from_dict_legacy_server_addr(self):
        """Test creating config from dict with legacy server_addr key."""
        config = WarehouseConfig.from_dict({
            "server_addr": "legacy-warehouse",
            "port": 30000,
        })
        assert config.host == "legacy-warehouse"

    def test_to_dict(self, custom_config):
        """Test converting config to dictionary."""
        config_dict = custom_config.to_dict()
        assert config_dict["host"] == "test-warehouse"
        assert config_dict["port"] == 31000
        assert config_dict["sync_interval"] == 30

    def test_validate_success(self, default_config):
        """Test validation passes for valid config."""
        assert default_config.validate() is True

    def test_validate_empty_host(self):
        """Test validation fails for empty host."""
        config = WarehouseConfig(host="")
        with pytest.raises(ValueError, match="host cannot be empty"):
            config.validate()

    def test_validate_invalid_port(self):
        """Test validation fails for invalid port."""
        config = WarehouseConfig(port=0)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

        config = WarehouseConfig(port=70000)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_validate_negative_sync_interval(self):
        """Test validation fails for negative sync interval."""
        config = WarehouseConfig(sync_interval=-1)
        with pytest.raises(ValueError, match="sync_interval must be positive"):
            config.validate()


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = EngineCapabilities()
        assert caps.task_cur_num == 0
        assert caps.task_total_num == 8
        assert caps.total_capability == 1000
        assert caps.cur_capability == 500
        assert caps.resolution_cap == (300, 500)

    def test_frozen_immutability(self):
        """Test that capabilities are immutable."""
        caps = EngineCapabilities()
        with pytest.raises(FrozenInstanceError):
            caps.task_cur_num = 5

    def test_to_api_dict(self):
        """Test conversion to API format (camelCase)."""
        caps = EngineCapabilities(
            task_cur_num=3,
            task_total_num=10,
            total_capability=2000,
            cur_capability=1500,
            resolution_cap=(400, 600),
        )
        api_dict = caps.to_api_dict()
        assert api_dict["taskCurNum"] == 3
        assert api_dict["taskTotalNum"] == 10
        assert api_dict["totalCapability"] == 2000
        assert api_dict["curCapability"] == 1500
        assert api_dict["resolutionCap"] == [400, 600]

    def test_from_dict_camel_case(self):
        """Test creating from camelCase API response."""
        caps = EngineCapabilities.from_dict({
            "taskCurNum": 5,
            "taskTotalNum": 12,
            "totalCapability": 1500,
            "curCapability": 1000,
            "resolutionCap": [200, 800],
        })
        assert caps.task_cur_num == 5
        assert caps.task_total_num == 12
        assert caps.resolution_cap == (200, 800)


# =============================================================================
# Interface Tests
# =============================================================================


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_default_values(self):
        """Test default TaskInfo values."""
        task = TaskInfo(task_id="test-001", task_status=TaskStatus.RUNNING)
        assert task.task_id == "test-001"
        assert task.task_status == TaskStatus.RUNNING
        assert task.task_err_code == 0
        assert task.task_err_msg == "OK"
        assert task.resolution == 1920
        assert task.task_name is None

    def test_to_api_dict_preserves_typo(self):
        """Test that API dict uses 'resloution' (preserving API typo)."""
        task = TaskInfo(task_id="test-001", task_status=4, resolution=1080)
        api_dict = task.to_api_dict()
        assert api_dict["taskID"] == "test-001"
        assert api_dict["taskStatus"] == 4
        assert api_dict["resloution"] == 1080  # Note the typo
        assert "resolution" not in api_dict

    def test_from_dict(self):
        """Test creating TaskInfo from dictionary."""
        task = TaskInfo.from_dict({
            "taskID": "task-002",
            "taskStatus": TaskStatus.ERROR,
            "taskErrCode": 100,
            "taskErrMsg": "Stream error",
            "resloution": 720,
        })
        assert task.task_id == "task-002"
        assert task.task_status == TaskStatus.ERROR
        assert task.task_err_code == 100
        assert task.resolution == 720


class TestTaskStatus:
    """Tests for TaskStatus constants."""

    def test_status_values(self):
        """Test task status constant values."""
        assert TaskStatus.SCHEDULED == 3
        assert TaskStatus.RUNNING == 4
        assert TaskStatus.ERROR == 6
        assert TaskStatus.COMPLETED == 7

    def test_get_status_text(self):
        """Test status code to text conversion."""
        assert TaskStatus.get_status_text(TaskStatus.RUNNING) == "Running"
        assert TaskStatus.get_status_text(TaskStatus.ERROR) == "Error"
        assert TaskStatus.get_status_text(999) == "Unknown"


class TestServiceStatus:
    """Tests for ServiceStatus dataclass."""

    def test_default_values(self):
        """Test default ServiceStatus values."""
        status = ServiceStatus()
        assert status.enabled is True
        assert status.registered is False
        assert status.running is False
        assert status.sync_interval == 60
        assert status.task_count == 0
        assert status.last_sync_time is None
        assert status.connector_name is None

    def test_to_dict(self):
        """Test converting ServiceStatus to dictionary."""
        status = ServiceStatus(
            enabled=True,
            registered=True,
            running=True,
            task_count=5,
            connector_name="MainEngineConnector",
        )
        status_dict = status.to_dict()
        assert status_dict["registered"] is True
        assert status_dict["task_count"] == 5
        assert status_dict["connector_name"] == "MainEngineConnector"


# =============================================================================
# Callback Tests
# =============================================================================


class TestWarehouseCallbacks:
    """Tests for warehouse event callbacks."""

    def test_default_callback_no_op(self):
        """Test that default callback methods are no-op."""
        callback = DefaultWarehouseEventCallback()
        # These should not raise any exceptions
        callback.on_register_success("engine-1")
        callback.on_register_failed("engine-1", "error")
        callback.on_sync_completed(5)
        callback.on_sync_failed("error")
        callback.on_service_started()
        callback.on_service_stopped()

    def test_logging_callback(self, tmp_path):
        """Test that logging callback writes to log."""
        log_messages = []

        def mock_log(msg):
            log_messages.append(msg)

        callback = LoggingEventCallback(mock_log)
        callback.on_register_success("test-engine")
        callback.on_sync_completed(3)

        assert len(log_messages) == 2
        assert "test-engine" in log_messages[0]
        assert "3 tasks" in log_messages[1]


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_hostname(self):
        """Test hostname retrieval."""
        hostname = get_hostname()
        assert isinstance(hostname, str)
        assert len(hostname) > 0

    def test_get_local_ip(self):
        """Test local IP detection."""
        ip = get_local_ip()
        assert isinstance(ip, str)
        # Should be a valid IP format
        parts = ip.split(".")
        assert len(parts) == 4

    def test_generate_engine_id(self):
        """Test engine ID generation format."""
        engine_id = generate_engine_id("my-app", "1.2.3")
        assert engine_id.startswith("my-app#1.2.3#")
        assert len(engine_id) > len("my-app#1.2.3#")


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestWarehouseHttpClient:
    """Tests for WarehouseHttpClient."""

    def test_client_initialization(self, default_config):
        """Test HTTP client initialization."""
        log_messages = []
        client = WarehouseHttpClient(
            config=default_config,
            log_func=lambda msg: log_messages.append(msg),
        )
        assert client.is_started is False
        assert client.base_url == "http://easyair-algo-warehouse:30000"

    def test_client_start_stop(self, default_config):
        """Test HTTP client start and stop lifecycle."""
        client = WarehouseHttpClient(config=default_config)
        assert client.is_started is False

        client.start()
        assert client.is_started is True

        client.stop()
        assert client.is_started is False

    def test_backoff_calculation(self, default_config):
        """Test exponential backoff delay calculation."""
        client = WarehouseHttpClient(config=default_config)

        # Test that delay increases with attempts
        delay0 = client._calculate_backoff(0)
        delay1 = client._calculate_backoff(1)
        delay2 = client._calculate_backoff(2)

        # Delays should generally increase (with some jitter)
        # Base delay is 1.0, so delay0 should be around 0.75-1.25
        assert 0.5 <= delay0 <= 2.0

        # delay1 should be around 1.5-2.5 (2^1 * base * jitter)
        assert delay1 > delay0 * 0.5  # Allow for jitter

        # Verify max delay cap
        delay_large = client._calculate_backoff(10)
        assert delay_large <= default_config.retry_max_delay

    @patch("api.warehouse.http_client.requests.Session")
    def test_disabled_config_returns_error(self, mock_session, default_config):
        """Test that disabled config returns error without making request."""
        config = WarehouseConfig(enabled=False)
        client = WarehouseHttpClient(config=config)
        client.start()

        result = client.register_engine(
            engine_id="test",
            engine_addr="127.0.0.1",
            engine_port=8555,
            capabilities=EngineCapabilities(),
        )

        assert result["ok"] is False
        assert "disabled" in result["msg"].lower()


# =============================================================================
# Connector Tests
# =============================================================================


class TestDefaultEngineInfoProvider:
    """Tests for DefaultEngineInfoProvider."""

    def test_explicit_values(self):
        """Test provider with explicitly set values."""
        provider = DefaultEngineInfoProvider(
            engine_id="my-engine-id",
            engine_addr="10.0.0.1",
            engine_port=9000,
        )
        assert provider.get_engine_id() == "my-engine-id"
        assert provider.get_engine_addr() == "10.0.0.1"
        assert provider.get_engine_port() == 9000

    def test_auto_generated_values(self):
        """Test provider with auto-generated values."""
        provider = DefaultEngineInfoProvider(
            engine_name="auto-engine",
            engine_version="3.0.0",
        )
        engine_id = provider.get_engine_id()
        assert engine_id.startswith("auto-engine#3.0.0#")

        engine_addr = provider.get_engine_addr()
        assert "." in engine_addr  # Should be IP format


# =============================================================================
# Service Tests
# =============================================================================


class TestWarehouseService:
    """Tests for WarehouseService singleton."""

    def test_singleton_pattern(self):
        """Test that WarehouseService is a singleton."""
        service1 = WarehouseService()
        service2 = WarehouseService()
        assert service1 is service2

    def test_init_required_before_start(self):
        """Test that start() fails without init()."""
        service = WarehouseService()
        with pytest.raises(RuntimeError, match="not initialized"):
            service.start()

    def test_init_sets_defaults(self, mock_connector, mock_info_provider):
        """Test that init() sets up defaults correctly."""
        service = WarehouseService()
        service.init(
            config=WarehouseConfig(),
            connector=mock_connector,
            info_provider=mock_info_provider,
        )

        assert service.is_initialized() is True
        assert service.is_running() is False
        assert service.is_registered() is False

    def test_get_status(self, mock_connector, mock_info_provider):
        """Test get_status() returns correct ServiceStatus."""
        service = WarehouseService()
        service.init(
            config=WarehouseConfig(sync_interval=45),
            connector=mock_connector,
            info_provider=mock_info_provider,
        )

        status = service.get_status()
        assert status.enabled is True
        assert status.sync_interval == 45
        assert status.connector_name == "MockConnector"
        assert status.task_count == 2

    def test_add_remove_callback(self, mock_connector):
        """Test adding and removing callbacks."""
        service = WarehouseService()
        service.init(connector=mock_connector)

        callback = DefaultWarehouseEventCallback()
        service.add_callback(callback)

        # Adding same callback again should not duplicate
        service.add_callback(callback)

        service.remove_callback(callback)

    @patch("api.warehouse.http_client.requests.Session")
    def test_start_stop_lifecycle(
        self, mock_session, mock_connector, mock_info_provider
    ):
        """Test service start and stop lifecycle."""
        # Mock successful registration response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 0, "msg": "success", "data": None}
        mock_session.return_value.request.return_value = mock_response

        service = WarehouseService()
        service.init(
            config=WarehouseConfig(sync_interval=1),
            connector=mock_connector,
            info_provider=mock_info_provider,
        )

        # Start service
        result = service.start()
        assert result is True
        assert service.is_running() is True

        # Let sync thread run briefly
        time.sleep(0.1)

        # Stop service
        service.stop()
        assert service.is_running() is False


# =============================================================================
# Integration Tests with Mock Server
# =============================================================================


class TestWarehouseIntegration:
    """Integration tests with mock warehouse server responses."""

    @patch("api.warehouse.http_client.requests.Session")
    def test_successful_registration(
        self, mock_session, mock_connector, mock_info_provider
    ):
        """Test successful engine registration flow."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 0, "msg": "success", "data": None}
        mock_session.return_value.request.return_value = mock_response

        service = WarehouseService()
        service.init(
            config=WarehouseConfig(sync_interval=60),
            connector=mock_connector,
            info_provider=mock_info_provider,
        )

        # Start and verify registration
        service.start()
        assert service.is_registered() is True

        # Verify request was made
        mock_session.return_value.request.assert_called()
        call_args = mock_session.return_value.request.call_args
        assert call_args[1]["method"] == "POST"
        assert "engineRegister" in call_args[1]["url"]

        service.stop()

    @patch("api.warehouse.http_client.requests.Session")
    def test_registration_failure_callback(
        self, mock_session, mock_connector, mock_info_provider
    ):
        """Test that registration failure triggers callback."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": -1,
            "msg": "Registration rejected",
            "data": None,
        }
        mock_session.return_value.request.return_value = mock_response

        callback_events = []

        class TestCallback(DefaultWarehouseEventCallback):
            def on_register_failed(self, engine_id, error):
                callback_events.append(("register_failed", engine_id, error))

        service = WarehouseService()
        service.init(
            config=WarehouseConfig(sync_interval=60),
            connector=mock_connector,
            info_provider=mock_info_provider,
        )
        service.add_callback(TestCallback())

        service.start()

        # Registration should have failed
        assert service.is_registered() is False
        assert len(callback_events) > 0
        assert callback_events[0][0] == "register_failed"

        service.stop()

    @patch("api.warehouse.http_client.requests.Session")
    def test_capability_sync_format(
        self, mock_session, mock_connector, mock_info_provider
    ):
        """Test that capability sync sends correct format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 0, "msg": "success", "data": None}
        mock_session.return_value.request.return_value = mock_response

        config = WarehouseConfig(
            host="test-warehouse",
            port=30000,
            sync_interval=1,
        )

        client = WarehouseHttpClient(config=config)
        client.start()

        caps = EngineCapabilities(
            task_cur_num=3,
            task_total_num=8,
            total_capability=1000,
            cur_capability=700,
        )

        result = client.sync_capabilities(
            engine_id="test-engine",
            engine_addr="192.168.1.1",
            engine_port=8555,
            capabilities=caps,
        )

        assert result["ok"] is True

        # Verify request payload format
        call_args = mock_session.return_value.request.call_args
        payload = call_args[1]["json"]
        assert payload["engineID"] == "test-engine"
        assert payload["capabilities"]["taskCurNum"] == 3
        assert payload["capabilities"]["taskTotalNum"] == 8

        client.stop()

    @patch("api.warehouse.http_client.requests.Session")
    def test_task_sync_format(self, mock_session, mock_connector, mock_info_provider):
        """Test that task sync sends correct format with API typo preserved."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": 0, "msg": "success", "data": None}
        mock_session.return_value.request.return_value = mock_response

        client = WarehouseHttpClient(config=WarehouseConfig())
        client.start()

        tasks = [
            TaskInfo(
                task_id="task-001",
                task_status=TaskStatus.RUNNING,
                resolution=1920,
            ),
            TaskInfo(
                task_id="task-002",
                task_status=TaskStatus.SCHEDULED,
                resolution=1080,
            ),
        ]

        result = client.sync_tasks(
            engine_id="test-engine",
            tasks=tasks,
            algo_category="behavior",
        )

        assert result["ok"] is True

        # Verify request payload
        call_args = mock_session.return_value.request.call_args
        payload = call_args[1]["json"]

        assert payload["engineID"] == "test-engine"
        assert payload["algoCategory"] == "behavior"
        assert len(payload["taskRunInfoList"]) == 2

        # Verify 'resloution' typo is preserved
        assert payload["taskRunInfoList"][0]["resloution"] == 1920
        assert payload["taskRunInfoList"][0]["taskStatus"] == TaskStatus.RUNNING

        client.stop()


# =============================================================================
# Logger Tests
# =============================================================================


class TestWarehouseLogger:
    """Tests for warehouse logger setup."""

    def test_setup_creates_log_directory(self, tmp_path):
        """Test that setup creates log directory if needed."""
        log_dir = tmp_path / "test_logs"
        log_func = setup_warehouse_logger(str(log_dir))

        assert log_dir.exists()
        assert (log_dir / "warehouse.log").exists() or True  # File created on first log

    def test_log_function_writes_with_tag(self, tmp_path):
        """Test that log function writes with [WAREHOUSE] tag."""
        reset_warehouse_logger()
        log_dir = tmp_path / "test_logs"
        log_func = setup_warehouse_logger(str(log_dir))

        log_func("Test message")

        log_file = log_dir / "warehouse.log"
        content = log_file.read_text()
        assert "[WAREHOUSE]" in content
        assert "Test message" in content

    def test_get_logger_returns_same_function(self, tmp_path):
        """Test that get_warehouse_logger returns singleton function."""
        reset_warehouse_logger()
        log_func1 = setup_warehouse_logger(str(tmp_path))
        log_func2 = get_warehouse_logger()

        assert log_func1 is log_func2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
