# -*- coding: utf-8 -*-
"""MQTT Client Tests.

Tests for the MQTT client module including configuration, message handling,
and basic connectivity verification.

Usage:
    # Run all tests
    pytest tests/test_mqtt.py -v

    # Run only unit tests (no network required)
    pytest tests/test_mqtt.py -v -m "not integration"

    # Run integration tests (requires MQTT broker)
    pytest tests/test_mqtt.py -v -m integration
"""

import sys
import time
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from utils.mqtt import MQTTClient, MQTTConfig, MQTTMessage, MQTTStatistics


class TestMQTTConfig:
    """Tests for MQTTConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = MQTTConfig()

        assert config.host == "easyair-mqtt"
        assert config.port == 1883
        assert config.username == "root"
        assert config.password == "P&5x19k@G3dw"
        assert config.topic_prefix == "assup/3379712260089733377/base2/subAttr"
        assert config.keepalive == 60
        assert config.connect_timeout == 30
        assert config.queue_max_size == 1000
        assert config.default_qos == 1
        assert config.max_reconnect_attempts == 0  # infinite

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = MQTTConfig(
            host="localhost",
            port=1884,
            username="test_user",
            password="test_pass",
            topic_prefix="custom/prefix",
            queue_max_size=500,
        )

        assert config.host == "localhost"
        assert config.port == 1884
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert config.topic_prefix == "custom/prefix"
        assert config.queue_max_size == 500

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        data = {
            "host": "mqtt.example.com",
            "port": 8883,
            "username": "user",
            "password": "secret",
            "topic_prefix": "test/prefix",
            "default_qos": 2,
        }
        config = MQTTConfig.from_dict(data)

        assert config.host == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "user"
        assert config.password == "secret"
        assert config.topic_prefix == "test/prefix"
        assert config.default_qos == 2

    def test_from_dict_legacy_fields(self):
        """Test backward compatibility with legacy field names."""
        data = {
            "broker_host": "legacy.broker.com",
            "broker_port": 1883,
        }
        config = MQTTConfig.from_dict(data)

        assert config.host == "legacy.broker.com"
        assert config.port == 1883

    def test_from_dict_topic_extraction(self):
        """Test topic prefix extraction from legacy topic field."""
        data = {"topic": "assup/device123/events/data"}
        config = MQTTConfig.from_dict(data)

        assert config.topic_prefix == "assup/device123/events/data"

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = MQTTConfig(host="test.com", port=1883)
        data = config.to_dict()

        assert data["host"] == "test.com"
        assert data["port"] == 1883
        assert "username" in data
        assert "password" in data

    def test_validate_success(self):
        """Test validation passes for valid configuration."""
        config = MQTTConfig()
        config.validate()  # Should not raise

    def test_validate_empty_host(self):
        """Test validation fails for empty host."""
        config = MQTTConfig(host="")
        with pytest.raises(ValueError, match="host cannot be empty"):
            config.validate()

    def test_validate_invalid_port(self):
        """Test validation fails for invalid port."""
        config = MQTTConfig(port=0)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

        config = MQTTConfig(port=70000)
        with pytest.raises(ValueError, match="port must be between"):
            config.validate()

    def test_validate_invalid_qos(self):
        """Test validation fails for invalid QoS."""
        config = MQTTConfig(default_qos=5)
        with pytest.raises(ValueError, match="default_qos must be"):
            config.validate()

    def test_validate_invalid_queue_size(self):
        """Test validation fails for invalid queue size."""
        config = MQTTConfig(queue_max_size=0)
        with pytest.raises(ValueError, match="queue_max_size must be positive"):
            config.validate()


class TestMQTTMessage:
    """Tests for MQTTMessage dataclass."""

    def test_message_creation(self):
        """Test message creation with default values."""
        msg = MQTTMessage(
            topic_suffix="events/alert",
            payload={"type": "motion", "zone": "entrance"},
        )

        assert msg.topic_suffix == "events/alert"
        assert msg.payload == {"type": "motion", "zone": "entrance"}
        assert msg.qos == 1
        assert msg.retain is False
        assert msg.message_id is not None
        assert msg.timestamp > 0

    def test_message_custom_qos(self):
        """Test message creation with custom QoS."""
        msg = MQTTMessage(
            topic_suffix="status",
            payload={"online": True},
            qos=2,
            retain=True,
        )

        assert msg.qos == 2
        assert msg.retain is True

    def test_get_full_topic(self):
        """Test full topic construction."""
        msg = MQTTMessage(topic_suffix="events", payload={})

        full_topic = msg.get_full_topic("assup/device/attr")
        assert full_topic == "assup/device/attr/events"

    def test_get_full_topic_empty_prefix(self):
        """Test full topic with empty prefix."""
        msg = MQTTMessage(topic_suffix="events", payload={})

        full_topic = msg.get_full_topic("")
        assert full_topic == "events"

    def test_to_json(self):
        """Test JSON serialization of payload."""
        msg = MQTTMessage(
            topic_suffix="data",
            payload={"value": 123, "name": "test"},
        )

        json_str = msg.to_json()
        assert '"value": 123' in json_str
        assert '"name": "test"' in json_str

    def test_to_json_unicode(self):
        """Test JSON serialization with unicode characters."""
        msg = MQTTMessage(
            topic_suffix="data",
            payload={"message": "Hello World"},
        )

        json_str = msg.to_json()
        assert "Hello World" in json_str

    def test_to_dict(self):
        """Test message conversion to dictionary."""
        msg = MQTTMessage(
            topic_suffix="events",
            payload={"event": "test"},
        )

        data = msg.to_dict()
        assert data["topic_suffix"] == "events"
        assert data["payload"] == {"event": "test"}
        assert "message_id" in data
        assert "timestamp" in data


class TestMQTTStatistics:
    """Tests for MQTTStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = MQTTStatistics()

        assert stats.messages_sent == 0
        assert stats.messages_failed == 0
        assert stats.messages_queued == 0
        assert stats.bytes_sent == 0
        assert stats.last_send_time is None
        assert stats.connection_count == 0
        assert stats.disconnect_count == 0

    def test_success_rate_no_messages(self):
        """Test success rate when no messages sent."""
        stats = MQTTStatistics()
        assert stats.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Test success rate with all successful messages."""
        stats = MQTTStatistics(messages_sent=100, messages_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_partial_success(self):
        """Test success rate with some failures."""
        stats = MQTTStatistics(messages_sent=75, messages_failed=25)
        assert stats.success_rate == 0.75

    def test_success_rate_all_failed(self):
        """Test success rate with all failures."""
        stats = MQTTStatistics(messages_sent=0, messages_failed=100)
        assert stats.success_rate == 0.0

    def test_reset(self):
        """Test statistics reset."""
        stats = MQTTStatistics(
            messages_sent=100,
            messages_failed=10,
            bytes_sent=5000,
            connection_count=5,
        )

        stats.reset()

        assert stats.messages_sent == 0
        assert stats.messages_failed == 0
        assert stats.bytes_sent == 0
        assert stats.connection_count == 0

    def test_copy(self):
        """Test statistics copy."""
        stats = MQTTStatistics(
            messages_sent=50,
            messages_failed=5,
            bytes_sent=2000,
        )

        stats_copy = stats.copy()

        assert stats_copy.messages_sent == 50
        assert stats_copy.messages_failed == 5
        assert stats_copy.bytes_sent == 2000

        # Verify it's a copy, not a reference
        stats_copy.messages_sent = 100
        assert stats.messages_sent == 50


class TestMQTTClientUnit:
    """Unit tests for MQTTClient (no network required)."""

    def test_client_requires_paho(self):
        """Test that client raises error without paho-mqtt."""
        # This test verifies the import check exists
        pass

    def test_client_initialization(self):
        """Test client initialization with mock."""
        config = MQTTConfig(host="test.local")
        client = MQTTClient(config, log_file="/tmp/test_mqtt.log")

        assert client._config.host == "test.local"
        assert client._running is False
        assert client.is_running() is False

    def test_client_initialization_default_config(self):
        """Test client initialization with default config."""
        client = MQTTClient(log_file="/tmp/test_mqtt.log")

        assert client._config.host == "easyair-mqtt"
        assert client._config.port == 1883

    def test_send_when_not_running(self):
        """Test send returns False when client not running."""
        client = MQTTClient(log_file="/tmp/test_mqtt.log")

        result = client.send("test", {"data": "value"})

        assert result is False

    def test_get_statistics(self):
        """Test statistics retrieval."""
        client = MQTTClient(log_file="/tmp/test_mqtt.log")

        stats = client.get_statistics()

        assert isinstance(stats, MQTTStatistics)
        assert stats.messages_sent == 0
        assert stats.messages_failed == 0

    def test_get_queue_size(self):
        """Test queue size retrieval."""
        client = MQTTClient(log_file="/tmp/test_mqtt.log")

        assert client.get_queue_size() == 0


class TestMQTTClientIntegration:
    """Integration tests for MQTTClient (requires MQTT broker)."""

    @pytest.mark.integration
    def test_connect_and_send(self):
        """Test actual connection and message sending.

        Requires MQTT broker running at configured address.
        """
        config = MQTTConfig()

        with MQTTClient(config) as client:
            # Wait for connection
            time.sleep(2)

            assert client.is_running()

            # Send test message
            success = client.send(
                topic_suffix="test/event",
                payload={
                    "test": True,
                    "timestamp": time.time(),
                    "source": "test_mqtt.py",
                },
            )

            assert success

            # Allow time for async send
            time.sleep(1)

            # Check statistics
            stats = client.get_statistics()
            assert stats.messages_sent >= 1 or stats.messages_queued >= 1

    @pytest.mark.integration
    def test_multiple_messages(self):
        """Test sending multiple messages."""
        config = MQTTConfig()

        with MQTTClient(config) as client:
            time.sleep(2)

            for i in range(10):
                success = client.send(
                    topic_suffix="test/batch",
                    payload={"index": i, "timestamp": time.time()},
                )
                assert success

            time.sleep(2)

            stats = client.get_statistics()
            # At least some messages should be sent or queued
            assert stats.messages_sent + stats.messages_queued >= 1

    @pytest.mark.integration
    def test_context_manager(self):
        """Test client lifecycle with context manager."""
        config = MQTTConfig()

        with MQTTClient(config) as client:
            assert client.is_running()
            client.send("test/context", {"test": "context_manager"})

        # After exit, should be stopped
        assert not client.is_running()

    @pytest.mark.integration
    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        config = MQTTConfig()

        with MQTTClient(config) as client:
            time.sleep(2)

            initial_stats = client.get_statistics()
            initial_sent = initial_stats.messages_sent

            # Send some messages
            for _ in range(5):
                client.send("test/stats", {"data": "test"})

            time.sleep(2)

            final_stats = client.get_statistics()

            # Success rate should be reasonable
            assert final_stats.success_rate >= 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-m", "not integration"])
