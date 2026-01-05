# -*- coding: utf-8 -*-
"""MQTT Configuration Module.

Provides type-safe configuration management for MQTT client connections.
Supports creation from dictionaries for integration with JSON configuration files.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MQTTConfig:
    """MQTT client configuration.

    Attributes:
        host: MQTT broker hostname or IP address.
        port: MQTT broker port number.
        username: Authentication username (optional).
        password: Authentication password (optional).
        topic_prefix: Prefix prepended to all topic suffixes.
        client_id: Unique client identifier (auto-generated if None).
        keepalive: Connection keepalive interval in seconds.
        connect_timeout: Maximum time to wait for connection in seconds.
        reconnect_delay: Delay between reconnection attempts in seconds.
        max_reconnect_attempts: Maximum reconnection attempts (0 = infinite).
        queue_max_size: Maximum number of messages in send queue.
        default_qos: Default Quality of Service level (0, 1, or 2).
    """

    # Connection settings
    host: str = "easyair-mqtt"
    port: int = 1883
    username: str = "root"
    password: str = "P&5x19k@G3dw"

    # Topic settings
    topic_prefix: str = "assup/3379712260089733377/base2/subAttr"

    # Client identification
    client_id: Optional[str] = None

    # Connection parameters
    keepalive: int = 60
    connect_timeout: int = 30
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 0  # 0 = infinite

    # Queue settings
    queue_max_size: int = 1000
    default_qos: int = 1

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.host:
            raise ValueError("host cannot be empty")

        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")

        if self.default_qos not in (0, 1, 2):
            raise ValueError(f"default_qos must be 0, 1, or 2, got {self.default_qos}")

        if self.queue_max_size <= 0:
            raise ValueError(
                f"queue_max_size must be positive, got {self.queue_max_size}"
            )

        if self.keepalive <= 0:
            raise ValueError(f"keepalive must be positive, got {self.keepalive}")

        if self.connect_timeout <= 0:
            raise ValueError(
                f"connect_timeout must be positive, got {self.connect_timeout}"
            )

        if self.reconnect_delay < 0:
            raise ValueError(
                f"reconnect_delay must be non-negative, got {self.reconnect_delay}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "topic_prefix": self.topic_prefix,
            "client_id": self.client_id,
            "keepalive": self.keepalive,
            "connect_timeout": self.connect_timeout,
            "reconnect_delay": self.reconnect_delay,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "queue_max_size": self.queue_max_size,
            "default_qos": self.default_qos,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MQTTConfig":
        """Create configuration from dictionary.

        Supports legacy field names for backward compatibility:
        - 'broker_host' -> 'host'
        - 'broker_port' -> 'port'
        - 'topic' -> extracts prefix if contains '/'

        Args:
            data: Configuration dictionary.

        Returns:
            MQTTConfig instance.
        """
        # Handle legacy field names
        host = data.get("host") or data.get("broker_host", "easyair-mqtt")
        port = data.get("port") or data.get("broker_port", 1883)

        # Handle topic_prefix from legacy 'topic' field
        topic_prefix = data.get("topic_prefix")
        if topic_prefix is None:
            topic = data.get("topic", "")
            if topic and "/" in topic:
                topic_prefix = topic
            else:
                topic_prefix = "assup/3379712260089733377/base2/subAttr"

        return cls(
            host=str(host),
            port=int(port),
            username=data.get("username", "root"),
            password=data.get("password", "P&5x19k@G3dw"),
            topic_prefix=topic_prefix,
            client_id=data.get("client_id"),
            keepalive=int(data.get("keepalive", 60)),
            connect_timeout=int(data.get("connect_timeout", 30)),
            reconnect_delay=float(data.get("reconnect_delay", 5.0)),
            max_reconnect_attempts=int(data.get("max_reconnect_attempts", 0)),
            queue_max_size=int(data.get("queue_max_size", 1000)),
            default_qos=int(data.get("default_qos", 1)),
        )
