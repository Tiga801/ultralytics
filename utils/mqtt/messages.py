# -*- coding: utf-8 -*-
"""MQTT Message Module.

Provides message data structures for MQTT publishing operations.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MQTTMessage:
    """MQTT message structure for publishing.

    Represents a message to be sent via MQTT. The full topic is constructed
    by combining the client's topic_prefix with the message's topic_suffix.

    Attributes:
        topic_suffix: Topic suffix to append to the client's topic prefix.
        payload: Message payload as a dictionary (will be JSON serialized).
        qos: Quality of Service level (0, 1, or 2).
        retain: Whether the broker should retain this message.
        message_id: Unique message identifier (auto-generated).
        timestamp: Message creation timestamp (auto-generated).
    """

    topic_suffix: str
    payload: Dict[str, Any]
    qos: int = 1
    retain: bool = False
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)

    def get_full_topic(self, prefix: str) -> str:
        """Construct full topic from prefix and suffix.

        Args:
            prefix: Topic prefix to prepend.

        Returns:
            Full topic string in format '{prefix}/{suffix}'.
        """
        if not prefix:
            return self.topic_suffix
        return f"{prefix}/{self.topic_suffix}"

    def to_json(self) -> str:
        """Serialize payload to JSON string.

        Returns:
            JSON string representation of the payload.
        """
        return json.dumps(self.payload, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation.

        Returns:
            Dictionary containing all message fields.
        """
        return {
            "topic_suffix": self.topic_suffix,
            "payload": self.payload,
            "qos": self.qos,
            "retain": self.retain,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
        }
