# -*- coding: utf-8 -*-
"""MQTT Statistics Module.

Provides statistics tracking for MQTT client operations including message counts,
success rates, and connection events.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MQTTStatistics:
    """MQTT client statistics for monitoring.

    Tracks message delivery statistics and connection events. All counters are
    thread-safe when accessed through MQTTClient methods.

    Attributes:
        messages_sent: Total number of successfully sent messages.
        messages_failed: Total number of failed message sends.
        messages_queued: Current number of messages in the queue.
        bytes_sent: Total bytes of payload data sent.
        last_send_time: Timestamp of the last successful send (Unix timestamp).
        connection_count: Number of successful connections.
        disconnect_count: Number of disconnections.
    """

    messages_sent: int = 0
    messages_failed: int = 0
    messages_queued: int = 0
    bytes_sent: int = 0
    last_send_time: Optional[float] = None
    connection_count: int = 0
    disconnect_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate message delivery success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no messages have been sent.
        """
        total = self.messages_sent + self.messages_failed
        if total == 0:
            return 1.0
        return self.messages_sent / total

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.messages_sent = 0
        self.messages_failed = 0
        self.messages_queued = 0
        self.bytes_sent = 0
        self.last_send_time = None
        self.connection_count = 0
        self.disconnect_count = 0

    def copy(self) -> "MQTTStatistics":
        """Create a copy of current statistics.

        Returns:
            New MQTTStatistics instance with copied values.
        """
        return MQTTStatistics(
            messages_sent=self.messages_sent,
            messages_failed=self.messages_failed,
            messages_queued=self.messages_queued,
            bytes_sent=self.bytes_sent,
            last_send_time=self.last_send_time,
            connection_count=self.connection_count,
            disconnect_count=self.disconnect_count,
        )
