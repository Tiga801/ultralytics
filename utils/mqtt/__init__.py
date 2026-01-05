# -*- coding: utf-8 -*-
"""MQTT Client Module.

Provides a process-safe MQTT client with asynchronous message queue for
publishing messages to an MQTT broker.

Example:
    Basic usage with context manager::

        from utils.mqtt import MQTTClient, MQTTConfig

        config = MQTTConfig(host="localhost", port=1883)
        with MQTTClient(config) as client:
            client.send("events/alert", {"type": "motion", "zone": "entrance"})

    Manual lifecycle management::

        from utils.mqtt import MQTTClient

        client = MQTTClient()  # Uses default configuration
        client.start()
        try:
            client.send("status", {"online": True})
        finally:
            client.stop()

    Creating configuration from dictionary::

        from utils.mqtt import MQTTClient, MQTTConfig

        config_dict = {
            "host": "mqtt.example.com",
            "port": 1883,
            "username": "user",
            "password": "secret",
        }
        config = MQTTConfig.from_dict(config_dict)
        client = MQTTClient(config)
"""

from .client import MQTTClient
from .config import MQTTConfig
from .messages import MQTTMessage
from .stats import MQTTStatistics

__all__ = [
    "MQTTClient",
    "MQTTConfig",
    "MQTTMessage",
    "MQTTStatistics",
]

__version__ = "1.0.0"
