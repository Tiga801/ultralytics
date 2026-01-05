# -*- coding: utf-8 -*-
"""MQTT Client Module.

Provides a process-safe MQTT client with asynchronous message queue.
Each task process should create its own MQTTClient instance.
"""

import logging
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, Optional
import paho.mqtt.client as mqtt

from .config import MQTTConfig
from .messages import MQTTMessage
from .stats import MQTTStatistics


class MQTTClient:
    """Process-safe MQTT client with asynchronous message queue.

    Designed for per-process usage in a multi-process architecture.
    Each task process should create and manage its own MQTTClient instance.

    Features:
    - Background sender thread for non-blocking message publishing
    - Automatic reconnection via paho-mqtt's loop_start()
    - Thread-safe statistics tracking
    - Dedicated logging to logs/mqtt.log
    - Context manager support

    Example:
        >>> from utils.mqtt import MQTTClient, MQTTConfig
        >>> config = MQTTConfig(host="localhost", port=1883)
        >>> with MQTTClient(config) as client:
        ...     client.send("events", {"event": "test", "value": 123})
    """

    def __init__(
        self,
        config: Optional[MQTTConfig] = None,
        log_file: str = "logs/mqtt.log",
    ):
        """Initialize MQTT client.

        Args:
            config: MQTT configuration. Uses default configuration if None.
            log_file: Path to MQTT log file.

        Raises:
            ImportError: If paho-mqtt is not installed.
        """
        self._config = config or MQTTConfig()
        self._config.validate()

        # Generate client ID if not provided
        self._client_id = (
            self._config.client_id or f"mqtt_{uuid.uuid4().hex[:8]}"
        )

        # MQTT client instance
        self._client: Optional[mqtt.Client] = None

        # Message queue for async sending
        self._message_queue: queue.Queue = queue.Queue(
            maxsize=self._config.queue_max_size
        )

        # Threading controls
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._running = False

        # Statistics with thread-safe access
        self._stats = MQTTStatistics()
        self._stats_lock = threading.Lock()

        # Setup logging
        self._logger = self._setup_logger(log_file)
        self._log_info(
            f"Client initialized - broker: {self._config.host}:{self._config.port}, "
            f"client_id: {self._client_id}"
        )

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Create dedicated MQTT logger.

        Args:
            log_file: Path to log file.

        Returns:
            Configured logger instance.
        """
        # Create unique logger name to avoid conflicts
        logger = logging.getLogger(f"mqtt.{self._client_id}")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # File handler
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console if file logging fails
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
            logger.warning(f"Failed to setup file logging: {e}")

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def _log_info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    # === Lifecycle Methods ===

    def start(self) -> bool:
        """Connect to broker and start background sender thread.

        Returns:
            True if connected and started successfully, False otherwise.
        """
        if self._running:
            self._log_warning("Client is already running")
            return True

        try:
            # Initialize MQTT client
            self._init_client()

            # Connect to broker
            if not self._connect():
                self._log_error("Failed to connect to broker")
                return False

            # Start sender thread
            self._stop_event.clear()
            self._running = True
            self._sender_thread = threading.Thread(
                target=self._sender_loop,
                daemon=True,
                name=f"MQTTSender-{self._client_id}",
            )
            self._sender_thread.start()

            self._log_info("Client started successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to start client: {e}")
            self._running = False
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Stop sender thread and disconnect from broker.

        Drains remaining messages before stopping if possible.

        Args:
            timeout: Maximum time in seconds to wait for thread to stop.
        """
        self._log_info("Stopping client...")

        self._running = False
        self._stop_event.set()

        # Send poison pill to wake up blocked queue.get()
        try:
            self._message_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for sender thread to finish
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=timeout)
            if self._sender_thread.is_alive():
                self._log_warning("Sender thread did not stop gracefully")

        # Disconnect MQTT client
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as e:
                self._log_error(f"Error during disconnect: {e}")
            self._client = None

        self._connected_event.clear()
        self._log_info("Client stopped")

    # === Message Sending ===

    def send(
        self,
        topic_suffix: str,
        payload: Dict[str, Any],
        qos: Optional[int] = None,
    ) -> bool:
        """Queue a message for asynchronous delivery.

        Messages are queued and sent by a background thread. This method
        never blocks the caller for network I/O.

        Args:
            topic_suffix: Topic suffix to append to the configured topic_prefix.
            payload: JSON-serializable message payload.
            qos: MQTT QoS level (0, 1, or 2). Uses default_qos if None.

        Returns:
            True if message was queued successfully, False if queue is full
            or client is not running.
        """
        if not self._running:
            self._log_warning("Cannot send: client not running")
            return False

        if qos is None:
            qos = self._config.default_qos

        message = MQTTMessage(
            topic_suffix=topic_suffix,
            payload=payload,
            qos=qos,
        )

        try:
            self._message_queue.put_nowait(message)
            with self._stats_lock:
                self._stats.messages_queued = self._message_queue.qsize()
            self._log_debug(f"Message queued: {topic_suffix}")
            return True
        except queue.Full:
            self._log_warning(
                f"Message queue full ({self._config.queue_max_size}), "
                f"message dropped: {topic_suffix}"
            )
            with self._stats_lock:
                self._stats.messages_failed += 1
            return False

    # === Status & Statistics ===

    def is_connected(self) -> bool:
        """Check if client is connected to broker.

        Returns:
            True if connected, False otherwise.
        """
        return self._client is not None and self._client.is_connected()

    def is_running(self) -> bool:
        """Check if sender thread is running.

        Returns:
            True if running, False otherwise.
        """
        return self._running

    def get_queue_size(self) -> int:
        """Get current message queue depth.

        Returns:
            Number of messages waiting in the queue.
        """
        return self._message_queue.qsize()

    def get_statistics(self) -> MQTTStatistics:
        """Get copy of current statistics.

        Returns a snapshot of statistics that is safe to use without locks.

        Returns:
            Copy of current MQTTStatistics.
        """
        with self._stats_lock:
            self._stats.messages_queued = self._message_queue.qsize()
            return self._stats.copy()

    # === Context Manager ===

    def __enter__(self) -> "MQTTClient":
        """Enter context manager, start client."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, stop client."""
        self.stop()

    # === Private Methods ===

    def _init_client(self) -> None:
        """Initialize paho-mqtt client instance."""
        # Use callback API version 2 for newer paho-mqtt (2.0+)
        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self._client_id,
            )
        except (TypeError, AttributeError):
            # Fallback for older paho-mqtt versions (< 2.0)
            self._client = mqtt.Client(client_id=self._client_id)

        # Set authentication if provided
        if self._config.username:
            self._client.username_pw_set(
                self._config.username,
                self._config.password,
            )

        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _connect(self) -> bool:
        """Connect to MQTT broker.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._client.connect(
                self._config.host,
                self._config.port,
                keepalive=self._config.keepalive,
            )

            # Start network loop (handles reconnection automatically)
            self._client.loop_start()

            # Wait for connection with timeout
            start_time = time.time()
            while not self._client.is_connected():
                if time.time() - start_time > self._config.connect_timeout:
                    self._log_error("Connection timeout")
                    return False
                time.sleep(0.1)

            with self._stats_lock:
                self._stats.connection_count += 1

            self._connected_event.set()
            return True

        except Exception as e:
            self._log_error(f"Connection failed: {e}")
            return False

    def _sender_loop(self) -> None:
        """Background thread loop for sending queued messages."""
        self._log_debug("Sender loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Get message with timeout to allow checking stop event
                message = self._message_queue.get(timeout=1.0)

                # Check for poison pill
                if message is None:
                    continue

                if isinstance(message, MQTTMessage):
                    self._publish_message(message)

            except queue.Empty:
                continue
            except Exception as e:
                self._log_error(f"Sender loop error: {e}")

        self._log_debug("Sender loop stopped")

    def _publish_message(self, message: MQTTMessage) -> bool:
        """Publish a single message to the broker.

        Args:
            message: Message to publish.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._client or not self._client.is_connected():
            self._log_warning("Cannot publish: not connected")
            with self._stats_lock:
                self._stats.messages_failed += 1
            return False

        try:
            # Construct full topic
            full_topic = message.get_full_topic(self._config.topic_prefix)
            payload_json = message.to_json()

            # Publish message
            result = self._client.publish(
                full_topic,
                payload_json,
                qos=message.qos,
                retain=message.retain,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                with self._stats_lock:
                    self._stats.messages_sent += 1
                    self._stats.bytes_sent += len(payload_json.encode("utf-8"))
                    self._stats.last_send_time = time.time()
                self._log_debug(
                    f"Message sent: {full_topic} (id: {message.message_id})"
                )
                return True
            else:
                with self._stats_lock:
                    self._stats.messages_failed += 1
                self._log_error(f"Publish failed with rc={result.rc}")
                return False

        except Exception as e:
            with self._stats_lock:
                self._stats.messages_failed += 1
            self._log_error(f"Publish error: {e}")
            return False

    # === MQTT Callbacks ===

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """Handle successful connection.

        Supports both paho-mqtt v1 and v2 callback signatures.
        """
        # Handle ConnectFlags object from v2 API
        reason_code = rc
        if hasattr(rc, "value"):
            reason_code = rc.value

        if reason_code == 0:
            self._log_info(
                f"Connected to {self._config.host}:{self._config.port}"
            )
            self._connected_event.set()
        else:
            self._log_error(f"Connection failed with rc={reason_code}")

    def _on_disconnect(self, client, userdata, *args) -> None:
        """Handle disconnection.

        Supports both paho-mqtt v1 and v2 callback signatures:
        - v1: (client, userdata, rc)
        - v2: (client, userdata, disconnect_flags, reason_code, properties)
        """
        self._connected_event.clear()
        with self._stats_lock:
            self._stats.disconnect_count += 1

        # Extract reason code from args
        # v1 API: args = (rc,)
        # v2 API: args = (disconnect_flags, reason_code, properties)
        reason_code = 0
        if len(args) >= 2:
            # v2 API - reason_code is second argument
            rc = args[1]
            reason_code = rc.value if hasattr(rc, "value") else rc
        elif len(args) == 1:
            # v1 API - rc is first argument
            rc = args[0]
            reason_code = rc.value if hasattr(rc, "value") else rc

        if reason_code == 0:
            self._log_info("Disconnected normally")
        else:
            self._log_warning(
                f"Unexpected disconnect (rc={reason_code}), auto-reconnect enabled"
            )
