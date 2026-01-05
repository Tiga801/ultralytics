"""Centralized logging utilities.

This module provides a unified logging interface for the task management system.

Logging Architecture:
- Main service log: easyair_YYYYMMDD_HHMMSS.log (console + file)
- Request log: requests.log (console + file, never cleaned)
- Stream log: streams.log (file only)
- Task logs: {stand_name}-{camera_name}-{task_id}.log (file only)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional


class Logger:
    """Centralized logger with tag-based logging support.

    This class provides a unified logging interface that supports:
    - Tag-based log messages for easy filtering
    - File and console output
    - Per-component loggers
    - Task-specific log files
    - Request logging (with persistence)
    - Stream processing logging

    Example:
        >>> Logger.init(log_dir="logs")
        >>> log = Logger.get_logging_method("ENGINE")
        >>> log("Engine started successfully")
        [2024-01-01 12:00:00] [ENGINE] Engine started successfully
    """

    _logger: Optional[logging.Logger] = None
    _log_dir: Optional[Path] = None
    _component_loggers: Dict[str, logging.Logger] = {}
    _task_loggers: Dict[str, logging.Logger] = {}
    _request_logger: Optional[logging.Logger] = None
    _stream_logger: Optional[logging.Logger] = None
    _initialized: bool = False
    _formatter: Optional[logging.Formatter] = None

    @classmethod
    def init(
        cls,
        log_dir: str = "logs",
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True,
    ) -> None:
        """Initialize the logging system.

        Args:
            log_dir: Directory for log files.
            level: Logging level (default INFO).
            console: Whether to output to console.
            file: Whether to output to file.
        """
        if cls._initialized:
            return

        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        # Create root logger
        cls._logger = logging.getLogger("easyair")
        cls._logger.setLevel(level)
        cls._logger.handlers.clear()

        # Log format
        cls._formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(cls._formatter)
            cls._logger.addHandler(console_handler)

        # File handler
        if file:
            log_file = cls._log_dir / f"easyair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(cls._formatter)
            cls._logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def get_logging_method(
        cls,
        tag_name: str,
        log_name: Optional[str] = None
    ) -> Callable[[str], None]:
        """Get a logging method with a specific tag.

        Args:
            tag_name: Tag to prepend to log messages.
            log_name: Optional logger name for component-specific logging.

        Returns:
            A callable that logs messages with the specified tag.

        Example:
            >>> log = Logger.get_logging_method("CAMERA")
            >>> log("Camera connected")
            [2024-01-01 12:00:00] [CAMERA] Camera connected
        """
        if not cls._initialized:
            cls.init()

        logger = cls._logger
        if log_name and log_name in cls._component_loggers:
            logger = cls._component_loggers[log_name]

        def log_method(message: str) -> None:
            logger.info(f"[{tag_name}] {message}")

        return log_method

    @classmethod
    def register_component_logger(
        cls,
        name: str,
        log_file: Optional[str] = None
    ) -> None:
        """Register a component-specific logger.

        Args:
            name: Component name.
            log_file: Optional separate log file for this component.
        """
        if not cls._initialized:
            cls.init()

        if name in cls._component_loggers:
            return

        logger = logging.getLogger(f"easyair.{name}")
        logger.setLevel(cls._logger.level)

        # Inherit handlers from root logger
        for handler in cls._logger.handlers:
            logger.addHandler(handler)

        # Add component-specific file handler if specified
        if log_file and cls._log_dir:
            file_path = cls._log_dir / log_file
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(cls._logger.level)
            file_handler.setFormatter(cls._logger.handlers[0].formatter)
            logger.addHandler(file_handler)

        cls._component_loggers[name] = logger

    @classmethod
    def title(cls, message: str, char: str = "=", width: int = 60) -> None:
        """Log a title message with decorative borders.

        Args:
            message: Title message.
            char: Character for border.
            width: Total width of the title.
        """
        if not cls._initialized:
            cls.init()

        border = char * width
        centered = message.center(width)
        cls._logger.info(border)
        cls._logger.info(centered)
        cls._logger.info(border)

    @classmethod
    def get_task_logger(
        cls,
        stand_name: str,
        camera_name: str,
        task_id: str
    ) -> Callable[[str], None]:
        """Get a logger for task-specific log file.

        Creates a log file: logs/{stand_name}-{camera_name}-{task_id}.log
        - File output only (no console)
        - Independent from main service log
        - Records task-specific runtime information, exceptions, and state changes

        Args:
            stand_name: Stand/location name.
            camera_name: Camera name.
            task_id: Task identifier.

        Returns:
            A callable that logs messages to the task-specific file.

        Example:
            >>> log = Logger.get_task_logger("stand1", "camera1", "task001")
            >>> log("Task started processing")
        """
        if not cls._initialized:
            cls.init()

        # Create unique logger key
        logger_key = f"{stand_name}-{camera_name}-{task_id}"

        # Return existing logger if already created
        if logger_key in cls._task_loggers:
            logger = cls._task_loggers[logger_key]

            def log_method(message: str) -> None:
                logger.info(f"[TASK-{task_id[:18]}] {message}")

            return log_method

        # Create new task-specific logger
        logger = logging.getLogger(f"easyair.task.{logger_key}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False  # Don't propagate to main logger

        # File handler only (no console)
        if cls._log_dir:
            log_file = cls._log_dir / f"{logger_key}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(cls._formatter or logging.Formatter(
                "[%(asctime)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(file_handler)

        cls._task_loggers[logger_key] = logger

        def log_method(message: str) -> None:
            logger.info(f"[TASK-{task_id[:18]}] {message}")

        return log_method

    @classmethod
    def get_request_logger(cls) -> Callable[[str], None]:
        """Get a logger for API requests.

        Creates: logs/requests.log
        - Both file and console output
        - Also outputs to main service log
        - No time-based rotation
        - Never cleaned/deleted (persistent)

        Returns:
            A callable that logs request messages.

        Example:
            >>> request_log = Logger.get_request_logger()
            >>> request_log("START_TASK task_001")
        """
        if not cls._initialized:
            cls.init()

        # Return existing logger if already created
        if cls._request_logger:
            def log_method(message: str) -> None:
                cls._request_logger.info(f"[REQUEST] {message}")
                # Also log to main service log
                if cls._logger:
                    cls._logger.info(f"[REQUEST] {message}")

            return log_method

        # Create request logger
        logger = logging.getLogger("easyair.requests")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False  # Don't propagate to root

        formatter = cls._formatter or logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler (append mode, never rotate)
        if cls._log_dir:
            log_file = cls._log_dir / "requests.log"
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._request_logger = logger

        def log_method(message: str) -> None:
            cls._request_logger.info(f"[REQUEST] {message}")
            # Also log to main service log
            if cls._logger:
                cls._logger.info(f"[REQUEST] {message}")

        return log_method

    @classmethod
    def get_stream_logger(cls) -> Callable[[str], None]:
        """Get a logger for stream processing.

        Creates: logs/streams.log
        - File output only (no console)
        - Independent from main service log
        - Records stream events: disconnection, instability, restart

        Returns:
            A callable that logs stream messages.

        Example:
            >>> stream_log = Logger.get_stream_logger()
            >>> stream_log("Stream 0 disconnected")
        """
        if not cls._initialized:
            cls.init()

        # Return existing logger if already created
        if cls._stream_logger:
            def log_method(message: str) -> None:
                cls._stream_logger.info(f"[STREAM] {message}")

            return log_method

        # Create stream logger
        logger = logging.getLogger("easyair.streams")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False  # Don't propagate to main logger

        formatter = cls._formatter or logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler only (no console)
        if cls._log_dir:
            log_file = cls._log_dir / "streams.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._stream_logger = logger

        def log_method(message: str) -> None:
            cls._stream_logger.info(f"[STREAM] {message}")

        return log_method

    @classmethod
    def get_log_dir(cls) -> Optional[Path]:
        """Get the log directory path.

        Returns:
            Path to log directory or None if not initialized.
        """
        return cls._log_dir


def get_logger(name: str = "easyair") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if not Logger._initialized:
        Logger.init()
    return logging.getLogger(name)
