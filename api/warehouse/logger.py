"""Algorithm Warehouse Logger Module.

This module provides a dedicated logging setup for the Algorithm Warehouse
service. All warehouse-related logs are written to a separate log file
(logs/warehouse.log) to maintain independence from the main service logging.

Functions:
    setup_warehouse_logger: Create and configure the warehouse logger.
    get_warehouse_logger: Get or create the singleton warehouse logger.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

# Module-level logger instance (singleton)
_warehouse_logger: Optional[logging.Logger] = None
_log_func: Optional[Callable[[str], None]] = None


def setup_warehouse_logger(log_dir: str = "logs") -> Callable[[str], None]:
    """Create and configure the dedicated warehouse logger.

    This function sets up a file-based logger that writes to logs/warehouse.log.
    The logger is independent from the main application logging and uses a
    consistent format with timestamps and the [WAREHOUSE] tag.

    Args:
        log_dir: Directory path for log files. Created if it doesn't exist.

    Returns:
        A callable that logs messages with the [WAREHOUSE] tag.
    """
    global _warehouse_logger, _log_func

    if _log_func is not None:
        return _log_func

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure the logger
    logger = logging.getLogger("easyair.warehouse")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicates
    logger.handlers.clear()

    # Prevent propagation to root logger
    logger.propagate = False

    # Create file handler for warehouse.log
    log_file = log_path / "warehouse.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Set log format: [timestamp] message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _warehouse_logger = logger

    def log_method(message: str) -> None:
        """Log a message with the [WAREHOUSE] tag.

        Args:
            message: The message to log.
        """
        logger.info(f"[WAREHOUSE] {message}")

    _log_func = log_method
    return log_method


def get_warehouse_logger() -> Callable[[str], None]:
    """Get the singleton warehouse logger function.

    If the logger has not been initialized, this function will set it up
    with the default log directory.

    Returns:
        A callable that logs messages with the [WAREHOUSE] tag.
    """
    global _log_func
    if _log_func is None:
        return setup_warehouse_logger()
    return _log_func


def reset_warehouse_logger() -> None:
    """Reset the warehouse logger for testing purposes.

    This clears the singleton logger instance, allowing setup_warehouse_logger
    to be called again with different parameters.
    """
    global _warehouse_logger, _log_func

    if _warehouse_logger is not None:
        for handler in _warehouse_logger.handlers[:]:
            handler.close()
            _warehouse_logger.removeHandler(handler)

    _warehouse_logger = None
    _log_func = None
