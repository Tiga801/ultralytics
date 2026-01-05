"""Thread-safe singleton metaclass.

This module provides a metaclass for creating singleton classes that are
thread-safe and can be used across the application.
"""

import threading
from typing import Any, Dict


class SingletonMeta(type):
    """Thread-safe singleton metaclass.

    This metaclass ensures that only one instance of a class exists,
    even when accessed from multiple threads simultaneously.

    Example:
        >>> class MyClass(metaclass=SingletonMeta):
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> obj1 = MyClass(1)
        >>> obj2 = MyClass(2)
        >>> obj1 is obj2
        True
        >>> obj1.value
        1
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """Create or return the singleton instance.

        Args:
            *args: Positional arguments for class initialization.
            **kwargs: Keyword arguments for class initialization.

        Returns:
            The singleton instance of the class.
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def clear_instance(mcs, cls: type) -> None:
        """Clear the singleton instance for a specific class.

        This is useful for testing or resetting state.

        Args:
            cls: The class whose instance should be cleared.
        """
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]

    @classmethod
    def clear_all_instances(mcs) -> None:
        """Clear all singleton instances.

        This is useful for testing or complete reset.
        """
        with mcs._lock:
            mcs._instances.clear()
