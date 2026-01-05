"""Task registry with factory pattern.

This module provides the TaskRegistry class for registering and creating
task instances based on task type.
"""

import threading
from typing import Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TaskBase


class TaskRegistry:
    """Factory pattern registry for task types.

    This class manages the registration and creation of task classes,
    allowing new task types to be added via decorator.

    Example:
        >>> @TaskRegistry.register("cross_line")
        ... class CrossLineTask(TaskBase):
        ...     pass
        ...
        >>> task = TaskRegistry.create("cross_line", "task_001")
    """

    _registry: Dict[str, Type["TaskBase"]] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, task_type: str) -> Callable[[Type["TaskBase"]], Type["TaskBase"]]:
        """Decorator to register a task class.

        Args:
            task_type: Task type identifier (e.g., "cross_line").

        Returns:
            Decorator function.

        Example:
            >>> @TaskRegistry.register("my_task")
            ... class MyTask(TaskBase):
            ...     pass
        """
        def decorator(task_cls: Type["TaskBase"]) -> Type["TaskBase"]:
            with cls._lock:
                cls._registry[task_type] = task_cls
            return task_cls
        return decorator

    @classmethod
    def register_class(cls, task_type: str, task_cls: Type["TaskBase"]) -> None:
        """Register a task class programmatically.

        Args:
            task_type: Task type identifier.
            task_cls: Task class to register.
        """
        with cls._lock:
            cls._registry[task_type] = task_cls

    @classmethod
    def get_class(cls, task_type: str) -> Optional[Type["TaskBase"]]:
        """Get the task class for a given type.

        Args:
            task_type: Task type identifier.

        Returns:
            Task class or None if not found.
        """
        with cls._lock:
            return cls._registry.get(task_type)

    @classmethod
    def create(cls, task_type: str, task_id: str) -> Optional["TaskBase"]:
        """Create a task instance.

        Args:
            task_type: Task type identifier.
            task_id: Unique task identifier.

        Returns:
            Task instance or None if type not found.

        Raises:
            ValueError: If task type is not registered.
        """
        task_cls = cls.get_class(task_type)
        if task_cls is None:
            raise ValueError(f"Unknown task type: {task_type}. "
                           f"Available types: {cls.get_supported_types()}")
        return task_cls(task_id)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of registered task types.

        Returns:
            List of task type identifiers.
        """
        with cls._lock:
            return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, task_type: str) -> bool:
        """Check if a task type is registered.

        Args:
            task_type: Task type identifier.

        Returns:
            True if registered.
        """
        with cls._lock:
            return task_type in cls._registry

    @classmethod
    def unregister(cls, task_type: str) -> bool:
        """Unregister a task type.

        Args:
            task_type: Task type identifier.

        Returns:
            True if unregistered, False if not found.
        """
        with cls._lock:
            if task_type in cls._registry:
                del cls._registry[task_type]
                return True
            return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered task types.

        WARNING: Use only for testing.
        """
        with cls._lock:
            cls._registry.clear()
