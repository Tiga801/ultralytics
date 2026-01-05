"""Solutions module - Concrete task implementations.

This module provides ready-to-use computer vision task implementations
that extend the TaskBase class.

Available Tasks:
    - TestTask: Detect objects crossing a defined line

Usage:
    Tasks are automatically registered with the TaskRegistry via decorators.
    Create tasks through the TaskRegistry.create() method.

    >>> from task import TaskRegistry
    >>> task = TaskRegistry.create("test_task", "task_001")

Or import specific task classes:

    >>> from solutions import TestTask
"""

from .test_task import TestTask

__all__ = [
    "TestTask"
]
