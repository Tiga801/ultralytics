"""Solutions module - Concrete task implementations.

This module provides ready-to-use computer vision task implementations
that extend the TaskBase class.

Available Tasks:
    - SamplingDetectionTask: Intelligent video frame sampling with MinIO upload

Usage:
    Tasks are automatically registered with the TaskRegistry via decorators.
    Create tasks through the TaskRegistry.create() method.

    >>> from task import TaskRegistry
    >>> task = TaskRegistry.create("sampling", "task_001")

Or import specific task classes:

    >>> from solutions import SamplingDetectionTask
"""

from .sampling_detection import SamplingDetectionTask

__all__ = [
    "SamplingDetectionTask",
]
