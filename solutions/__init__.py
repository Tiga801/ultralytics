"""Solutions module - Concrete task implementations.

This module provides ready-to-use computer vision task implementations
that extend the TaskBase class.

Available Tasks:
    - CrossLineDetectionTask: Detect objects crossing a defined line
    - RegionIntrusionTask: Detect objects entering restricted areas
    - FaceDetectionTask: Face detection and optional recognition
    - CrowdDensityTask: Crowd density estimation and monitoring

Usage:
    Tasks are automatically registered with the TaskRegistry via decorators.
    Create tasks through the TaskRegistry.create() method.

    >>> from task import TaskRegistry
    >>> task = TaskRegistry.create("cross_line", "task_001")

Or import specific task classes:

    >>> from solutions import CrossLineDetectionTask
"""

from .cross_line import CrossLineDetectionTask
from .region_intrusion import RegionIntrusionTask
from .face_detection import FaceDetectionTask
from .crowd_density import CrowdDensityTask

__all__ = [
    "CrossLineDetectionTask",
    "RegionIntrusionTask",
    "FaceDetectionTask",
    "CrowdDensityTask",
]
