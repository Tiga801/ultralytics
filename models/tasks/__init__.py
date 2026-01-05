# Task-specific predictors

from .base import BasePredictor
from .detect import DetectionPredictor
from .classify import ClassificationPredictor
from .pose import PosePredictor

__all__ = [
    "BasePredictor",
    "DetectionPredictor",
    "ClassificationPredictor",
    "PosePredictor",
]
