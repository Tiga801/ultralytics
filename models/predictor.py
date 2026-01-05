"""
Predictor - Main inference API with unified interface.
No dependencies on ultralytics internals.
"""

from typing import Dict, List, Optional, Type, Union

import numpy as np

from .backend import ModelBackend
from .results import Results
from .tasks.base import BasePredictor
from .tasks.classify import ClassificationPredictor
from .tasks.detect import DetectionPredictor
from .tasks.pose import PosePredictor


# Task to predictor class mapping
TASK_MAP: Dict[str, Type[BasePredictor]] = {
    "detect": DetectionPredictor,
    "classify": ClassificationPredictor,
    "pose": PosePredictor,
}


def create_predictor(
    model_path: str,
    task: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> BasePredictor:
    """
    Create a task-specific predictor.

    Factory function that auto-detects task from model metadata
    and creates the appropriate predictor.

    Args:
        model_path: Path to model weights
        task: Task type ('detect', 'classify', 'pose'). Auto-detected if None.
        device: Device for inference ('cuda', 'cpu')
        **kwargs: Additional predictor arguments

    Returns:
        Task-specific predictor instance

    Example:
        >>> predictor = create_predictor("yolo11n.pt")
        >>> results = predictor("image.jpg")
    """
    # Auto-detect task if not specified
    if task is None:
        # Temporarily load model to get task
        backend = ModelBackend(model_path, device=device)
        task = backend.task or "detect"
        del backend  # Free memory

    # Validate task
    task = task.lower()
    if task not in TASK_MAP:
        raise ValueError(f"Unknown task: {task}. Supported: {list(TASK_MAP.keys())}")

    # Create predictor
    predictor_cls = TASK_MAP[task]
    return predictor_cls(model_path=model_path, device=device, **kwargs)


class Predictor:
    """
    Unified predictor interface for YOLO models.

    Automatically detects model task and uses the appropriate predictor.
    Supports PyTorch (.pt), ONNX (.onnx), and TensorRT (.engine) models.

    Args:
        model_path: Path to model weights
        task: Task type ('detect', 'classify', 'pose'). Auto-detected if None.
        device: Device for inference ('cuda', 'cuda:0', 'cpu')
        conf: Confidence threshold (default 0.25)
        iou: IoU threshold for NMS (default 0.45)
        max_det: Maximum detections per image (default 300)
        fp16: Use FP16 inference
        fuse: Fuse Conv+BN layers (PyTorch only)

    Example:
        >>> # Basic usage
        >>> predictor = Predictor("yolo11n.pt")
        >>> results = predictor("image.jpg")
        >>> results[0].plot()

        >>> # With custom thresholds
        >>> predictor = Predictor("yolo11n.pt", conf=0.5, iou=0.6)
        >>> results = predictor(["img1.jpg", "img2.jpg"])

        >>> # Pose estimation
        >>> predictor = Predictor("yolo11n-pose.pt", task="pose")
        >>> results = predictor("person.jpg")
        >>> print(results[0].keypoints)

        >>> # Classification
        >>> predictor = Predictor("yolo11n-cls.pt", task="classify")
        >>> results = predictor("dog.jpg")
        >>> print(results[0].probs.top5)
    """

    def __init__(
        self,
        model_path: str,
        task: Optional[str] = None,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        fp16: bool = False,
        fuse: bool = True,
    ):
        """Initialize predictor."""
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.fp16 = fp16
        self.fuse = fuse

        # Auto-detect task
        if task is None:
            backend = ModelBackend(model_path, device=device, fp16=fp16, fuse=fuse)
            task = backend.task or "detect"
            self._backend = backend
        else:
            self._backend = None

        self.task = task.lower()

        # Create internal predictor
        self._predictor = self._create_predictor()

        # Expose model properties
        self.names = self._predictor.names
        self.stride = self._predictor.stride
        self.imgsz = self._predictor.imgsz
        self.nc = self._predictor.nc

    def _create_predictor(self) -> BasePredictor:
        """Create the task-specific predictor."""
        if self.task not in TASK_MAP:
            raise ValueError(f"Unknown task: {self.task}. Supported: {list(TASK_MAP.keys())}")

        predictor_cls = TASK_MAP[self.task]

        # Build kwargs based on task
        kwargs = {
            "model_path": self.model_path,
            "device": self.device,
            "fp16": self.fp16,
            "fuse": self.fuse,
        }

        if self.task in ("detect", "pose"):
            kwargs.update({
                "conf": self.conf,
                "iou": self.iou,
                "max_det": self.max_det,
            })

        return predictor_cls(**kwargs)

    def __call__(
        self,
        source: Union[str, np.ndarray, List[str], List[np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs,
    ) -> List[Results]:
        """
        Run inference on source.

        Args:
            source: Image path(s) or numpy array(s) in BGR format
            conf: Override confidence threshold
            iou: Override IoU threshold
            **kwargs: Additional arguments

        Returns:
            List of Results objects
        """
        return self._predictor(source, conf=conf, iou=iou, **kwargs)

    def predict(
        self,
        source: Union[str, np.ndarray, List[str], List[np.ndarray]],
        **kwargs,
    ) -> List[Results]:
        """
        Alias for __call__.

        Args:
            source: Image path(s) or numpy array(s)
            **kwargs: Additional arguments

        Returns:
            List of Results objects
        """
        return self(source, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Predictor(model={self.model_path}, task={self.task}, "
            f"device={self.device}, conf={self.conf}, iou={self.iou})"
        )


# Convenience aliases
YOLO = Predictor


def detect(
    source: Union[str, np.ndarray, List[str], List[np.ndarray]],
    model_path: str,
    device: str = "cuda",
    conf: float = 0.25,
    iou: float = 0.45,
    **kwargs,
) -> List[Results]:
    """
    Convenience function for object detection.

    Args:
        source: Image path(s) or numpy array(s)
        model_path: Path to detection model
        device: Inference device
        conf: Confidence threshold
        iou: IoU threshold
        **kwargs: Additional arguments

    Returns:
        List of Results objects
    """
    predictor = Predictor(model_path, task="detect", device=device, conf=conf, iou=iou, **kwargs)
    return predictor(source)


def classify(
    source: Union[str, np.ndarray, List[str], List[np.ndarray]],
    model_path: str,
    device: str = "cuda",
    **kwargs,
) -> List[Results]:
    """
    Convenience function for image classification.

    Args:
        source: Image path(s) or numpy array(s)
        model_path: Path to classification model
        device: Inference device
        **kwargs: Additional arguments

    Returns:
        List of Results objects
    """
    predictor = Predictor(model_path, task="classify", device=device, **kwargs)
    return predictor(source)


def pose(
    source: Union[str, np.ndarray, List[str], List[np.ndarray]],
    model_path: str,
    device: str = "cuda",
    conf: float = 0.25,
    iou: float = 0.45,
    **kwargs,
) -> List[Results]:
    """
    Convenience function for pose estimation.

    Args:
        source: Image path(s) or numpy array(s)
        model_path: Path to pose model
        device: Inference device
        conf: Confidence threshold
        iou: IoU threshold
        **kwargs: Additional arguments

    Returns:
        List of Results objects
    """
    predictor = Predictor(model_path, task="pose", device=device, conf=conf, iou=iou, **kwargs)
    return predictor(source)
