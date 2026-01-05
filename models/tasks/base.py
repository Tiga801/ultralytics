"""
Base Predictor - Abstract base class for task-specific predictors.
No dependencies on ultralytics internals.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..backend import ModelBackend
from ..preprocess import load_image, preprocess_images
from ..results import Results


class BasePredictor(ABC):
    """
    Abstract base class for inference predictors.

    Provides common functionality for model loading, preprocessing, and inference.
    Task-specific postprocessing is implemented in subclasses.

    Args:
        model_path: Path to model weights
        device: Device for inference ('cuda', 'cuda:0', 'cpu')
        conf: Confidence threshold
        iou: IoU threshold for NMS
        fp16: Use FP16 inference
        fuse: Fuse Conv+BN layers (PyTorch only)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        fp16: bool = False,
        fuse: bool = True,
    ):
        """Initialize predictor with model."""
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.fp16 = fp16

        # Load model
        self.model = ModelBackend(
            weights=model_path,
            device=device,
            fp16=fp16,
            fuse=fuse,
        )

        # Get model properties
        self.names = self.model.names
        self.stride = self.model.stride
        self.imgsz = self.model.imgsz
        self.nc = self.model.nc
        self.task = self.model.task

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
            source: Image path(s) or numpy array(s)
            conf: Confidence threshold (overrides init value)
            iou: IoU threshold (overrides init value)
            **kwargs: Additional arguments for postprocessing

        Returns:
            List of Results objects
        """
        # Override thresholds if provided
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        # Load images
        images, paths = self._load_source(source)

        # Preprocess
        t0 = time.time()
        im, orig_imgs, ratio_pads = self.preprocess(images)
        preprocess_time = time.time() - t0

        # Inference
        t1 = time.time()
        preds = self.inference(im)
        inference_time = time.time() - t1

        # Postprocess
        t2 = time.time()
        results = self.postprocess(preds, im, orig_imgs, ratio_pads, paths, conf=conf, iou=iou, **kwargs)
        postprocess_time = time.time() - t2

        # Add timing info
        for r in results:
            r.speed = {
                "preprocess": preprocess_time * 1000 / len(images),
                "inference": inference_time * 1000 / len(images),
                "postprocess": postprocess_time * 1000 / len(images),
            }

        return results

    def _load_source(
        self, source: Union[str, np.ndarray, List[str], List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load images from source.

        Args:
            source: Image path(s) or numpy array(s)

        Returns:
            Tuple of (images, paths)
        """
        # Normalize to list
        if isinstance(source, (str, Path)):
            source = [str(source)]
        elif isinstance(source, np.ndarray):
            source = [source]

        images = []
        paths = []

        for item in source:
            if isinstance(item, (str, Path)):
                img = load_image(str(item))
                images.append(img)
                paths.append(str(item))
            elif isinstance(item, np.ndarray):
                images.append(item)
                paths.append("")
            else:
                raise ValueError(f"Unsupported source type: {type(item)}")

        return images, paths

    def preprocess(
        self, images: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[float, Tuple[float, float]]]]:
        """
        Preprocess images for inference.

        Args:
            images: List of images (BGR)

        Returns:
            Tuple of (batch_tensor, original_images, ratio_pads)
        """
        return preprocess_images(
            images,
            imgsz=self.imgsz,
            device=self.model.device,
            fp16=self.fp16,
            stride=self.stride,
            auto=False,
        )

    def inference(self, im: torch.Tensor) -> torch.Tensor:
        """
        Run model inference.

        Args:
            im: Preprocessed batch tensor (B, C, H, W)

        Returns:
            Model predictions
        """
        return self.model(im)

    @abstractmethod
    def postprocess(
        self,
        preds: torch.Tensor,
        img: torch.Tensor,
        orig_imgs: List[np.ndarray],
        ratio_pads: List[Tuple[float, Tuple[float, float]]],
        paths: List[str],
        **kwargs,
    ) -> List[Results]:
        """
        Process model predictions.

        Args:
            preds: Raw model predictions
            img: Preprocessed image tensor
            orig_imgs: Original images
            ratio_pads: Letterbox ratio and padding info
            paths: Image paths
            **kwargs: Additional arguments

        Returns:
            List of Results objects
        """
        raise NotImplementedError
