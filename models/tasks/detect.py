"""
Detection Predictor - Object detection inference.
No dependencies on ultralytics internals.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..postprocess import non_max_suppression, scale_boxes
from ..results import Results
from .base import BasePredictor


class DetectionPredictor(BasePredictor):
    """
    Object detection predictor.

    Performs inference with YOLOv8/v11 detection models.

    Args:
        model_path: Path to model weights
        device: Device for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS
        max_det: Maximum detections per image
        classes: Filter by class indices
        agnostic_nms: Class-agnostic NMS
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        agnostic_nms: bool = False,
        fp16: bool = False,
        fuse: bool = True,
    ):
        """Initialize detection predictor."""
        super().__init__(
            model_path=model_path,
            device=device,
            conf=conf,
            iou=iou,
            fp16=fp16,
            fuse=fuse,
        )
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms

    def postprocess(
        self,
        preds: torch.Tensor,
        img: torch.Tensor,
        orig_imgs: List[np.ndarray],
        ratio_pads: List[Tuple[float, Tuple[float, float]]],
        paths: List[str],
        conf: float = 0.25,
        iou: float = 0.45,
        **kwargs,
    ) -> List[Results]:
        """
        Postprocess detection predictions.

        Args:
            preds: Raw model predictions
            img: Preprocessed image tensor
            orig_imgs: Original images
            ratio_pads: Letterbox ratio and padding info
            paths: Image paths
            conf: Confidence threshold
            iou: IoU threshold

        Returns:
            List of Results objects
        """
        # Handle tuple output (some models return multiple outputs)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # Apply NMS
        preds = non_max_suppression(
            preds,
            conf_thres=conf,
            iou_thres=iou,
            classes=kwargs.get("classes", self.classes),
            agnostic=kwargs.get("agnostic_nms", self.agnostic_nms),
            max_det=kwargs.get("max_det", self.max_det),
            nc=self.nc,
        )

        results = []
        for i, (pred, orig_img, ratio_pad, path) in enumerate(
            zip(preds, orig_imgs, ratio_pads, paths)
        ):
            # Scale boxes to original image size
            if len(pred) > 0:
                pred[:, :4] = scale_boxes(
                    img.shape[2:],  # (H, W) of preprocessed image
                    pred[:, :4],
                    orig_img.shape[:2],  # (H, W) of original image
                    ratio_pad,
                )

            # Create Results object
            result = Results(
                orig_img=orig_img,
                path=path,
                names=self.names,
                boxes=pred[:, :6] if len(pred) > 0 else torch.zeros((0, 6), device=pred.device),
            )
            results.append(result)

        return results
