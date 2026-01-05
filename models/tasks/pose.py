"""
Pose Predictor - Pose estimation inference.
No dependencies on ultralytics internals.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..postprocess import non_max_suppression, scale_boxes, scale_coords
from ..results import Results
from .base import BasePredictor


class PosePredictor(BasePredictor):
    """
    Pose estimation predictor.

    Performs inference with YOLO pose models.
    Extends detection with keypoint extraction.

    Args:
        model_path: Path to model weights
        device: Device for inference
        conf: Confidence threshold
        iou: IoU threshold for NMS
        max_det: Maximum detections per image
        kpt_shape: Keypoint shape (num_keypoints, dims) - auto-detected from model
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        kpt_shape: Optional[Tuple[int, int]] = None,
        fp16: bool = False,
        fuse: bool = True,
    ):
        """Initialize pose predictor."""
        super().__init__(
            model_path=model_path,
            device=device,
            conf=conf,
            iou=iou,
            fp16=fp16,
            fuse=fuse,
        )
        self.max_det = max_det

        # Get keypoint shape from model or use default COCO (17 keypoints, 3 dims: x, y, conf)
        self.kpt_shape = kpt_shape or self.model.kpt_shape or (17, 3)

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
        Postprocess pose predictions.

        Args:
            preds: Raw model predictions
            img: Preprocessed image tensor
            orig_imgs: Original images
            ratio_pads: Letterbox ratio and padding info
            paths: Image paths
            conf: Confidence threshold
            iou: IoU threshold

        Returns:
            List of Results objects with boxes and keypoints
        """
        # Handle tuple output
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # Handle transposed format: (batch, features, predictions) -> (batch, predictions, features)
        # YOLO models output (batch, 4+nc+kpts, num_preds) which needs to be transposed
        if preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(1, 2)

        # Calculate number of classes (for pose, typically 1 class: person)
        nkpt = self.kpt_shape[0]
        ndim = self.kpt_shape[1]
        nc = preds.shape[2] - 4 - nkpt * ndim  # classes = total - box - keypoints

        # Apply NMS
        preds_nms = non_max_suppression(
            preds,
            conf_thres=conf,
            iou_thres=iou,
            max_det=kwargs.get("max_det", self.max_det),
            nc=max(nc, 1),  # At least 1 class
        )

        results = []
        for i, (pred, orig_img, ratio_pad, path) in enumerate(
            zip(preds_nms, orig_imgs, ratio_pads, paths)
        ):
            if len(pred) == 0:
                # No detections
                result = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.names,
                    boxes=torch.zeros((0, 6), device=pred.device),
                    keypoints=torch.zeros((0, nkpt, ndim), device=pred.device),
                )
            else:
                # Scale boxes
                pred_boxes = pred[:, :6].clone()
                pred_boxes[:, :4] = scale_boxes(
                    img.shape[2:],
                    pred_boxes[:, :4],
                    orig_img.shape[:2],
                    ratio_pad,
                )

                # Extract and reshape keypoints
                pred_kpts = pred[:, 6:].view(len(pred), nkpt, ndim).clone()

                # Scale keypoint coordinates
                pred_kpts = scale_coords(
                    img.shape[2:],
                    pred_kpts,
                    orig_img.shape[:2],
                    ratio_pad,
                )

                result = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.names,
                    boxes=pred_boxes,
                    keypoints=pred_kpts,
                )

            results.append(result)

        return results
