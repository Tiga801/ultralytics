"""
Classification Predictor - Image classification inference.
No dependencies on ultralytics internals.
"""

from typing import List, Tuple

import numpy as np
import torch

from ..preprocess import preprocess_classify
from ..results import Results
from .base import BasePredictor


class ClassificationPredictor(BasePredictor):
    """
    Image classification predictor.

    Performs inference with YOLO classification models.

    Args:
        model_path: Path to model weights
        device: Device for inference
        fp16: Use FP16 inference
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        fp16: bool = False,
        fuse: bool = True,
        **kwargs,
    ):
        """Initialize classification predictor."""
        # Classification doesn't use conf/iou thresholds
        super().__init__(
            model_path=model_path,
            device=device,
            conf=0.0,  # Not used
            iou=0.0,   # Not used
            fp16=fp16,
            fuse=fuse,
        )

        # Default classification image size
        if self.imgsz == (640, 640):
            self.imgsz = (224, 224)

    def preprocess(
        self, images: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[None]]:
        """
        Preprocess images for classification.

        Uses center crop and ImageNet normalization instead of letterbox.

        Args:
            images: List of images (BGR)

        Returns:
            Tuple of (batch_tensor, original_images, None)
        """
        im, orig_imgs = preprocess_classify(
            images,
            imgsz=self.imgsz,
            device=self.model.device,
            fp16=self.fp16,
        )
        # Return None for ratio_pads (not used in classification)
        return im, orig_imgs, [None] * len(images)

    def postprocess(
        self,
        preds: torch.Tensor,
        img: torch.Tensor,
        orig_imgs: List[np.ndarray],
        ratio_pads: List[None],
        paths: List[str],
        **kwargs,
    ) -> List[Results]:
        """
        Postprocess classification predictions.

        YOLO classification models output softmax probabilities directly,
        so no additional softmax is applied.

        Args:
            preds: Model predictions (probabilities from softmax)
            img: Preprocessed image tensor
            orig_imgs: Original images
            ratio_pads: Not used (None)
            paths: Image paths

        Returns:
            List of Results objects
        """
        # Handle tuple output
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # YOLO classification models already output probabilities (after softmax)
        # No additional softmax needed - just use predictions directly
        probs = preds

        results = []
        for i, (prob, orig_img, path) in enumerate(zip(probs, orig_imgs, paths)):
            result = Results(
                orig_img=orig_img,
                path=path,
                names=self.names,
                probs=prob,
            )
            results.append(result)

        return results
