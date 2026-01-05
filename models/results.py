"""
Results - Container classes for prediction outputs.
No dependencies on ultralytics internals.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class BaseTensor:
    """
    Base class for tensor wrappers with device operations.

    Provides common methods for moving data between devices and converting to numpy.
    """

    def __init__(self, data: torch.Tensor, orig_shape: Tuple[int, int]):
        """
        Initialize BaseTensor.

        Args:
            data: Tensor data
            orig_shape: Original image shape (H, W)
        """
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> torch.Size:
        """Return tensor shape."""
        return self.data.shape

    def __len__(self) -> int:
        """Return number of elements."""
        return len(self.data)

    def __getitem__(self, idx):
        """Index into tensor."""
        return self.__class__(self.data[idx], self.orig_shape)

    def cpu(self) -> "BaseTensor":
        """Move to CPU."""
        return self.__class__(self.data.cpu(), self.orig_shape)

    def cuda(self) -> "BaseTensor":
        """Move to CUDA."""
        return self.__class__(self.data.cuda(), self.orig_shape)

    def to(self, device: Union[str, torch.device]) -> "BaseTensor":
        """Move to specified device."""
        return self.__class__(self.data.to(device), self.orig_shape)

    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.cpu().numpy()


class Boxes(BaseTensor):
    """
    Bounding boxes container.

    Stores detection boxes in xyxy format with confidence and class.
    Data format: (N, 6) where columns are [x1, y1, x2, y2, conf, cls]
    """

    @property
    def xyxy(self) -> torch.Tensor:
        """Boxes in xyxy format (x1, y1, x2, y2)."""
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor:
        """Confidence scores."""
        return self.data[:, 4]

    @property
    def cls(self) -> torch.Tensor:
        """Class indices."""
        return self.data[:, 5]

    @property
    def xywh(self) -> torch.Tensor:
        """Boxes in xywh format (center_x, center_y, width, height)."""
        xyxy = self.xyxy
        xywh = torch.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2  # center x
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2  # center y
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
        return xywh

    @property
    def xyxyn(self) -> torch.Tensor:
        """Boxes in xyxy format normalized to [0, 1]."""
        xyxy = self.xyxy.clone()
        xyxy[:, [0, 2]] /= self.orig_shape[1]  # x / width
        xyxy[:, [1, 3]] /= self.orig_shape[0]  # y / height
        return xyxy

    @property
    def xywhn(self) -> torch.Tensor:
        """Boxes in xywh format normalized to [0, 1]."""
        xywh = self.xywh.clone()
        xywh[:, [0, 2]] /= self.orig_shape[1]  # x / width
        xywh[:, [1, 3]] /= self.orig_shape[0]  # y / height
        return xywh


class Keypoints(BaseTensor):
    """
    Keypoints container for pose estimation.

    Data format: (N, K, 3) where K is number of keypoints and 3 is [x, y, conf]
    """

    @property
    def xy(self) -> torch.Tensor:
        """Keypoint coordinates (x, y)."""
        return self.data[..., :2]

    @property
    def xyn(self) -> torch.Tensor:
        """Keypoint coordinates normalized to [0, 1]."""
        xy = self.xy.clone()
        xy[..., 0] /= self.orig_shape[1]  # x / width
        xy[..., 1] /= self.orig_shape[0]  # y / height
        return xy

    @property
    def conf(self) -> Optional[torch.Tensor]:
        """Keypoint confidence scores (if available)."""
        if self.data.shape[-1] == 3:
            return self.data[..., 2]
        return None

    @property
    def has_visible(self) -> bool:
        """Check if visibility/confidence data is available."""
        return self.data.shape[-1] == 3


class Probs(BaseTensor):
    """
    Classification probabilities container.

    Data format: (num_classes,) probability vector
    """

    @property
    def top1(self) -> int:
        """Top-1 class index."""
        return int(self.data.argmax())

    @property
    def top5(self) -> List[int]:
        """Top-5 class indices."""
        return self.data.argsort(descending=True)[:5].tolist()

    @property
    def top1conf(self) -> torch.Tensor:
        """Top-1 class confidence."""
        return self.data[self.top1]

    @property
    def top5conf(self) -> torch.Tensor:
        """Top-5 class confidences."""
        return self.data[self.top5]


class Results:
    """
    Container for inference results.

    Stores original image, predictions, and provides visualization methods.

    Attributes:
        orig_img: Original image (H, W, C) in BGR format
        orig_shape: Original image shape (H, W)
        path: Path to source image
        names: Class names dictionary
        boxes: Detected bounding boxes (Boxes object)
        keypoints: Detected keypoints for pose (Keypoints object)
        probs: Classification probabilities (Probs object)
        speed: Inference timing dict
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str = "",
        names: Dict[int, str] = None,
        boxes: torch.Tensor = None,
        keypoints: torch.Tensor = None,
        probs: torch.Tensor = None,
    ):
        """
        Initialize Results.

        Args:
            orig_img: Original image (BGR)
            path: Image path
            names: Class names dictionary
            boxes: Detection boxes (N, 6)
            keypoints: Pose keypoints (N, K, 3)
            probs: Classification probabilities
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]  # (H, W)
        self.path = path
        self.names = names or {}
        self.speed = {"preprocess": 0, "inference": 0, "postprocess": 0}

        # Wrap tensors
        self._boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self._keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self._probs = Probs(probs, self.orig_shape) if probs is not None else None

    @property
    def boxes(self) -> Optional[Boxes]:
        """Detection boxes."""
        return self._boxes

    @property
    def keypoints(self) -> Optional[Keypoints]:
        """Pose keypoints."""
        return self._keypoints

    @property
    def probs(self) -> Optional[Probs]:
        """Classification probabilities."""
        return self._probs

    def cpu(self) -> "Results":
        """Move all tensors to CPU."""
        result = Results(
            self.orig_img,
            self.path,
            self.names,
        )
        if self._boxes is not None:
            result._boxes = self._boxes.cpu()
        if self._keypoints is not None:
            result._keypoints = self._keypoints.cpu()
        if self._probs is not None:
            result._probs = self._probs.cpu()
        result.speed = self.speed
        return result

    def cuda(self) -> "Results":
        """Move all tensors to CUDA."""
        result = Results(
            self.orig_img,
            self.path,
            self.names,
        )
        if self._boxes is not None:
            result._boxes = self._boxes.cuda()
        if self._keypoints is not None:
            result._keypoints = self._keypoints.cuda()
        if self._probs is not None:
            result._probs = self._probs.cuda()
        result.speed = self.speed
        return result

    def numpy(self) -> "Results":
        """Convert all tensors to numpy."""
        return self.cpu()

    def plot(
        self,
        conf: bool = True,
        line_width: Optional[int] = None,
        font_size: Optional[float] = None,
        labels: bool = True,
        boxes: bool = True,
        kpts: bool = True,
        probs: bool = True,
        img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Plot predictions on image.

        Args:
            conf: Show confidence scores
            line_width: Line width for boxes
            font_size: Font size for labels
            labels: Show labels
            boxes: Show bounding boxes
            kpts: Show keypoints
            probs: Show classification probabilities
            img: Alternative image to plot on

        Returns:
            Annotated image (BGR)
        """
        # Import here to avoid circular import
        from .annotator import Annotator

        # Use original or provided image
        img = self.orig_img.copy() if img is None else img.copy()

        # Create annotator
        annotator = Annotator(img, line_width=line_width, font_size=font_size)

        # Plot boxes
        if self._boxes is not None and boxes and len(self._boxes) > 0:
            for i, box in enumerate(self._boxes.data):
                x1, y1, x2, y2, conf_val, cls_idx = box.tolist()
                cls_idx = int(cls_idx)
                cls_name = self.names.get(cls_idx, str(cls_idx))

                # Build label
                if labels:
                    if conf:
                        label = f"{cls_name} {conf_val:.2f}"
                    else:
                        label = cls_name
                else:
                    label = ""

                # Draw box
                annotator.box_label(
                    [x1, y1, x2, y2],
                    label=label,
                    color=annotator.get_color(cls_idx),
                )

        # Plot keypoints
        if self._keypoints is not None and kpts and len(self._keypoints) > 0:
            for kpt in self._keypoints.data:
                annotator.kpts(kpt, self.orig_shape)

        # Plot classification probabilities
        if self._probs is not None and probs:
            # Get top 5 classes
            top5_idx = self._probs.top5
            text_lines = []
            for idx in top5_idx:
                cls_name = self.names.get(idx, str(idx))
                prob = self._probs.data[idx].item()
                text_lines.append(f"{cls_name}: {prob:.2%}")
            text = "\n".join(text_lines)
            annotator.text((10, 30), text, txt_color=(255, 255, 255))

        return annotator.result()

    def save(self, filename: str):
        """
        Save annotated image to file.

        Args:
            filename: Output file path
        """
        img = self.plot()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename, img)

    def __len__(self) -> int:
        """Return number of detections."""
        if self._boxes is not None:
            return len(self._boxes)
        if self._keypoints is not None:
            return len(self._keypoints)
        if self._probs is not None:
            return 1
        return 0

    def __repr__(self) -> str:
        """String representation."""
        s = f"Results(path={self.path}, "
        if self._boxes is not None:
            s += f"boxes={len(self._boxes)}, "
        if self._keypoints is not None:
            s += f"keypoints={len(self._keypoints)}, "
        if self._probs is not None:
            s += f"probs={self._probs.top1}, "
        s += f"shape={self.orig_shape})"
        return s

    def verbose(self) -> str:
        """Generate verbose string description of results."""
        if self._probs is not None:
            # Classification result
            cls_name = self.names.get(self._probs.top1, str(self._probs.top1))
            return f"{cls_name} ({self._probs.top1conf.item():.2%})"

        if self._boxes is None or len(self._boxes) == 0:
            return "No detections"

        # Count detections per class
        counts = {}
        for box in self._boxes.data:
            cls_idx = int(box[5].item())
            cls_name = self.names.get(cls_idx, str(cls_idx))
            counts[cls_name] = counts.get(cls_name, 0) + 1

        # Format string
        parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()]
        return ", ".join(parts)
