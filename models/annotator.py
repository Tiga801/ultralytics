"""
Annotator - Visualization utilities for drawing predictions on images.
No dependencies on ultralytics internals.
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


# COCO keypoint skeleton connections (0-indexed)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # arms and shoulders
    [6, 12], [7, 13],  # body
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],  # torso and legs
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # face
]

# Keypoint colors for COCO (17 keypoints)
KPT_COLORS = [
    (0, 255, 0),    # nose
    (0, 255, 0),    # left_eye
    (0, 255, 0),    # right_eye
    (0, 255, 0),    # left_ear
    (0, 255, 0),    # right_ear
    (255, 128, 0),  # left_shoulder
    (255, 128, 0),  # right_shoulder
    (255, 128, 0),  # left_elbow
    (255, 128, 0),  # right_elbow
    (255, 128, 0),  # left_wrist
    (255, 128, 0),  # right_wrist
    (0, 255, 255),  # left_hip
    (0, 255, 255),  # right_hip
    (0, 255, 255),  # left_knee
    (0, 255, 255),  # right_knee
    (0, 255, 255),  # left_ankle
    (0, 255, 255),  # right_ankle
]

# Limb colors for skeleton
LIMB_COLORS = [
    (0, 255, 255),  # left ankle to left knee
    (0, 255, 255),  # left knee to left hip
    (0, 255, 255),  # right ankle to right knee
    (0, 255, 255),  # right knee to right hip
    (0, 255, 255),  # left hip to right hip
    (255, 128, 0),  # left hip to left shoulder
    (255, 128, 0),  # right hip to right shoulder
    (255, 128, 0),  # left shoulder to right shoulder
    (255, 128, 0),  # left shoulder to left elbow
    (255, 128, 0),  # right shoulder to right elbow
    (255, 128, 0),  # left elbow to left wrist
    (255, 128, 0),  # right elbow to right wrist
    (0, 255, 0),    # left eye to right eye
    (0, 255, 0),    # nose to left eye
    (0, 255, 0),    # nose to right eye
    (0, 255, 0),    # left eye to left ear
    (0, 255, 0),    # right eye to right ear
    (0, 255, 0),    # left ear to left shoulder
    (0, 255, 0),    # right ear to right shoulder
]


def get_color(idx: int, bgr: bool = True) -> Tuple[int, int, int]:
    """
    Get a unique color for a given index.

    Args:
        idx: Index for color
        bgr: Return BGR format (default True)

    Returns:
        Color tuple (B, G, R) or (R, G, B)
    """
    # Color palette (20 distinct colors)
    palette = [
        (255, 56, 56),    # red
        (255, 157, 151),  # light coral
        (255, 112, 31),   # orange
        (255, 178, 29),   # gold
        (207, 210, 49),   # yellow-green
        (72, 249, 10),    # lime
        (146, 204, 23),   # yellow-green
        (61, 219, 134),   # spring green
        (26, 147, 52),    # forest green
        (0, 212, 187),    # cyan
        (44, 153, 168),   # teal
        (0, 194, 255),    # sky blue
        (52, 69, 147),    # navy
        (100, 115, 255),  # blue
        (0, 24, 236),     # royal blue
        (132, 56, 255),   # purple
        (82, 0, 133),     # dark purple
        (203, 56, 255),   # magenta
        (255, 149, 200),  # pink
        (255, 55, 199),   # hot pink
    ]

    color = palette[idx % len(palette)]
    return color if bgr else (color[2], color[1], color[0])


class Annotator:
    """
    Image annotator for drawing bounding boxes, labels, and keypoints.

    Args:
        im: Image to annotate (BGR format)
        line_width: Line width for drawing (auto-calculated if None)
        font_size: Font size for text (auto-calculated if None)
    """

    def __init__(
        self,
        im: np.ndarray,
        line_width: Optional[int] = None,
        font_size: Optional[float] = None,
    ):
        """Initialize annotator with image."""
        self.im = im.copy()
        self.lw = line_width or max(round(sum(im.shape[:2]) / 2 * 0.003), 2)
        self.sf = font_size or max(self.lw - 1, 1) / 3  # font scale
        self.tf = max(self.lw - 1, 1)  # font thickness

    def box_label(
        self,
        box: List[float],
        label: str = "",
        color: Tuple[int, int, int] = (128, 128, 128),
        txt_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Draw bounding box with optional label.

        Args:
            box: Box coordinates [x1, y1, x2, y2]
            label: Text label
            color: Box color (BGR)
            txt_color: Text color (BGR)
        """
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))

        # Draw rectangle
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)

        # Draw label
        if label:
            # Get text size
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
            h += 3  # Add padding

            # Check if label fits above box
            outside = p1[1] - h >= 3
            p2_label = (p1[0] + w, p1[1] - h if outside else p1[1] + h)

            # Draw label background
            cv2.rectangle(self.im, p1, p2_label, color, -1, cv2.LINE_AA)

            # Draw text
            text_pos = (p1[0], p1[1] - 2 if outside else p1[1] + h - 1)
            cv2.putText(
                self.im,
                label,
                text_pos,
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )

    def kpts(
        self,
        kpts: np.ndarray,
        shape: Tuple[int, int] = (640, 640),
        radius: int = 5,
        kpt_line: bool = True,
        conf_thres: float = 0.25,
    ):
        """
        Draw keypoints and skeleton connections.

        Args:
            kpts: Keypoints array (K, 2/3) with [x, y] or [x, y, conf]
            shape: Original image shape (H, W)
            radius: Keypoint circle radius
            kpt_line: Draw skeleton lines
            conf_thres: Confidence threshold for showing keypoints
        """
        if isinstance(kpts, np.ndarray):
            kpts = kpts.copy()
        else:
            kpts = kpts.cpu().numpy()

        nkpt = len(kpts)
        has_conf = kpts.shape[-1] == 3

        # Draw skeleton lines first (below keypoints)
        if kpt_line and nkpt == 17:  # COCO skeleton
            for i, (idx1, idx2) in enumerate(SKELETON):
                idx1 -= 1  # Convert to 0-indexed
                idx2 -= 1

                # Skip if out of bounds
                if idx1 >= nkpt or idx2 >= nkpt:
                    continue

                # Check confidence
                if has_conf:
                    if kpts[idx1, 2] < conf_thres or kpts[idx2, 2] < conf_thres:
                        continue

                # Get positions
                pos1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
                pos2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))

                # Skip invalid positions
                if pos1[0] < 0 or pos1[1] < 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue

                # Draw line
                color = LIMB_COLORS[i % len(LIMB_COLORS)]
                cv2.line(self.im, pos1, pos2, color, thickness=int(np.ceil(self.lw / 2)), lineType=cv2.LINE_AA)

        # Draw keypoints
        for i, kpt in enumerate(kpts):
            # Check confidence
            if has_conf and kpt[2] < conf_thres:
                continue

            x, y = int(kpt[0]), int(kpt[1])

            # Skip invalid positions
            if x < 0 or y < 0:
                continue

            # Get color
            color = KPT_COLORS[i % len(KPT_COLORS)]

            # Draw filled circle
            cv2.circle(self.im, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    def text(
        self,
        pos: Tuple[int, int],
        text: str,
        txt_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Optional[Tuple[int, int, int]] = None,
        anchor: str = "top-left",
    ):
        """
        Draw text on image.

        Args:
            pos: Position (x, y)
            text: Text to draw (supports newlines)
            txt_color: Text color (BGR)
            bg_color: Background color (BGR), None for no background
            anchor: Text anchor point ('top-left', 'bottom-left')
        """
        lines = text.split("\n")
        line_height = cv2.getTextSize("A", 0, fontScale=self.sf, thickness=self.tf)[0][1] + 10

        x, y = pos

        for i, line in enumerate(lines):
            line_y = y + i * line_height

            if bg_color is not None:
                # Draw background
                w, h = cv2.getTextSize(line, 0, fontScale=self.sf, thickness=self.tf)[0]
                cv2.rectangle(
                    self.im,
                    (x - 2, line_y - h - 3),
                    (x + w + 2, line_y + 3),
                    bg_color,
                    -1,
                )

            # Draw text
            cv2.putText(
                self.im,
                line,
                (x, line_y),
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )

    def get_color(self, idx: int) -> Tuple[int, int, int]:
        """Get color for a class index."""
        return get_color(idx)

    def result(self) -> np.ndarray:
        """Return annotated image."""
        return self.im


def plot_results(
    image: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    names: Optional[dict] = None,
    line_width: Optional[int] = None,
) -> np.ndarray:
    """
    Plot detection/pose/classification results on image.

    Args:
        image: Image (BGR)
        boxes: Bounding boxes (N, 6) with [x1, y1, x2, y2, conf, cls]
        keypoints: Keypoints (N, K, 3)
        probs: Classification probabilities
        names: Class names dictionary
        line_width: Line width

    Returns:
        Annotated image
    """
    names = names or {}
    annotator = Annotator(image, line_width=line_width)

    # Draw boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            cls = int(cls)
            cls_name = names.get(cls, str(cls))
            label = f"{cls_name} {conf:.2f}"
            annotator.box_label([x1, y1, x2, y2], label, color=get_color(cls))

    # Draw keypoints
    if keypoints is not None:
        for kpt in keypoints:
            annotator.kpts(kpt, image.shape[:2])

    # Draw classification
    if probs is not None:
        top5 = np.argsort(probs)[::-1][:5]
        text_lines = []
        for idx in top5:
            cls_name = names.get(idx, str(idx))
            text_lines.append(f"{cls_name}: {probs[idx]:.2%}")
        annotator.text((10, 30), "\n".join(text_lines), txt_color=(255, 255, 255))

    return annotator.result()
