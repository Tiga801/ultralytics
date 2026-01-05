"""Utility functions for object tracking.

This module provides coordinate conversion utilities and IoU calculations
that replace ultralytics internal dependencies.
"""

from __future__ import annotations

import logging
import numpy as np

# Simple logger
LOGGER = logging.getLogger("tracks")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2) format.

    Args:
        x: Bounding boxes in xywh format, shape (N, 4) or (4,).

    Returns:
        Bounding boxes in xyxy format.
    """
    y = np.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw  # x1
    y[..., 1] = x[..., 1] - dh  # y1
    y[..., 2] = x[..., 0] + dw  # x2
    y[..., 3] = x[..., 1] + dh  # y2
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height) format.

    Args:
        x: Bounding boxes in xyxy format, shape (N, 4) or (4,).

    Returns:
        Bounding boxes in xywh format.
    """
    y = np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2ltwh(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (x_center, y_center, width, height) to (left, top, width, height) format.

    Args:
        x: Bounding boxes in xywh format, shape (N, 4) or (4,).

    Returns:
        Bounding boxes in ltwh (top-left-width-height) format.
    """
    y = np.asarray(x).copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # left
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top
    return y


def ltwh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (left, top, width, height) to (x1, y1, x2, y2) format.

    Args:
        x: Bounding boxes in ltwh format, shape (N, 4) or (4,).

    Returns:
        Bounding boxes in xyxy format.
    """
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]  # x1
    y[..., 1] = x[..., 1]  # y1
    y[..., 2] = x[..., 0] + x[..., 2]  # x2
    y[..., 3] = x[..., 1] + x[..., 3]  # y2
    return y


def ltwh2xywh(x: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from (left, top, width, height) to (x_center, y_center, width, height) format.

    Args:
        x: Bounding boxes in ltwh format, shape (N, 4) or (4,).

    Returns:
        Bounding boxes in xywh format.
    """
    y = np.asarray(x).copy()
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # x center
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # y center
    return y


def bbox_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate IoU between two sets of bounding boxes in xyxy format.

    Args:
        box1: First set of boxes, shape (N, 4) in xyxy format.
        box2: Second set of boxes, shape (M, 4) in xyxy format.
        eps: Small value to avoid division by zero.

    Returns:
        IoU matrix of shape (N, M).
    """
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    if box1.ndim == 1:
        box1 = box1.reshape(1, -1)
    if box2.ndim == 1:
        box2 = box2.reshape(1, -1)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area - inter_area

    return inter_area / (union_area + eps)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Calculate intersection over area (or IoU) between two sets of bounding boxes.

    Args:
        box1: First set of boxes, shape (N, 4) in xyxy format.
        box2: Second set of boxes, shape (M, 4) in xyxy format.
        iou: If True, calculate IoU instead of IoA.
        eps: Small value to avoid division by zero.

    Returns:
        IoA or IoU matrix of shape (N, M).
    """
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    if box1.ndim == 1:
        box1 = box1.reshape(1, -1)
    if box2.ndim == 1:
        box2 = box2.reshape(1, -1)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    return inter_area / (area + eps)
