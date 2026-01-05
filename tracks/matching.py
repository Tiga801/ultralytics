"""Matching algorithms for object tracking.

This module provides linear assignment, IoU distance, and embedding distance functions
for data association in multi-object tracking.
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from .utils import bbox_ioa

# Try to import lap for faster linear assignment
try:
    import lap

    HAVE_LAP = True
except ImportError:
    HAVE_LAP = False


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix: The matrix containing cost values for assignments, with shape (N, M).
        thresh: Threshold for considering an assignment valid.
        use_lap: Use lap.lapjv for the assignment. If False, scipy is used.

    Returns:
        Tuple of (matched_indices, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap and HAVE_LAP:
        # Use lap.lapjv
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks: List of tracks 'a' or bounding boxes (as numpy arrays in xyxy format).
        btracks: List of tracks 'b' or bounding boxes (as numpy arrays in xyxy format).

    Returns:
        Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # Extract xyxy from track objects
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_ioa(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32),
            iou=True,
        )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks: List of tracks, where each track contains embedding features.
        detections: List of detections, where each detection contains embedding features.
        metric: Metric for distance computation (e.g., 'cosine', 'euclidean').

    Returns:
        Cost matrix computed based on embeddings with shape (N, M).
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix: The matrix containing cost values for assignments, with shape (N, M).
        detections: List of detections, each containing a score attribute.

    Returns:
        Fused similarity matrix with shape (N, M).
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
