"""BOTSORT implementation for multi-object tracking with ReID and GMC.

This module provides the BOTSORT class for tracking multiple objects in video frames
with optional ReID features and Global Motion Compensation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .byte_tracker import BYTETracker
from .config import TrackerConfig, config_to_namespace
from .gmc import GMC
from .kalman_filter import KalmanFilterXYWH
from .strack import BOTrack
from . import matching


class BOTSORT(BYTETracker):
    """An extended version of the BYTETracker class, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh: Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh: Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder: Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc: An instance of the GMC algorithm for data association.
        args: Parsed configuration containing tracking parameters.
    """

    def __init__(self, args: TrackerConfig | Any, frame_rate: int = 30):
        """Initialize BOTSORT object with ReID module and GMC algorithm.

        Args:
            args: TrackerConfig or namespace containing tracking parameters.
            frame_rate: Frame rate of the video being processed.
        """
        super().__init__(args, frame_rate)

        # Initialize GMC
        self.gmc = GMC(method=self.args.gmc_method)

        # ReID module setup
        self.proximity_thresh = self.args.proximity_thresh
        self.appearance_thresh = self.args.appearance_thresh

        # Handle encoder: can be a callable, True for auto, or None
        if hasattr(args, 'encoder') and args.encoder is not None:
            self.encoder = args.encoder
        elif isinstance(args, TrackerConfig) and args.encoder is not None:
            self.encoder = args.encoder
        elif self.args.with_reid:
            # Auto mode: features provided externally via feats parameter
            self.encoder = lambda feats, s: [f for f in feats] if feats is not None else None
        else:
            self.encoder = None

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        """Return an instance of KalmanFilterXYWH for predicting and updating object states."""
        return KalmanFilterXYWH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[BOTrack]:
        """Initialize object tracks using detection bounding boxes, scores, class labels, and optional ReID features.

        Args:
            results: Detection results object or numpy array.
            img: Optional image or feature vectors for ReID.

        Returns:
            List of BOTrack objects.
        """
        if hasattr(results, 'conf'):
            # Object with attributes
            if len(results) == 0:
                return []
            bboxes = np.asarray(results.xywh if hasattr(results, 'xywh') else results[:, :4])
            bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
            conf = np.asarray(results.conf)
            cls = np.asarray(results.cls)

            if self.args.with_reid and self.encoder is not None and img is not None:
                try:
                    features = self.encoder(img, bboxes)
                    return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, conf, cls, features)]
                except Exception:
                    pass
            return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, conf, cls)]

        elif isinstance(results, np.ndarray):
            # Numpy array format [x, y, w, h, conf, cls]
            if len(results) == 0:
                return []
            bboxes = np.concatenate([results[:, :4], np.arange(len(results)).reshape(-1, 1)], axis=-1)
            conf = results[:, 4]
            cls = results[:, 5] if results.shape[1] > 5 else np.zeros(len(results))

            if self.args.with_reid and self.encoder is not None and img is not None:
                try:
                    features = self.encoder(img, bboxes)
                    return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, conf, cls, features)]
                except Exception:
                    pass
            return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, conf, cls)]

        return []

    def get_dists(self, tracks: list[BOTrack], detections: list[BOTrack]) -> np.ndarray:
        """Calculate distances between tracks and detections using IoU and optionally ReID embeddings.

        Args:
            tracks: List of BOTrack objects.
            detections: List of BOTrack objects.

        Returns:
            Distance matrix.
        """
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            # Check if tracks and detections have features
            has_features = (
                len(tracks) > 0 and hasattr(tracks[0], 'smooth_feat') and tracks[0].smooth_feat is not None and
                len(detections) > 0 and hasattr(detections[0], 'curr_feat') and detections[0].curr_feat is not None
            )

            if has_features:
                emb_dists = matching.embedding_distance(tracks, detections) / 2.0
                emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
                emb_dists[dists_mask] = 1.0
                dists = np.minimum(dists, emb_dists)

        return dists

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        """Predict the mean and covariance of multiple object tracks using a shared Kalman filter."""
        BOTrack.multi_predict(tracks)

    def reset(self) -> None:
        """Reset the BOTSORT tracker to its initial state, clearing all tracked objects and internal states."""
        super().reset()
        self.gmc.reset_params()
