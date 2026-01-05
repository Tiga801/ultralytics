"""Standalone tracking module for multi-object tracking.

This module provides complete object tracking functionality independent of
ultralytics internals, including BYTETracker and BOTSORT algorithms.

Example usage:
    from tracks import BYTETracker, TrackerConfig, draw_tracks

    # Create tracker
    config = TrackerConfig(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        match_thresh=0.8,
    )
    tracker = BYTETracker(config)

    # Update with detections (numpy array format: [x, y, w, h, conf, cls])
    detections = np.array([
        [100, 200, 50, 80, 0.95, 0],
        [300, 150, 60, 90, 0.87, 1],
    ])
    tracks = tracker.update(detections, frame)

    # Visualize
    annotated = draw_tracks(frame, tracks)
"""

from .base import BaseTrack, TrackState
from .byte_tracker import BYTETracker
from .bot_sort import BOTSORT
from .config import TrackerConfig
from .gmc import GMC
from .kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH
from .strack import BOTrack, STrack
from .visualize import draw_tracks, draw_track_trail, update_track_history, get_color
from . import matching

__all__ = [
    # Trackers
    "BYTETracker",
    "BOTSORT",
    # Track classes
    "BaseTrack",
    "STrack",
    "BOTrack",
    "TrackState",
    # Configuration
    "TrackerConfig",
    # Kalman filters
    "KalmanFilterXYAH",
    "KalmanFilterXYWH",
    # Motion compensation
    "GMC",
    # Visualization
    "draw_tracks",
    "draw_track_trail",
    "update_track_history",
    "get_color",
    # Matching module
    "matching",
]
