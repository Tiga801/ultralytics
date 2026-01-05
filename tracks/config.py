"""Configuration for object trackers.

This module provides configuration dataclass for ByteTrack and BoTSORT trackers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """Configuration for object trackers.

    This dataclass holds all configuration parameters for ByteTrack and BoTSORT trackers.

    Attributes:
        track_high_thresh: Threshold for high-confidence detections in first association stage.
        track_low_thresh: Threshold for low-confidence detections in second association stage.
        new_track_thresh: Minimum confidence score required to create a new track.
        track_buffer: Number of frames to keep lost tracks before removal.
        match_thresh: IoU threshold for matching tracks with detections.
        fuse_score: Whether to fuse detection scores with IoU for matching.
        gmc_method: Global Motion Compensation method ('sparseOptFlow', 'orb', 'sift', 'ecc', 'none').
        proximity_thresh: IoU threshold for spatial proximity in BoTSORT.
        appearance_thresh: Appearance similarity threshold for BoTSORT.
        with_reid: Whether to use ReID features for BoTSORT.
        encoder: External feature encoder callable for ReID (optional).
    """

    # Common parameters
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    # BoTSORT specific parameters
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False
    encoder: Any = None  # External feature encoder callable

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.track_high_thresh <= 1.0:
            raise ValueError(f"track_high_thresh must be in [0, 1], got {self.track_high_thresh}")
        if not 0.0 <= self.track_low_thresh <= 1.0:
            raise ValueError(f"track_low_thresh must be in [0, 1], got {self.track_low_thresh}")
        if not 0.0 <= self.new_track_thresh <= 1.0:
            raise ValueError(f"new_track_thresh must be in [0, 1], got {self.new_track_thresh}")
        if self.track_buffer < 0:
            raise ValueError(f"track_buffer must be non-negative, got {self.track_buffer}")
        if not 0.0 <= self.match_thresh <= 1.0:
            raise ValueError(f"match_thresh must be in [0, 1], got {self.match_thresh}")

    @classmethod
    def bytetrack_default(cls) -> TrackerConfig:
        """Return default configuration for ByteTrack."""
        return cls(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )

    @classmethod
    def botsort_default(cls) -> TrackerConfig:
        """Return default configuration for BoTSORT."""
        return cls(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )


# Simple namespace-like wrapper for compatibility
class SimpleNamespace:
    """Simple namespace for attribute access from dict-like objects."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({items})"


def config_to_namespace(config: TrackerConfig) -> SimpleNamespace:
    """Convert TrackerConfig to namespace for backward compatibility.

    Args:
        config: TrackerConfig instance.

    Returns:
        SimpleNamespace with same attributes.
    """
    return SimpleNamespace(
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
        fuse_score=config.fuse_score,
        gmc_method=config.gmc_method,
        proximity_thresh=config.proximity_thresh,
        appearance_thresh=config.appearance_thresh,
        with_reid=config.with_reid,
        model=config.encoder,
    )
