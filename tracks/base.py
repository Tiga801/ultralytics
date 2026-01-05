"""Base classes for object tracking.

This module defines the base classes and structures for object tracking.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np


class TrackState:
    """Enumeration class representing the possible states of an object being tracked.

    Attributes:
        New: State when the object is newly detected.
        Tracked: State when the object is successfully tracked in subsequent frames.
        Lost: State when the object is no longer tracked.
        Removed: State when the object is removed from tracking.
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking, providing foundational attributes and methods.

    Attributes:
        _count: Class-level counter for unique track IDs.
        track_id: Unique identifier for the track.
        is_activated: Flag indicating whether the track is currently active.
        state: Current state of the track.
        history: Ordered history of the track's states.
        features: List of features extracted from the object for tracking.
        curr_feature: The current feature of the object being tracked.
        score: The confidence score of the tracking.
        start_frame: The frame number where tracking started.
        frame_id: The most recent frame ID processed by the track.
        time_since_update: Frames passed since the last update.
        location: The location of the object in the context of multi-camera tracking.
    """

    _count = 0

    def __init__(self):
        """Initialize a new track with a unique ID and foundational tracking attributes."""
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        """Return the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """Increment and return the next unique global track ID for object tracking."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        """Activate the track with provided arguments, initializing necessary attributes for tracking."""
        raise NotImplementedError

    def predict(self) -> None:
        """Predict the next state of the track based on the current state and tracking model."""
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the track with new observations and data, modifying its state and attributes accordingly."""
        raise NotImplementedError

    def mark_lost(self) -> None:
        """Mark the track as lost by updating its state to TrackState.Lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark the track as removed by setting its state to TrackState.Removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        """Reset the global track ID counter to its initial value."""
        BaseTrack._count = 0
