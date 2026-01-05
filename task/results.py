"""Task result containers.

This module provides the TaskResult class for encapsulating task processing
results including detections, events, and visualization data.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TaskResult:
    """Container for task processing results.

    This class encapsulates all output from a single frame processing,
    including detections, events for alerting, and visualization data.

    Attributes:
        task_id: Identifier of the task that produced this result.
        timestamp: Unix timestamp when the result was produced.
        frame_id: Sequential frame number.
        detections: List of detection results.
        events: List of events for alerting (e.g., line crossing).
        plot_im: Annotated image for visualization.
        speed: Timing information for profiling.
        metadata: Additional task-specific data.
    """

    task_id: str
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    # Detection results
    detections: List[Dict[str, Any]] = field(default_factory=list)

    # Event data (for alerting)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Visualization
    plot_im: Optional[np.ndarray] = None

    # Performance metrics
    speed: Dict[str, float] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_events(self) -> bool:
        """Check if result contains any events.

        Returns:
            True if there are events.
        """
        return len(self.events) > 0

    def has_detections(self) -> bool:
        """Check if result contains any detections.

        Returns:
            True if there are detections.
        """
        return len(self.detections) > 0

    def add_detection(
        self,
        bbox: List[float],
        class_id: int,
        class_name: str,
        confidence: float,
        track_id: Optional[int] = None,
        **kwargs
    ) -> None:
        """Add a detection to the results.

        Args:
            bbox: Bounding box [x1, y1, x2, y2].
            class_id: Class index.
            class_name: Class name.
            confidence: Detection confidence.
            track_id: Optional tracking ID.
            **kwargs: Additional detection attributes.
        """
        detection = {
            "bbox": bbox,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "track_id": track_id,
            **kwargs
        }
        self.detections.append(detection)

    def add_event(
        self,
        event_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Add an event to the results.

        Args:
            event_type: Type of event (e.g., "line_crossing", "intrusion").
            description: Human-readable description.
            data: Additional event data.
            **kwargs: Additional event attributes.
        """
        event = {
            "event_type": event_type,
            "description": description,
            "timestamp": self.timestamp,
            "data": data or {},
            **kwargs
        }
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding image).

        Returns:
            Dictionary representation.
        """
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detections": self.detections,
            "events": self.events,
            "speed": self.speed,
            "metadata": self.metadata,
            "has_plot": self.plot_im is not None,
        }

    def to_mqtt_payload(self) -> Dict[str, Any]:
        """Convert to MQTT message payload.

        Returns:
            Dictionary suitable for MQTT publishing.
        """
        return {
            "taskId": self.task_id,
            "timestamp": int(self.timestamp * 1000),  # milliseconds
            "frameId": self.frame_id,
            "events": self.events,
            "detectionCount": len(self.detections),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TaskResult(task_id={self.task_id}, "
            f"detections={len(self.detections)}, "
            f"events={len(self.events)})"
        )


@dataclass
class BatchTaskResult:
    """Container for batch processing results.

    Used when processing multiple frames or aggregating results.
    """

    task_id: str
    results: List[TaskResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_result(self, result: TaskResult) -> None:
        """Add a result to the batch.

        Args:
            result: TaskResult to add.
        """
        self.results.append(result)

    def finalize(self) -> None:
        """Mark the batch as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get batch processing duration.

        Returns:
            Duration in seconds.
        """
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def total_events(self) -> int:
        """Get total number of events across all results.

        Returns:
            Total event count.
        """
        return sum(len(r.events) for r in self.results)

    @property
    def total_detections(self) -> int:
        """Get total number of detections across all results.

        Returns:
            Total detection count.
        """
        return sum(len(r.detections) for r in self.results)

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events from all results.

        Returns:
            List of all events.
        """
        events = []
        for result in self.results:
            events.extend(result.events)
        return events
