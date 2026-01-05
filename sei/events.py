# -*- coding: utf-8 -*-
"""
Event System Module

This module provides event definitions and trigger mechanisms for the
SEI video streaming pipeline. Events trigger recording operations and
are embedded in SEI data for downstream analysis.

Features:
- Event dataclass with metadata
- Configurable event triggers from inference results
- Event factory methods
- Event type definitions

Supported Event Types:
- cross_line: Object crossed a defined line
- intrusion: Object entered a restricted region
- detection: Specific object class detected
- anomaly: Unusual pattern detected
- custom: User-defined events

Usage:
    >>> from sei.events import SeiEvent, EventTrigger
    >>> trigger = EventTrigger({"type": "detection", "classes": ["person"]})
    >>> event = trigger.evaluate(inference_data, task_id="task001")
    >>> if event:
    ...     recorder.start_recording(event.event_id, event.event_type)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable


# Event type constants
EVENT_TYPE_CROSS_LINE = "cross_line"
EVENT_TYPE_INTRUSION = "intrusion"
EVENT_TYPE_DETECTION = "detection"
EVENT_TYPE_ANOMALY = "anomaly"
EVENT_TYPE_CUSTOM = "custom"

# All supported event types
SUPPORTED_EVENT_TYPES = [
    EVENT_TYPE_CROSS_LINE,
    EVENT_TYPE_INTRUSION,
    EVENT_TYPE_DETECTION,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_CUSTOM
]


@dataclass
class SeiEvent:
    """
    SEI Event Representation.

    Represents an event that can trigger recording and be embedded
    in SEI data for downstream analysis.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event (cross_line, intrusion, etc.)
        task_id: Associated task identifier
        timestamp: Event occurrence timestamp
        metadata: Additional event-specific data
        duration: Event duration (if known)
        confidence: Event confidence score (0-1)
        source: Event source (e.g., "inference", "manual")

    Example:
        >>> event = SeiEvent(
        ...     event_id="evt_001",
        ...     event_type="cross_line",
        ...     task_id="task001",
        ...     timestamp=time.time(),
        ...     metadata={"line_id": "line_1", "direction": "in"}
        ... )
    """
    event_id: str
    event_type: str
    task_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    confidence: float = 1.0
    source: str = "inference"

    @classmethod
    def create(
        cls,
        event_type: str,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = "inference"
    ) -> "SeiEvent":
        """
        Factory method to create event with auto-generated ID.

        Args:
            event_type: Type of event
            task_id: Associated task ID
            metadata: Event metadata
            confidence: Confidence score
            source: Event source

        Returns:
            New SeiEvent instance
        """
        return cls(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            task_id=task_id,
            timestamp=time.time(),
            metadata=metadata or {},
            confidence=confidence,
            source=source
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeiEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", f"evt_{uuid.uuid4().hex[:12]}"),
            event_type=data.get("event_type", EVENT_TYPE_CUSTOM),
            task_id=data.get("task_id", "unknown"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            duration=data.get("duration"),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "dict")
        )

    @classmethod
    def from_inference_result(
        cls,
        result: Dict[str, Any],
        task_id: str,
        event_type: str = EVENT_TYPE_DETECTION
    ) -> Optional["SeiEvent"]:
        """
        Create event from inference result.

        Extracts event information from model inference output.

        Args:
            result: Inference result dictionary
            task_id: Task identifier
            event_type: Event type to create

        Returns:
            SeiEvent if result indicates event, None otherwise
        """
        # Check for events in result
        events = result.get("events", [])
        if events:
            # Use first event
            event_data = events[0]
            return cls(
                event_id=event_data.get("event_id", f"evt_{uuid.uuid4().hex[:12]}"),
                event_type=event_data.get("type", event_type),
                task_id=task_id,
                timestamp=event_data.get("timestamp", time.time()),
                metadata=event_data,
                confidence=event_data.get("confidence", 1.0),
                source="inference"
            )

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "duration": self.duration,
            "confidence": self.confidence,
            "source": self.source
        }

    def to_sei_dict(self) -> Dict[str, Any]:
        """
        Convert to compact dictionary for SEI embedding.

        Uses abbreviated keys to minimize payload size.
        """
        data = {
            "eid": self.event_id,
            "et": self.event_type,
            "ts": self.timestamp,
            "conf": self.confidence
        }
        if self.metadata:
            data["meta"] = self.metadata
        return data


@dataclass
class TriggerCondition:
    """
    Event Trigger Condition.

    Defines a single condition for event triggering.

    Attributes:
        field: Field path in inference data to check
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, contains, exists)
        value: Value to compare against
    """
    field: str
    operator: str
    value: Any

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """
        Evaluate condition against data.

        Args:
            data: Data to check

        Returns:
            True if condition is met
        """
        # Navigate to field (supports dot notation)
        parts = self.field.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                current = None

            if current is None:
                break

        # Apply operator
        if self.operator == "exists":
            return current is not None

        if current is None:
            return False

        if self.operator == "eq":
            return current == self.value
        elif self.operator == "ne":
            return current != self.value
        elif self.operator == "gt":
            return current > self.value
        elif self.operator == "lt":
            return current < self.value
        elif self.operator == "gte":
            return current >= self.value
        elif self.operator == "lte":
            return current <= self.value
        elif self.operator == "contains":
            return self.value in current if hasattr(current, '__contains__') else False
        elif self.operator == "in":
            return current in self.value if hasattr(self.value, '__contains__') else False

        return False


class EventTrigger:
    """
    Event Trigger Evaluator.

    Evaluates inference results against configured conditions to
    determine if events should be triggered.

    Supported Trigger Types:
    - detection: Trigger on specific class detections
    - count: Trigger when detection count exceeds threshold
    - cross_line: Trigger on line crossing events
    - intrusion: Trigger on region intrusion events
    - custom: Custom condition-based triggers

    Example:
        >>> config = {
        ...     "type": "detection",
        ...     "classes": ["person", "car"],
        ...     "min_confidence": 0.7
        ... }
        >>> trigger = EventTrigger(config)
        >>> event = trigger.evaluate(inference_data, "task001")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_func: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize event trigger.

        Args:
            config: Trigger configuration
            log_func: Logging function
        """
        self._config = config or {}
        self._log = log_func or (lambda x: None)

        # Parse configuration
        self._trigger_type = self._config.get("type", "detection")
        self._enabled = self._config.get("enabled", True)
        self._cooldown = self._config.get("cooldown", 0.0)  # seconds between events

        # Type-specific configuration
        self._classes = self._config.get("classes", [])
        self._min_confidence = self._config.get("min_confidence", 0.5)
        self._min_count = self._config.get("min_count", 1)
        self._conditions = self._parse_conditions(self._config.get("conditions", []))

        # State
        self._last_trigger_time = 0.0
        self._trigger_count = 0

    def evaluate(
        self,
        inference_data: Dict[str, Any],
        task_id: str
    ) -> Optional[SeiEvent]:
        """
        Evaluate inference data for event trigger.

        Args:
            inference_data: Model inference results
            task_id: Task identifier

        Returns:
            SeiEvent if triggered, None otherwise
        """
        if not self._enabled:
            return None

        # Check cooldown
        current_time = time.time()
        if current_time - self._last_trigger_time < self._cooldown:
            return None

        # Check for pre-existing events in data
        if "events" in inference_data and inference_data["events"]:
            event = self._create_event_from_data(
                inference_data["events"][0],
                task_id
            )
            if event:
                self._last_trigger_time = current_time
                self._trigger_count += 1
            return event

        # Evaluate based on trigger type
        triggered = False
        event_metadata = {}

        if self._trigger_type == "detection":
            triggered, event_metadata = self._evaluate_detection(inference_data)
        elif self._trigger_type == "count":
            triggered, event_metadata = self._evaluate_count(inference_data)
        elif self._trigger_type == "cross_line":
            triggered, event_metadata = self._evaluate_cross_line(inference_data)
        elif self._trigger_type == "intrusion":
            triggered, event_metadata = self._evaluate_intrusion(inference_data)
        elif self._trigger_type == "custom":
            triggered, event_metadata = self._evaluate_custom(inference_data)

        if triggered:
            self._last_trigger_time = current_time
            self._trigger_count += 1
            return SeiEvent.create(
                event_type=self._trigger_type,
                task_id=task_id,
                metadata=event_metadata,
                source="trigger"
            )

        return None

    def _evaluate_detection(
        self,
        data: Dict[str, Any]
    ) -> tuple:
        """Evaluate detection-based trigger."""
        detections = data.get("detections", data.get("tracks", []))

        if not detections:
            return False, {}

        # Check for target classes
        matched = []
        for det in detections:
            cls_name = det.get("class", det.get("cls", det.get("name", "")))
            confidence = det.get("confidence", det.get("conf", 1.0))

            if confidence < self._min_confidence:
                continue

            if not self._classes or cls_name in self._classes:
                matched.append({
                    "class": cls_name,
                    "confidence": confidence,
                    "bbox": det.get("bbox", det.get("box", []))
                })

        if len(matched) >= self._min_count:
            return True, {"matched_detections": matched[:5]}  # Limit metadata size

        return False, {}

    def _evaluate_count(
        self,
        data: Dict[str, Any]
    ) -> tuple:
        """Evaluate count-based trigger."""
        detections = data.get("detections", data.get("tracks", []))

        count = 0
        for det in detections:
            cls_name = det.get("class", det.get("cls", det.get("name", "")))
            confidence = det.get("confidence", det.get("conf", 1.0))

            if confidence < self._min_confidence:
                continue

            if not self._classes or cls_name in self._classes:
                count += 1

        threshold = self._config.get("threshold", self._min_count)
        if count >= threshold:
            return True, {"count": count, "threshold": threshold}

        return False, {}

    def _evaluate_cross_line(
        self,
        data: Dict[str, Any]
    ) -> tuple:
        """Evaluate cross-line trigger."""
        # Check for explicit cross-line events
        cross_events = data.get("cross_line_events", [])
        if cross_events:
            return True, {"crosses": cross_events[:5]}

        # Check tracks for line crossing
        tracks = data.get("tracks", [])
        for track in tracks:
            if track.get("crossed_line"):
                return True, {
                    "track_id": track.get("track_id"),
                    "line_id": track.get("line_id"),
                    "direction": track.get("direction")
                }

        return False, {}

    def _evaluate_intrusion(
        self,
        data: Dict[str, Any]
    ) -> tuple:
        """Evaluate intrusion trigger."""
        # Check for explicit intrusion events
        intrusion_events = data.get("intrusion_events", [])
        if intrusion_events:
            return True, {"intrusions": intrusion_events[:5]}

        # Check tracks for region intrusion
        tracks = data.get("tracks", [])
        for track in tracks:
            if track.get("in_region"):
                return True, {
                    "track_id": track.get("track_id"),
                    "region_id": track.get("region_id")
                }

        return False, {}

    def _evaluate_custom(
        self,
        data: Dict[str, Any]
    ) -> tuple:
        """Evaluate custom condition trigger."""
        if not self._conditions:
            return False, {}

        # Check all conditions (AND logic)
        for condition in self._conditions:
            if not condition.evaluate(data):
                return False, {}

        return True, {"trigger": "custom_conditions"}

    def _create_event_from_data(
        self,
        event_data: Dict[str, Any],
        task_id: str
    ) -> Optional[SeiEvent]:
        """Create event from existing event data."""
        return SeiEvent(
            event_id=event_data.get("event_id", f"evt_{uuid.uuid4().hex[:12]}"),
            event_type=event_data.get("type", event_data.get("event_type", "detection")),
            task_id=task_id,
            timestamp=event_data.get("timestamp", time.time()),
            metadata=event_data,
            confidence=event_data.get("confidence", 1.0),
            source="data"
        )

    def _parse_conditions(
        self,
        conditions: List[Dict[str, Any]]
    ) -> List[TriggerCondition]:
        """Parse condition configurations."""
        parsed = []
        for cond in conditions:
            parsed.append(TriggerCondition(
                field=cond.get("field", ""),
                operator=cond.get("operator", "exists"),
                value=cond.get("value")
            ))
        return parsed

    @property
    def trigger_count(self) -> int:
        """Get total trigger count."""
        return self._trigger_count

    @property
    def enabled(self) -> bool:
        """Check if trigger is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable trigger."""
        self._enabled = True

    def disable(self) -> None:
        """Disable trigger."""
        self._enabled = False

    def reset_cooldown(self) -> None:
        """Reset cooldown timer."""
        self._last_trigger_time = 0.0
