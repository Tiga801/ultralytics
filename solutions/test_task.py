"""Cross line detection task.

This module provides the CrossLineDetectionTask class that detects
objects crossing a defined line in the video stream.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

from task import TaskBase, TaskRegistry, TaskResult
from tracks import BYTETracker, TrackerConfig


@TaskRegistry.register("test_task")
class TestTask(TaskBase):
    """Task for detecting objects crossing a defined line.

    This task monitors a video stream for objects (people, vehicles, etc.)
    crossing a user-defined line, useful for counting entries/exits,
    boundary violations, etc.

    Features:
        - Configurable detection line with direction
        - Object tracking to detect crossings
        - Support for multiple object classes
        - Directional counting (in/out)
    """

    def requires_stream(self) -> bool:
        """This task requires video stream input."""
        return True

    def _init_in_process(self) -> None:
        """Initialize resources in the subprocess.

        Loads the detection model and initializes tracking state.
        """
        from models import Predictor

        self.log("Initializing CrossLineDetectionTask...")

        # Load model
        model_path = self.task_config.get_extra("model_path", "weights/yolo11n.pt")
        device = self.task_config.get_extra("device", "cuda")
        conf = self.task_config.get_extra("confidence", 0.5)
        iou = self.task_config.get_extra("iou", 0.45)

        self._predictor = Predictor(
            model_path=model_path,
            task="detect",
            device=device,
            conf=conf,
            iou=iou,
        )

        # Initialize tracker
        tracker_config = TrackerConfig.bytetrack_default()
        self.tracker = BYTETracker(tracker_config, frame_rate=30)  # Todo: fps

        # Parse line coordinates from config
        self._line_coords = self._parse_line_coords()

        # Track objects that have crossed
        self._crossed_ids: set = set()

        # Tracking state for each object: {track_id: last_position}
        self._track_positions: Dict[int, Tuple[float, float]] = {}

        # Counters
        self._count_in = 0
        self._count_out = 0

        # Target classes (default: person=0)
        self._target_classes = self.task_config.get_extra(
            "target_classes", [0]
        )

        self.log(f"CrossLineDetectionTask initialized with line: {self._line_coords}")

    def _cleanup_in_process(self) -> None:
        """Release resources."""
        self._predictor = None
        self._track_positions.clear()
        self._crossed_ids.clear()
        self.log("CrossLineDetectionTask cleaned up")

    def _parse_line_coords(self) -> List[Tuple[int, int]]:
        """Parse line coordinates from task config.

        Returns:
            List of (x, y) points defining the line, or empty list if
            no config provided (will use frame dimensions as default).
        """
        # Get from areasInfo in config
        areas_info = self.task_config.areas_info
        if areas_info and len(areas_info) > 0:
            first_area = areas_info[0]
            if "line" in first_area:
                line_points = first_area["line"]
                return [(p["x"], p["y"]) for p in line_points]
            elif "points" in first_area:
                points = first_area["points"]
                return [(p["x"], p["y"]) for p in points[:2]]

        # Return empty - default line will be set on first frame
        return []

    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a frame for line crossing detection.

        Args:
            frame: BGR image.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with crossing events and visualizations.
        """
        result = TaskResult(task_id=self.task_id)
        result.timestamp = timestamp

        # Set default line coordinates on first frame if not configured
        if not self._line_coords:
            height, width = frame.shape[:2]
            self._line_coords = [(0, height // 2), (width, height // 2)]
            self.log(f"Using default line across middle: {self._line_coords}")

        # Run detection
        predictions = self._predictor(frame)
        if not predictions or len(predictions) == 0:
            return result

        # Get boxes with tracking
        boxes = predictions[0].boxes
        if boxes is None or len(boxes) == 0:
            return result

        # Convert to numpy array [x, y, w, h, conf, cls]
        # Todo: move other target class
        cls_index = boxes.cls.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        # Stack into detection array
        detections = np.column_stack([xywh, conf, cls_index])
        
        # Update tracker
        cross_line_tracks = self.tracker.update(detections, frame)

        # Check for crossings
        events = []
        
        # Check line crossing
        for cross_line_track in cross_line_tracks:
            x1, y1, x2, y2 = map(int, cross_line_track[:4])
            track_id = int(cross_line_track[4])
            conf = float(cross_line_track[5])
            cls_id = int(cross_line_track[6]) if len(cross_line_track) > 6 else 0

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if track_id in self._track_positions:
                prev_pos = self._track_positions[track_id]
                crossing = self._check_line_crossing(prev_pos, (cx, cy))

                if crossing != 0 and track_id not in self._crossed_ids:
                    self._crossed_ids.add(track_id)
                    if crossing > 0:
                        self._count_in += 1
                        direction = "in"
                    else:
                        self._count_out += 1
                        direction = "out"

                    event = {
                        "type": "line_crossing",
                        "track_id": track_id,
                        "class_id": cls_id,
                        "direction": direction,
                        "position": (cx, cy),
                        "timestamp": timestamp,
                        "count_in": self._count_in,
                        "count_out": self._count_out,
                    }
                    events.append(event)
                    self.log(f"Line crossing: {direction}, total in={self._count_in}, out={self._count_out}")
            
            # Update tracking position
            self._track_positions[track_id] = (cx, cy)

        # Build result
        result.detections = detections
        result.events = events
        # result.add_metric("count_in", self._count_in)
        # result.add_metric("count_out", self._count_out)
        # result.add_metric("total_crossings", self._count_in + self._count_out)

        # Generate visualization
        # vis_frame = self._draw_visualization(frame, detections)
        # result.visualization = vis_frame

        return result

    def _check_line_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
    ) -> int:
        """Check if movement crosses the line.

        Args:
            prev_pos: Previous position (x, y).
            curr_pos: Current position (x, y).

        Returns:
            0 if no crossing, 1 for positive direction, -1 for negative.
        """
        if len(self._line_coords) < 2:
            return 0

        p1, p2 = self._line_coords[0], self._line_coords[1]

        # Compute cross product to determine which side of line
        def side(point):
            return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
                    (p2[1] - p1[1]) * (point[0] - p1[0]))

        prev_side = side(prev_pos)
        curr_side = side(curr_pos)

        # Check if they are on different sides
        if prev_side * curr_side < 0:
            return 1 if curr_side > 0 else -1

        return 0

    def _draw_visualization(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw visualization on frame.

        Args:
            frame: Original frame.
            detections: List of detection dictionaries.

        Returns:
            Annotated frame.
        """
        vis = frame.copy()

        # Draw detection line
        if len(self._line_coords) >= 2:
            p1 = tuple(map(int, self._line_coords[0]))
            p2 = tuple(map(int, self._line_coords[1]))
            cv2.line(vis, p1, p2, (0, 255, 255), 2)

        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            track_id = det.get("track_id", -1)

            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if track_id not in self._crossed_ids else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw counters
        counter_text = f"IN: {self._count_in} | OUT: {self._count_out}"
        cv2.putText(vis, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return vis
