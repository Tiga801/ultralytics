"""Region intrusion detection task.

This module provides the RegionIntrusionTask class that detects
objects entering defined restricted areas.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from task import TaskBase, TaskRegistry, TaskResult


@TaskRegistry.register("region_intrusion")
class RegionIntrusionTask(TaskBase):
    """Task for detecting objects entering restricted regions.

    This task monitors video streams for objects (people, vehicles, etc.)
    entering user-defined restricted areas, useful for security monitoring,
    safety zone enforcement, etc.

    Features:
        - Support for polygon and rectangle regions
        - Multiple region monitoring
        - Per-region intrusion counting
        - Configurable dwell time alerts
    """

    def requires_stream(self) -> bool:
        """This task requires video stream input."""
        return True

    def _init_in_process(self) -> None:
        """Initialize resources in the subprocess."""
        from models import Predictor

        self.log("Initializing RegionIntrusionTask...")

        # Load model
        model_path = self.task_config.get_extra("model_path", "weights/yolo11n.pt")
        device = self.task_config.get_extra("device", "cuda")
        conf = self.task_config.get_extra("confidence", 0.5)
        iou = self.task_config.get_extra("iou", 0.45)

        self._predictor = Predictor(
            model_path=model_path,
            device=device,
            conf=conf,
            iou=iou,
        )

        # Parse regions from config
        self._regions = self._parse_regions()

        # Track intrusions: {region_id: set(track_ids)}
        self._region_intrusions: Dict[str, set] = {
            r["id"]: set() for r in self._regions
        }

        # Object dwell times: {track_id: {region_id: first_seen_timestamp}}
        self._dwell_times: Dict[int, Dict[str, float]] = {}

        # Total intrusion counts per region
        self._intrusion_counts: Dict[str, int] = {
            r["id"]: 0 for r in self._regions
        }

        # Target classes
        self._target_classes = self.task_config.get_extra(
            "target_classes", [0]  # Default: person only
        )

        # Dwell time threshold (seconds)
        self._dwell_threshold = self.task_config.get_extra(
            "dwell_threshold", 0.0  # Immediate alert by default
        )

        self.log(f"RegionIntrusionTask initialized with {len(self._regions)} regions")

    def _cleanup_in_process(self) -> None:
        """Release resources."""
        self._predictor = None
        self._region_intrusions.clear()
        self._dwell_times.clear()
        self.log("RegionIntrusionTask cleaned up")

    def _parse_regions(self) -> List[Dict]:
        """Parse region definitions from task config.

        Returns:
            List of region dictionaries with id, name, and polygon points.
        """
        regions = []
        areas_info = self.task_config.areas_info

        if areas_info:
            for i, area in enumerate(areas_info):
                region = {
                    "id": area.get("id", f"region_{i}"),
                    "name": area.get("name", f"Region {i}"),
                    "points": [],
                }

                if "points" in area:
                    points = area["points"]
                    region["points"] = [(p["x"], p["y"]) for p in points]
                elif "polygon" in area:
                    polygon = area["polygon"]
                    region["points"] = [(p["x"], p["y"]) for p in polygon]

                if len(region["points"]) >= 3:
                    regions.append(region)

        # Return empty if no config - default region will be set on first frame
        return regions

    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a frame for region intrusion detection.

        Args:
            frame: BGR image.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with intrusion events and visualizations.
        """
        result = TaskResult(task_id=self.task_id)
        result.timestamp = timestamp

        # Set default region on first frame if not configured
        if not self._regions:
            height, width = frame.shape[:2]
            self._regions = [{
                "id": "default",
                "name": "Default Region",
                "points": [
                    (width // 4, height // 4),
                    (3 * width // 4, height // 4),
                    (3 * width // 4, 3 * height // 4),
                    (width // 4, 3 * height // 4),
                ],
            }]
            self._region_intrusions = {"default": set()}
            self._intrusion_counts = {"default": 0}
            self.log(f"Using default region: {self._regions[0]['points']}")

        # Run detection
        predictions = self._predictor(frame)
        if not predictions or len(predictions) == 0:
            return result

        pred = predictions[0]
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            return result

        events = []
        detections = []
        current_intrusions: Dict[str, set] = {r["id"]: set() for r in self._regions}

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id not in self._target_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            track_id = int(box.id[0]) if box.id is not None else i

            detection = {
                "track_id": track_id,
                "class_id": cls_id,
                "bbox": xyxy.tolist(),
                "center": (cx, cy),
                "confidence": conf,
                "in_regions": [],
            }

            # Check each region
            for region in self._regions:
                region_id = region["id"]
                if self._point_in_polygon((cx, cy), region["points"]):
                    detection["in_regions"].append(region_id)
                    current_intrusions[region_id].add(track_id)

                    # Track dwell time
                    if track_id not in self._dwell_times:
                        self._dwell_times[track_id] = {}

                    if region_id not in self._dwell_times[track_id]:
                        self._dwell_times[track_id][region_id] = timestamp

                    dwell_time = timestamp - self._dwell_times[track_id][region_id]

                    # Check for new intrusion event
                    if (track_id not in self._region_intrusions[region_id] and
                            dwell_time >= self._dwell_threshold):
                        self._region_intrusions[region_id].add(track_id)
                        self._intrusion_counts[region_id] += 1

                        event = {
                            "type": "region_intrusion",
                            "region_id": region_id,
                            "region_name": region["name"],
                            "track_id": track_id,
                            "class_id": cls_id,
                            "position": (cx, cy),
                            "dwell_time": dwell_time,
                            "timestamp": timestamp,
                            "total_intrusions": self._intrusion_counts[region_id],
                        }
                        events.append(event)
                        self.log(f"Intrusion in {region['name']}: track_id={track_id}")

            detections.append(detection)

        # Clean up objects that left regions
        for region_id in self._region_intrusions:
            left_objects = self._region_intrusions[region_id] - current_intrusions[region_id]
            for track_id in left_objects:
                if track_id in self._dwell_times:
                    if region_id in self._dwell_times[track_id]:
                        del self._dwell_times[track_id][region_id]
            self._region_intrusions[region_id] = current_intrusions[region_id]

        # Build result
        result.detections = detections
        result.events = events

        for region in self._regions:
            region_id = region["id"]
            result.add_metric(
                f"intrusions_{region_id}",
                self._intrusion_counts[region_id]
            )
            result.add_metric(
                f"current_in_{region_id}",
                len(current_intrusions[region_id])
            )

        # Generate visualization
        vis_frame = self._draw_visualization(frame, detections, current_intrusions)
        result.visualization = vis_frame

        return result

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if a point is inside a polygon using ray casting.

        Args:
            point: (x, y) point to check.
            polygon: List of (x, y) vertices.

        Returns:
            True if point is inside polygon.
        """
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y) and
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _draw_visualization(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        current_intrusions: Dict[str, set]
    ) -> np.ndarray:
        """Draw visualization on frame.

        Args:
            frame: Original frame.
            detections: List of detection dictionaries.
            current_intrusions: Current objects in each region.

        Returns:
            Annotated frame.
        """
        vis = frame.copy()

        # Draw regions
        for region in self._regions:
            region_id = region["id"]
            points = np.array(region["points"], dtype=np.int32)

            # Color based on intrusion status
            has_intrusion = len(current_intrusions.get(region_id, set())) > 0
            color = (0, 0, 255) if has_intrusion else (0, 255, 0)

            # Draw filled polygon with transparency
            overlay = vis.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

            # Draw outline
            cv2.polylines(vis, [points], True, color, 2)

            # Draw region name and count
            centroid = points.mean(axis=0).astype(int)
            count = self._intrusion_counts.get(region_id, 0)
            label = f"{region['name']}: {count}"
            cv2.putText(vis, label, tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            track_id = det.get("track_id", -1)
            in_regions = det.get("in_regions", [])

            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if in_regions else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis
