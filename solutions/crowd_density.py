"""Crowd density estimation task.

This module provides the CrowdDensityTask class for estimating
crowd density and detecting overcrowding in video streams.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from task import TaskBase, TaskRegistry, TaskResult


@TaskRegistry.register("crowd_density")
class CrowdDensityTask(TaskBase):
    """Task for crowd density estimation and monitoring.

    This task monitors video streams to estimate crowd density,
    count people, and generate alerts when density exceeds thresholds.

    Features:
        - Real-time people counting
        - Density estimation per region
        - Overcrowding alerts
        - Heat map generation
    """

    def requires_stream(self) -> bool:
        """This task requires video stream input."""
        return True

    def _init_in_process(self) -> None:
        """Initialize resources in the subprocess."""
        from models import Predictor

        self.log("Initializing CrowdDensityTask...")

        # Load detection model
        model_path = self.task_config.get_extra("model_path", "weights/yolo11n.pt")
        device = self.task_config.get_extra("device", "cuda")
        conf = self.task_config.get_extra("confidence", 0.3)  # Lower conf for crowds
        iou = self.task_config.get_extra("iou", 0.5)

        self._predictor = Predictor(
            model_path=model_path,
            device=device,
            conf=conf,
            iou=iou,
        )

        # Parse monitoring regions
        self._regions = self._parse_regions()

        # Density thresholds
        self._density_thresholds = {
            "low": self.task_config.get_extra("threshold_low", 5),
            "medium": self.task_config.get_extra("threshold_medium", 15),
            "high": self.task_config.get_extra("threshold_high", 30),
        }

        # Alert threshold (people per region)
        self._alert_threshold = self.task_config.get_extra("alert_threshold", 20)

        # Statistics
        self._max_count = 0
        self._alert_count = 0

        # Heat map accumulator
        self._heat_map: Optional[np.ndarray] = None
        self._heat_decay = self.task_config.get_extra("heat_decay", 0.95)

        # Target class (person = 0)
        self._person_class = 0

        self.log(f"CrowdDensityTask initialized with {len(self._regions)} regions")

    def _cleanup_in_process(self) -> None:
        """Release resources."""
        self._predictor = None
        self._heat_map = None
        self.log("CrowdDensityTask cleaned up")

    def _parse_regions(self) -> List[Dict]:
        """Parse monitoring regions from config.

        Returns:
            List of region definitions.
        """
        regions = []
        areas_info = self.task_config.areas_info

        if areas_info:
            for i, area in enumerate(areas_info):
                region = {
                    "id": area.get("id", f"zone_{i}"),
                    "name": area.get("name", f"Zone {i}"),
                    "points": [],
                    "max_capacity": area.get("max_capacity", self._alert_threshold),
                }

                if "points" in area:
                    region["points"] = [(p["x"], p["y"]) for p in area["points"]]

                if len(region["points"]) >= 3:
                    regions.append(region)

        # Return empty if no config - default region will be set on first frame
        return regions

    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a frame for crowd density estimation.

        Args:
            frame: BGR image.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with crowd metrics and visualizations.
        """
        result = TaskResult(task_id=self.task_id)
        result.timestamp = timestamp

        # Set default full-frame region on first frame if not configured
        if not self._regions:
            height, width = frame.shape[:2]
            self._regions = [{
                "id": "full_frame",
                "name": "Full Frame",
                "points": [(0, 0), (width, 0), (width, height), (0, height)],
                "max_capacity": self._alert_threshold,
            }]
            self.log(f"Using default full-frame region: {width}x{height}")

        # Initialize heat map if needed
        if self._heat_map is None:
            self._heat_map = np.zeros(frame.shape[:2], dtype=np.float32)

        # Decay heat map
        self._heat_map *= self._heat_decay

        # Run detection
        predictions = self._predictor(frame)
        if not predictions or len(predictions) == 0:
            result.add_metric("total_count", 0)
            return result

        pred = predictions[0]
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            result.add_metric("total_count", 0)
            return result

        # Process detections
        detections = []
        region_counts: Dict[str, int] = {r["id"]: 0 for r in self._regions}
        events = []

        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id != self._person_class:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            detection = {
                "bbox": xyxy.tolist(),
                "center": (cx, cy),
                "confidence": conf,
            }

            # Update heat map
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            y2 = max(0, min(y2, frame.shape[0] - 1))
            if x2 > x1 and y2 > y1:
                self._heat_map[y1:y2, x1:x2] += 1.0

            # Count in regions
            for region in self._regions:
                if self._point_in_polygon((cx, cy), region["points"]):
                    region_counts[region["id"]] += 1
                    break

            detections.append(detection)

        total_count = len(detections)
        self._max_count = max(self._max_count, total_count)

        # Check for overcrowding alerts
        for region in self._regions:
            region_id = region["id"]
            count = region_counts[region_id]
            max_capacity = region.get("max_capacity", self._alert_threshold)

            if count >= max_capacity:
                self._alert_count += 1
                event = {
                    "type": "overcrowding_alert",
                    "region_id": region_id,
                    "region_name": region["name"],
                    "count": count,
                    "max_capacity": max_capacity,
                    "density_level": self._get_density_level(count),
                    "timestamp": timestamp,
                }
                events.append(event)
                self.log(f"Overcrowding alert in {region['name']}: {count}/{max_capacity}")

        # Build result
        result.detections = detections
        result.events = events
        result.add_metric("total_count", total_count)
        result.add_metric("max_count", self._max_count)
        result.add_metric("alert_count", self._alert_count)

        for region in self._regions:
            region_id = region["id"]
            result.add_metric(f"count_{region_id}", region_counts[region_id])
            result.add_metric(
                f"density_{region_id}",
                self._get_density_level(region_counts[region_id])
            )

        # Generate visualization
        vis_frame = self._draw_visualization(frame, detections, region_counts)
        result.visualization = vis_frame

        return result

    def _get_density_level(self, count: int) -> str:
        """Get density level string based on count.

        Args:
            count: Number of people.

        Returns:
            Density level string.
        """
        if count >= self._density_thresholds["high"]:
            return "high"
        elif count >= self._density_thresholds["medium"]:
            return "medium"
        elif count >= self._density_thresholds["low"]:
            return "low"
        else:
            return "very_low"

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if point is inside polygon.

        Args:
            point: (x, y) coordinates.
            polygon: List of polygon vertices.

        Returns:
            True if inside.
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
        region_counts: Dict[str, int]
    ) -> np.ndarray:
        """Draw visualization with heat map overlay.

        Args:
            frame: Original frame.
            detections: People detections.
            region_counts: Count per region.

        Returns:
            Annotated frame with heat map.
        """
        vis = frame.copy()

        # Overlay heat map
        if self._heat_map is not None:
            # Normalize heat map
            heat_normalized = self._heat_map / (self._heat_map.max() + 1e-6)
            heat_normalized = np.clip(heat_normalized, 0, 1)

            # Apply colormap
            heat_colored = cv2.applyColorMap(
                (heat_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            # Blend with original
            alpha = 0.3
            vis = cv2.addWeighted(heat_colored, alpha, vis, 1 - alpha, 0)

        # Draw regions
        for region in self._regions:
            region_id = region["id"]
            points = np.array(region["points"], dtype=np.int32)
            count = region_counts.get(region_id, 0)
            max_cap = region.get("max_capacity", self._alert_threshold)

            # Color based on capacity usage
            usage = count / max_cap if max_cap > 0 else 0
            if usage >= 1.0:
                color = (0, 0, 255)  # Red - overcrowded
            elif usage >= 0.7:
                color = (0, 165, 255)  # Orange - warning
            else:
                color = (0, 255, 0)  # Green - normal

            cv2.polylines(vis, [points], True, color, 2)

            # Draw region label
            centroid = points.mean(axis=0).astype(int)
            label = f"{region['name']}: {count}/{max_cap}"
            cv2.putText(vis, label, tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw detections
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Draw summary
        total = len(detections)
        density = self._get_density_level(total)
        summary = f"Total: {total} | Density: {density} | Max: {self._max_count}"
        cv2.putText(vis, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return vis
