"""ROI (Region of Interest) utilities for video analytics tasks.

This module provides functions to parse ROI configurations from API requests
and filter detection targets based on ROI regions.

ROI Configuration Format (API input with normalized coordinates):
    - Rectangular: {"start": {"x": "0.1", "y": "0.1"}, "end": {"x": "0.9", "y": "0.9"}, "type": "rect"}
    - Polygonal: {"points": [{"x": "0.1", "y": "0.2"}, ...], "type": "polygon"}

Internal Format (pixel coordinates in areas_info):
    {"id": "roi_1", "type": "polygon", "points": [{"x": 100, "y": 200}, ...]}
"""

from typing import Any, Dict, List, Optional, Tuple


def parse_roi_from_api(
    config_param: Dict[str, Any],
    frame_width: int,
    frame_height: int
) -> List[Dict[str, Any]]:
    """Parse roi_1 through roi_5 from API config and convert to areas_info format.

    This function:
    - Extracts ROI fields (roi_1 to roi_5) from config
    - Converts normalized coordinates (0.0-1.0) to pixel coordinates
    - Converts rectangular ROI to 4-corner polygon format
    - Returns unified areas_info format for internal use

    Args:
        config_param: The configParam dict from API request.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.

    Returns:
        List of region dicts in areas_info format with pixel coordinates.
    """
    areas_info = []

    for i in range(1, 6):  # roi_1 through roi_5
        roi_key = f"roi_{i}"
        roi_data = config_param.get(roi_key)

        if not roi_data or not isinstance(roi_data, list) or len(roi_data) == 0:
            continue

        roi_item = roi_data[0]  # First item in the list
        roi_type = roi_item.get("type", "")

        region = None
        if roi_type == "rect":
            region = _convert_rect_to_polygon(roi_item, frame_width, frame_height, roi_key)
        elif roi_type == "polygon":
            region = _convert_polygon(roi_item, frame_width, frame_height, roi_key)

        if region:
            areas_info.append(region)

    return areas_info


def _validate_and_clamp_coord(value: Any, max_val: int) -> int:
    """Validate and clamp coordinate value.

    Args:
        value: Coordinate value (string or number, normalized 0.0-1.0).
        max_val: Maximum pixel value (frame dimension).

    Returns:
        Clamped pixel coordinate.
    """
    try:
        norm = float(value)
        norm = max(0.0, min(1.0, norm))  # Clamp to [0, 1]
        return int(norm * max_val)
    except (ValueError, TypeError):
        return 0


def _convert_rect_to_polygon(
    roi_item: Dict,
    width: int,
    height: int,
    roi_id: str
) -> Optional[Dict[str, Any]]:
    """Convert rectangular ROI to polygon format with pixel coordinates.

    Input format:
        {"start": {"x": "0.1", "y": "0.1"}, "end": {"x": "0.9", "y": "0.9"}, "type": "rect"}

    Output format:
        {"id": "roi_1", "type": "polygon", "original_type": "rect",
         "points": [{"x": 100, "y": 100}, ...]}

    Args:
        roi_item: ROI configuration dict.
        width: Frame width in pixels.
        height: Frame height in pixels.
        roi_id: Unique identifier for this ROI.

    Returns:
        Region dict in areas_info format, or None if invalid.
    """
    try:
        start = roi_item.get("start", {})
        end = roi_item.get("end", {})

        # Convert normalized coordinates to pixel coordinates
        x1 = _validate_and_clamp_coord(start.get("x", 0), width)
        y1 = _validate_and_clamp_coord(start.get("y", 0), height)
        x2 = _validate_and_clamp_coord(end.get("x", 1), width)
        y2 = _validate_and_clamp_coord(end.get("y", 1), height)

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Validate dimensions
        if x2 <= x1 or y2 <= y1:
            return None

        # Create 4-corner polygon (clockwise from top-left)
        return {
            "id": roi_id,
            "type": "polygon",
            "original_type": "rect",
            "points": [
                {"x": x1, "y": y1},  # Top-left
                {"x": x2, "y": y1},  # Top-right
                {"x": x2, "y": y2},  # Bottom-right
                {"x": x1, "y": y2},  # Bottom-left
            ]
        }
    except (ValueError, TypeError, KeyError):
        return None


def _convert_polygon(
    roi_item: Dict,
    width: int,
    height: int,
    roi_id: str
) -> Optional[Dict[str, Any]]:
    """Convert polygon ROI to areas_info format with pixel coordinates.

    Input format:
        {"points": [{"x": "0.1", "y": "0.2"}, ...], "type": "polygon"}

    Output format:
        {"id": "roi_2", "type": "polygon", "original_type": "polygon",
         "points": [{"x": 100, "y": 200}, ...]}

    Args:
        roi_item: ROI configuration dict.
        width: Frame width in pixels.
        height: Frame height in pixels.
        roi_id: Unique identifier for this ROI.

    Returns:
        Region dict in areas_info format, or None if invalid.
    """
    try:
        points = roi_item.get("points", [])
        if len(points) < 3:
            return None  # Need at least 3 points for a polygon

        converted_points = []
        for p in points:
            x = _validate_and_clamp_coord(p.get("x", 0), width)
            y = _validate_and_clamp_coord(p.get("y", 0), height)
            converted_points.append({"x": x, "y": y})

        return {
            "id": roi_id,
            "type": "polygon",
            "original_type": "polygon",
            "points": converted_points
        }
    except (ValueError, TypeError, KeyError):
        return None


class ROIFilter:
    """ROI filter for use during task processing.

    Initialize once in _init_in_process(), then call is_target_in_roi()
    for each detection to check if it should be processed.

    Example:
        >>> roi_filter = ROIFilter(areas_info)
        >>> for detection in detections:
        ...     if not roi_filter.is_target_in_roi(x1, y1, x2, y2):
        ...         continue  # Skip target outside ROI
        ...     # Process detection...
    """

    def __init__(self, areas_info: List[Dict[str, Any]]):
        """Initialize ROI filter from areas_info.

        Args:
            areas_info: List of region dicts with pixel coordinate points.
        """
        self._regions: List[List[Tuple[int, int]]] = []
        self._enabled = False

        if areas_info:
            for area in areas_info:
                points = area.get("points", [])
                if len(points) >= 3:
                    polygon = [(p["x"], p["y"]) for p in points]
                    self._regions.append(polygon)

            self._enabled = len(self._regions) > 0

    @property
    def enabled(self) -> bool:
        """Check if ROI filtering is active."""
        return self._enabled

    @property
    def region_count(self) -> int:
        """Get number of configured ROI regions."""
        return len(self._regions)

    def is_target_in_roi(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> bool:
        """Check if target's center point is inside any ROI region.

        Computes the center point of the bounding box and checks if it
        falls within any of the configured ROI polygons.

        Args:
            x1, y1, x2, y2: Bounding box coordinates (top-left, bottom-right).

        Returns:
            True if:
            - No ROI configured (full frame processing), or
            - Center point is inside at least one ROI region
            False if center point is outside all ROI regions.
        """
        if not self._enabled:
            return True  # No ROI = pass all targets

        # Compute center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Check if center is in ANY region
        return any(
            self._point_in_polygon(center_x, center_y, region)
            for region in self._regions
        )

    @staticmethod
    def _point_in_polygon(
        x: float,
        y: float,
        polygon: List[Tuple[int, int]]
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm.

        Args:
            x, y: Point coordinates to check.
            polygon: List of (x, y) vertex tuples.

        Returns:
            True if point is inside polygon.
        """
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
