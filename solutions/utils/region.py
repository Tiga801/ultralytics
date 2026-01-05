from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import cv2


def parse_detection_region(areas_info) -> Optional[List[Tuple[int, int]]]:
        """Parse detection region from config.

        Returns:
            List of polygon points or None for full frame.
        """
        if areas_info and len(areas_info) > 0:
            first_area = areas_info[0]
            if "points" in first_area:
                points = first_area["points"]
                return [(p["x"], p["y"]) for p in points]
        return None


def point_in_region(point: Tuple[float, float], region) -> bool:
        """Check if point is in detection region.

        Args:
            point: (x, y) coordinates.

        Returns:
            True if in region or no region defined.
        """
        if not region:
            return True

        x, y = point
        n = len(region)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = region[i]
            xj, yj = region[j]

            if ((yi > y) != (yj > y) and
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside


def apply_region_mask(frame: np.ndarray, region) -> np.ndarray:
        """Apply detection region mask to frame.

        Args:
            frame: Original frame.

        Returns:
            Masked frame.
        """
        if not region:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = np.array(region, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked