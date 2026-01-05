from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import cv2


def face_detection_visualization(
        frame: np.ndarray,
        # detections: List[Dict],
        # region
    ) -> np.ndarray:
        """Draw visualization on frame.

        Args:
            frame: Original frame.
            detections: List of face detections.

        Returns:
            Annotated frame.
        """
        vis = frame.copy()

        # Draw detection region if specified
        # if region:
        #     points = np.array(region, dtype=np.int32)
        #     cv2.polylines(vis, [points], True, (255, 255, 0), 2)

        # # Draw faces with track IDs and scores
        # for det in detections:
        #     bbox = det["bbox"]
        #     track_id = det.get("face_id", -1)
        #     kpt_sum = det.get("kpt_sum", 0)
        #     face_score = det.get("face_score", 0)

        #     x1, y1, x2, y2 = map(int, bbox)
        #     cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #     # Draw label with track ID and scores
        #     label = f"ID:{track_id} k:{kpt_sum:.2f} f:{face_score:.2f}"
        #     cv2.putText(vis, label, (x1, y1 - 10),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis