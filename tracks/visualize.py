"""Visualization utilities for object tracking.

This module provides functions for drawing tracked objects on video frames.
"""

from __future__ import annotations

import cv2
import numpy as np


def get_color(track_id: int, palette: str = "default") -> tuple[int, int, int]:
    """Get a unique color for a given track ID.

    Args:
        track_id: The track ID to get color for.
        palette: Color palette to use ('default', 'vivid', 'pastel').

    Returns:
        BGR color tuple.
    """
    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = ((track_id * golden_ratio) % 1.0) * 180  # OpenCV uses 0-180 for hue

    if palette == "vivid":
        saturation = 255
        value = 255
    elif palette == "pastel":
        saturation = 100
        value = 255
    else:  # default
        saturation = 200
        value = 220

    # Create HSV color and convert to BGR
    hsv = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    class_names: dict | list | None = None,
    show_conf: bool = True,
    show_class: bool = True,
    show_id: bool = True,
    thickness: int = 2,
    font_scale: float = 0.6,
    color_by_id: bool = True,
) -> np.ndarray:
    """Draw tracked objects on a frame.

    Args:
        frame: Input image array (BGR format).
        tracks: Array of tracked objects with shape (N, 7+) containing
            [x1, y1, x2, y2, track_id, score, class, ...].
        class_names: Optional dictionary or list mapping class indices to names.
        show_conf: Whether to show confidence scores.
        show_class: Whether to show class names/indices.
        show_id: Whether to show track IDs.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for text labels.
        color_by_id: If True, color by track ID; otherwise, color by class.

    Returns:
        Annotated frame with tracked objects drawn.
    """
    annotated = frame.copy()

    if len(tracks) == 0:
        return annotated

    for track in tracks:
        # Parse track data
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        score = float(track[5])
        cls = int(track[6]) if len(track) > 6 else 0

        # Get color
        if color_by_id:
            color = get_color(track_id)
        else:
            color = get_color(cls)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track_id}")
        if show_class:
            if class_names is not None:
                if isinstance(class_names, dict):
                    cls_name = class_names.get(cls, str(cls))
                else:
                    cls_name = class_names[cls] if cls < len(class_names) else str(cls)
            else:
                cls_name = str(cls)
            label_parts.append(cls_name)
        if show_conf:
            label_parts.append(f"{score:.2f}")

        label = " ".join(label_parts)

        if label:
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1,  # Filled
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
                cv2.LINE_AA,
            )

    return annotated


def draw_track_trail(
    frame: np.ndarray,
    track_history: dict[int, list[tuple[int, int]]],
    max_length: int = 30,
    thickness: int = 2,
) -> np.ndarray:
    """Draw tracking trails for objects.

    Args:
        frame: Input image array (BGR format).
        track_history: Dictionary mapping track IDs to lists of center points.
        max_length: Maximum trail length to draw.
        thickness: Line thickness for trails.

    Returns:
        Annotated frame with tracking trails drawn.
    """
    annotated = frame.copy()

    for track_id, points in track_history.items():
        if len(points) < 2:
            continue

        color = get_color(track_id)
        points_to_draw = points[-max_length:]

        for i in range(1, len(points_to_draw)):
            # Fade effect: older points are more transparent
            alpha = i / len(points_to_draw)
            fade_color = tuple(int(c * alpha) for c in color)

            cv2.line(
                annotated,
                points_to_draw[i - 1],
                points_to_draw[i],
                fade_color,
                thickness,
                cv2.LINE_AA,
            )

    return annotated


def update_track_history(
    track_history: dict[int, list[tuple[int, int]]],
    tracks: np.ndarray,
    max_length: int = 30,
) -> dict[int, list[tuple[int, int]]]:
    """Update tracking history with new track positions.

    Args:
        track_history: Dictionary mapping track IDs to lists of center points.
        tracks: Array of tracked objects with shape (N, 7+).
        max_length: Maximum history length per track.

    Returns:
        Updated track history dictionary.
    """
    current_ids = set()

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        current_ids.add(track_id)

        # Calculate center point
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append(center)

        # Limit history length
        if len(track_history[track_id]) > max_length:
            track_history[track_id] = track_history[track_id][-max_length:]

    return track_history
