"""Test script for the standalone tracking module.

This script demonstrates the usage of the tracks module with RTSP video stream
and YOLO object detection model.

Test Input Specifications:
- RTSP video stream: rtsp://192.168.2.71:8554/mystream3
- Object detection model path: weights/yolo11m.pt

Test Workflow:
1. Load the object detection model
2. Perform correct inference on the video stream
3. Invoke the tracking module using inference results
4. Visualize tracking results (draw bounding boxes and track_id)
5. Save visualization results for the first 1000 frames
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import tracking module
from tracks import BYTETracker, BOTSORT, TrackerConfig, draw_tracks

# Import YOLO for detection
from ultralytics import YOLO


def test_bytetracker_with_rtsp():
    """Test BYTETracker with RTSP stream and YOLO detection."""
    print("=" * 60)
    print("Testing BYTETracker with RTSP Stream")
    print("=" * 60)

    # Configuration
    rtsp_url = "rtsp://192.168.2.71:8554/mystream3"
    model_path = "weights/yolo11m.pt"
    output_dir = Path(project_root) / "results"
    output_video_path = output_dir / "tracking_bytetracker.mp4"
    max_frames = 1000

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully")

    # Initialize tracker
    config = TrackerConfig.bytetrack_default()
    tracker = BYTETracker(config, frame_rate=30)
    print(f"Tracker initialized: BYTETracker")

    # Open video stream
    print(f"Opening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Cannot open RTSP stream: {rtsp_url}")
        print("Trying to use a local video file or webcam for testing...")
        # Fallback to webcam or local file for testing
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open any video source")
            return False

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Class names (COCO dataset)
    class_names = model.names if hasattr(model, 'names') else None

    frame_count = 0
    total_detections = 0
    total_tracks = 0

    print(f"Processing up to {max_frames} frames...")
    print("-" * 60)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"End of stream at frame {frame_count}")
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Get detection results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Convert to numpy array [x, y, w, h, conf, cls]
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_index = boxes.cls.cpu().numpy()

            # Stack into detection array
            detections = np.column_stack([xywh, conf, cls_index])
            total_detections += len(detections)
        else:
            detections = np.empty((0, 6))

        # Update tracker
        tracks = tracker.update(detections, frame)
        total_tracks += len(tracks)

        # Draw tracks on frame
        annotated = draw_tracks(
            frame,
            tracks,
            class_names=class_names,
            show_conf=True,
            show_class=True,
            show_id=True,
        )

        # Add frame info
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame
        out.write(annotated)

        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Cleanup
    cap.release()
    out.release()

    print("-" * 60)
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracks: {total_tracks}")
    print(f"Average detections per frame: {total_detections / max(frame_count, 1):.2f}")
    print(f"Average tracks per frame: {total_tracks / max(frame_count, 1):.2f}")
    print(f"Output saved to: {output_video_path}")
    print("=" * 60)

    return True


def test_botsort_with_rtsp():
    """Test BOTSORT with RTSP stream and YOLO detection."""
    print("=" * 60)
    print("Testing BOTSORT with RTSP Stream")
    print("=" * 60)

    # Configuration
    rtsp_url = "rtsp://192.168.2.71:8554/mystream3"
    model_path = "weights/yolo11m.pt"
    output_dir = Path(project_root) / "results"
    output_video_path = output_dir / "tracking_botsort.mp4"
    max_frames = 1000

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Model loaded successfully")

    # Initialize tracker with GMC
    config = TrackerConfig.botsort_default()
    tracker = BOTSORT(config, frame_rate=30)
    print(f"Tracker initialized: BOTSORT with GMC method: {config.gmc_method}")

    # Open video stream
    print(f"Opening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Cannot open RTSP stream: {rtsp_url}")
        print("Trying to use a local video file or webcam for testing...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open any video source")
            return False

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Class names
    class_names = model.names if hasattr(model, 'names') else None

    frame_count = 0
    total_detections = 0
    total_tracks = 0

    print(f"Processing up to {max_frames} frames...")
    print("-" * 60)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"End of stream at frame {frame_count}")
            break

        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Get detection results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_index = boxes.cls.cpu().numpy()
            detections = np.column_stack([xywh, conf, cls_index])
            total_detections += len(detections)
        else:
            detections = np.empty((0, 6))

        # Update tracker (pass frame for GMC)
        tracks = tracker.update(detections, frame)
        total_tracks += len(tracks)

        # Draw tracks on frame
        annotated = draw_tracks(
            frame,
            tracks,
            class_names=class_names,
            show_conf=True,
            show_class=True,
            show_id=True,
        )

        # Add frame info
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame
        out.write(annotated)

        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Cleanup
    cap.release()
    out.release()

    print("-" * 60)
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracks: {total_tracks}")
    print(f"Average detections per frame: {total_detections / max(frame_count, 1):.2f}")
    print(f"Average tracks per frame: {total_tracks / max(frame_count, 1):.2f}")
    print(f"Output saved to: {output_video_path}")
    print("=" * 60)

    return True


def test_tracker_basic():
    """Basic unit test for tracker without video stream."""
    print("=" * 60)
    print("Running Basic Unit Tests")
    print("=" * 60)

    # Test TrackerConfig
    print("Testing TrackerConfig...")
    config = TrackerConfig()
    assert config.track_high_thresh == 0.25
    assert config.match_thresh == 0.8
    print("  TrackerConfig: OK")

    # Test BYTETracker initialization
    print("Testing BYTETracker initialization...")
    tracker = BYTETracker(config)
    assert tracker.frame_id == 0
    assert len(tracker.tracked_stracks) == 0
    print("  BYTETracker: OK")

    # Test BOTSORT initialization
    print("Testing BOTSORT initialization...")
    botsort_config = TrackerConfig.botsort_default()
    botsort_tracker = BOTSORT(botsort_config)
    assert botsort_tracker.frame_id == 0
    print("  BOTSORT: OK")

    # Test tracker update with synthetic data
    print("Testing tracker update with synthetic detections...")
    # Create synthetic detections [x, y, w, h, conf, cls]
    detections = np.array([
        [100, 100, 50, 80, 0.9, 0],
        [200, 150, 60, 90, 0.85, 1],
    ], dtype=np.float32)

    # Create a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Update tracker
    tracks = tracker.update(detections, dummy_frame)
    print(f"  First update: {len(tracks)} tracks")

    # Second update with slightly moved detections
    detections2 = np.array([
        [105, 105, 50, 80, 0.88, 0],
        [205, 155, 60, 90, 0.82, 1],
    ], dtype=np.float32)

    tracks = tracker.update(detections2, dummy_frame)
    print(f"  Second update: {len(tracks)} tracks")
    assert len(tracks) > 0, "Should have at least one track"
    print("  Tracker update: OK")

    # Test draw_tracks
    print("Testing draw_tracks...")
    annotated = draw_tracks(dummy_frame, tracks)
    assert annotated.shape == dummy_frame.shape
    print("  draw_tracks: OK")

    # Test tracker reset
    print("Testing tracker reset...")
    tracker.reset()
    assert tracker.frame_id == 0
    assert len(tracker.tracked_stracks) == 0
    print("  Tracker reset: OK")

    print("-" * 60)
    print("All basic tests passed!")
    print("=" * 60)

    return True


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test tracking module")
    parser.add_argument("--test", choices=["basic", "bytetracker", "botsort", "all"],
                        default="bytetracker", help="Which test to run")
    parser.add_argument("--rtsp", type=str, default="rtsp://192.168.2.71:8554/mystream3",
                        help="RTSP stream URL")
    parser.add_argument("--model", type=str, default="weights/yolo11m.pt",
                        help="Path to YOLO model")
    parser.add_argument("--frames", type=int, default=1000,
                        help="Maximum frames to process")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Tracking Module Test Suite")
    print("=" * 60 + "\n")

    # Run basic tests first
    if args.test in ["basic", "all"]:
        if not test_tracker_basic():
            print("Basic tests failed!")
            return 1

    # Run RTSP stream tests
    if args.test in ["bytetracker", "all"]:
        try:
            test_bytetracker_with_rtsp()
        except Exception as e:
            print(f"BYTETracker test failed with error: {e}")

    if args.test in ["botsort", "all"]:
        try:
            test_botsort_with_rtsp()
        except Exception as e:
            print(f"BOTSORT test failed with error: {e}")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
