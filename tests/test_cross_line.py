#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Line Detection End-to-End Test Script

This script performs end-to-end testing of the cross-line detection pipeline
with optional SEI streaming and video recording.

Usage:
    # Basic test with defaults
    python tests/test_cross_line.py

    # With SEI streaming and recording
    python tests/test_cross_line.py --enable-sei-streaming --enable-recording

    # Custom configuration
    python tests/test_cross_line.py -m weights/yolo11n.pt -r rtsp://camera/stream
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stream import StreamLoader
from models import Predictor
from tracks import BYTETracker, TrackerConfig


@dataclass
class CrossLineTestConfig:
    """Configuration for cross-line detection test."""

    model_path: str
    rtsp_url: str
    enable_sei_streaming: bool
    enable_recording: bool
    rtmp_url: str
    output_dir: str
    confidence: float
    iou: float
    device: str
    sei_frame_skip: int = 3  # Process every Nth frame for SEI streaming


class CrossLineTest:
    """End-to-end cross-line detection test runner."""

    def __init__(self, config: CrossLineTestConfig):
        """Initialize test with configuration.

        Args:
            config: Test configuration with model, stream, and output settings.
        """
        self.config = config

        # Components
        self.stream_loader: Optional[StreamLoader] = None
        self.predictor: Optional[Predictor] = None
        self.tracker: Optional[BYTETracker] = None
        self.sei_pipeline = None

        # State
        self.stream_width = 0
        self.stream_height = 0
        self.stream_fps = 0.0
        self.line_coords: List[Tuple[int, int]] = []
        self.track_positions: Dict[int, Tuple[float, float]] = {}
        self.crossed_ids = set()
        self.count_in = 0
        self.count_out = 0
        self.frame_count = 0
        self.event_count = 0
        self.dropped_frames = 0  # Track SEI queue overflows
        self.sei_frame_skip = config.sei_frame_skip  # Process every Nth frame for SEI
        self.sei_frame_counter = 0  # Counter for frame skipping
        self.start_time = 0.0
        self.last_stats_time = 0.0

    def setup(self) -> bool:
        """Initialize all components in sequence.

        Returns:
            True if successful, False if any component fails.
        """
        print("=" * 80)
        print("Cross-Line Detection Test")
        print("=" * 80)

        # Print configuration
        print("\nConfiguration:")
        print(f"  Model:              {self.config.model_path}")
        print(f"  RTSP URL:           {self.config.rtsp_url}")
        print(f"  SEI Streaming:      {'Enabled' if self.config.enable_sei_streaming else 'Disabled'}")
        print(f"  Recording:          {'Enabled' if self.config.enable_recording else 'Disabled'}")
        if self.config.enable_sei_streaming or self.config.enable_recording:
            print(f"  RTMP URL:           {self.config.rtmp_url}")
            print(f"  SEI Frame Skip:     {self.config.sei_frame_skip} (1/{self.config.sei_frame_skip} frames pushed)")
        print(f"  Confidence:         {self.config.confidence}")
        print(f"  IoU Threshold:      {self.config.iou}")
        print(f"  Device:             {self.config.device}")

        print("\nInitializing components...")

        # 1. Initialize stream
        if not self._init_stream():
            print("ERROR: Failed to initialize stream")
            return False

        # 2. Initialize model
        if not self._init_model():
            print("ERROR: Failed to initialize model")
            return False

        # 3. Initialize tracker
        if not self._init_tracker():
            print("ERROR: Failed to initialize tracker")
            return False

        # 4. Create cross-line
        self._create_cross_line()

        # 5. Initialize SEI pipeline (if enabled)
        if self.config.enable_sei_streaming or self.config.enable_recording:
            if not self._init_sei_pipeline():
                print("WARNING: Failed to initialize SEI pipeline")
                # Don't fail, just disable SEI features

        self.start_time = time.time()
        self.last_stats_time = self.start_time

        print("\nStarting processing loop... (Press Ctrl+C to stop)")
        print("=" * 80)

        return True

    def _init_stream(self) -> bool:
        """Initialize RTSP stream loader.

        Returns:
            True if successful, False otherwise.
        """
        try:
            print("  Initializing stream loader...", end=" ", flush=True)

            self.stream_loader = StreamLoader(
                sources=self.config.rtsp_url
            )

            # Try to get first frame to validate stream
            try:
                for _, imgs, _ in self.stream_loader:
                    frame = imgs[0]
                    self.stream_height, self.stream_width, _ = frame.shape

                    # Get FPS if available
                    self.stream_fps = self.stream_loader.fps[0] if hasattr(
                        self.stream_loader, 'fps'
                    ) and self.stream_loader.fps else 25.0

                    print(f"\r  [OK] Stream connected ({self.stream_width}x{self.stream_height} @ {self.stream_fps:.1f} fps)")
                    return True

                print("\r  ERROR: No frames from stream")
                return False

            except StopIteration:
                print("\r  ERROR: Stream returned no frames")
                return False

        except Exception as e:
            print(f"\r  ERROR: {e}")
            return False

    def _init_model(self) -> bool:
        """Initialize YOLO predictor.

        Returns:
            True if successful, False otherwise.
        """
        try:
            print("  Initializing model...", end=" ", flush=True)

            self.predictor = Predictor(
                model_path=self.config.model_path,
                task='detect',
                device=self.config.device,
                conf=self.config.confidence,
                iou=self.config.iou,
                fp16=False,
                fuse=True
            )

            # Get model info
            num_classes = len(self.predictor.names) if hasattr(
                self.predictor, 'names'
            ) else 80

            print(f"\r  [OK] Model loaded ({num_classes} classes, device: {self.config.device})")
            return True

        except Exception as e:
            print(f"\r  ERROR: {e}")
            return False

    def _init_tracker(self) -> bool:
        """Initialize BYTETracker.

        Returns:
            True if successful, False otherwise.
        """
        try:
            print("  Initializing tracker...", end=" ", flush=True)

            tracker_config = TrackerConfig.bytetrack_default()
            frame_rate = int(self.stream_fps) if self.stream_fps > 0 else 25
            self.tracker = BYTETracker(tracker_config, frame_rate=frame_rate)

            print(f"\r  [OK] Tracker initialized (fps: {frame_rate})")
            return True

        except Exception as e:
            print(f"\r  ERROR: {e}")
            return False

    def _create_cross_line(self):
        """Create default horizontal cross-line at mid-height."""
        # Horizontal line across middle of frame
        self.line_coords = [(0, self.stream_height // 2), (self.stream_width, self.stream_height // 2)]

        print(
            f"  [OK] Cross-line defined: ({self.line_coords[0][0]}, {self.line_coords[0][1]}) -> "
            f"({self.line_coords[1][0]}, {self.line_coords[1][1]})"
        )

    def _init_sei_pipeline(self) -> bool:
        """Initialize SEI streaming pipeline if enabled.

        Returns:
            True if successful, False otherwise.
        """
        try:
            print("  Initializing SEI pipeline...", end=" ", flush=True)

            from sei import SeiStreamingPipeline
            from sei.config import SeiConfig, StreamConfig, OutputConfig, RecorderConfig

            sei_config = SeiConfig(enable=True)

            # Calculate effective fps based on frame skipping
            # If skipping every Nth frame, effective fps = source_fps / N
            effective_fps = self.stream_fps / self.sei_frame_skip

            stream_config = StreamConfig(
                width=self.stream_width,
                height=self.stream_height,
                fps=effective_fps,
                buffer_size=100,
                ffmpeg_path="ffmpeg"
            )

            recorder_config = RecorderConfig(
                output_dir=self.config.output_dir,
                pre_event_frames=50,
                post_event_frames=50
            )

            output_config = OutputConfig(
                rtmp_enabled=self.config.enable_sei_streaming,
                recording_enabled=self.config.enable_recording,
                stream_config=stream_config,
                recorder_config=recorder_config
            )

            self.sei_pipeline = SeiStreamingPipeline(
                sei_config=sei_config,
                stream_config=stream_config,
                output_config=output_config
            )

            # Start pipeline
            rtmp_url = self.config.rtmp_url if self.config.enable_sei_streaming else None
            self.sei_pipeline.start(rtmp_url)

            print(f"\r  [OK] SEI pipeline started")
            return True

        except ImportError:
            print("\r  WARNING: SEI module not available, disabling SEI features")
            return False
        except Exception as e:
            print(f"\r  WARNING: {e}")
            return False

    def run(self):
        """Main processing loop.

        Continuously processes frames from RTSP stream with detection,
        tracking, and cross-line detection. Optionally pushes to SEI pipeline.
        """
        if not self.setup():
            print("\nSetup failed. Exiting.")
            return

        try:
            for _, imgs, _ in self.stream_loader:
                frame = imgs[0]
                current_timestamp = time.time()

                # 1. Run detection
                try:
                    results = self.predictor(frame)
                    boxes = results[0].boxes if results else None

                    if boxes is None or len(boxes) == 0:
                        tracked_objects = np.array([])
                        detections = np.array([])
                    else:
                        # 2. Prepare tracking input
                        cls_index = boxes.cls.cpu().numpy()
                        xywh = boxes.xywh.cpu().numpy()
                        conf = boxes.conf.cpu().numpy()
                        detections = np.column_stack([xywh, conf, cls_index])

                        # 3. Update tracker
                        tracked_objects = self.tracker.update(detections, frame)

                except Exception as e:
                    print(f"ERROR in inference: {e}")
                    continue

                # 4. Check line crossings
                events = []
                try:
                    for obj in tracked_objects:
                        x1, y1, x2, y2, track_id, score, cls_id = obj[:7]
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        track_id = int(track_id)

                        # Update position tracking
                        if track_id in self.track_positions:
                            crossing = self._check_line_crossing(
                                self.track_positions[track_id], (cx, cy)
                            )

                            if crossing != 0 and track_id not in self.crossed_ids:
                                self.crossed_ids.add(track_id)

                                if crossing > 0:
                                    self.count_in += 1
                                    direction = "IN"
                                else:
                                    self.count_out += 1
                                    direction = "OUT"

                                event = {
                                    "type": "line_crossing",
                                    "track_id": track_id,
                                    "class_id": int(cls_id),
                                    "direction": direction.lower(),
                                    "position": (float(cx), float(cy)),
                                    "timestamp": current_timestamp,
                                    "count_in": self.count_in,
                                    "count_out": self.count_out
                                }
                                events.append(event)
                                self.event_count += 1

                                # Print event
                                time_str = time.strftime('%H:%M:%S')
                                print(
                                    f"[{time_str}] CROSS-LINE EVENT: Track #{track_id} crossed {direction}\n"
                                    f"  Position: ({cx:.0f}, {cy:.0f}), Confidence: {score:.2f}\n"
                                    f"  Total: IN={self.count_in} OUT={self.count_out}"
                                )

                        # Update position
                        self.track_positions[track_id] = (cx, cy)

                except Exception as e:
                    print(f"ERROR in line crossing detection: {e}")

                # 5. Push to SEI pipeline (if enabled) - ASYNC with FRAME SKIPPING
                if self.sei_pipeline:
                    self.sei_frame_counter += 1

                    # Only push every Nth frame to avoid queue overflow
                    if self.sei_frame_counter >= self.sei_frame_skip:
                        self.sei_frame_counter = 0

                        try:
                            # Format detections as list of dicts for SEI events module
                            det_list = []
                            if len(detections) > 0:
                                for det in detections:
                                    x, y, w, h, conf, cls = det[:6]
                                    det_list.append({
                                        "bbox": [float(x), float(y), float(w), float(h)],
                                        "confidence": float(conf),
                                        "class": int(cls)
                                    })

                            # Format tracks as list of dicts
                            track_list = []
                            if len(tracked_objects) > 0:
                                for track in tracked_objects:
                                    x1, y1, x2, y2, track_id, score, cls_id = track[:7]
                                    track_list.append({
                                        "track_id": int(track_id),
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "confidence": float(score),
                                        "class": int(cls_id)
                                    })

                            inference_data = {
                                "detections": det_list,
                                "tracks": track_list,
                                "events": events
                            }

                            # Use async push - returns immediately, non-blocking
                            success = self.sei_pipeline.push_frame_async(
                                frame,
                                inference_data=inference_data,
                                task_id="cross_line_test"
                            )

                            if not success:
                                # Queue full - frame dropped
                                self.dropped_frames += 1
                                if self.dropped_frames % 10 == 1:  # Log every 10th drop
                                    print(f"WARNING: SEI queue full, {self.dropped_frames} frames dropped")

                        except Exception as e:
                            print(f"WARNING: SEI push failed: {e}")

                self.frame_count += 1

                # 6. Print statistics every 10 seconds
                current_time = time.time()
                if current_time - self.last_stats_time >= 10.0:
                    self._print_statistics()
                    self.last_stats_time = current_time

        except KeyboardInterrupt:
            print("\n\nTest interrupted by user (Ctrl+C)")

        finally:
            self.cleanup()

    def _check_line_crossing(
        self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float]
    ) -> int:
        """Check if movement crosses the defined line.

        Uses cross-product method to determine which side of the line
        a point is on.

        Args:
            prev_pos: Previous position (x, y).
            curr_pos: Current position (x, y).

        Returns:
            0 if no crossing, 1 for positive direction, -1 for negative.
        """
        if len(self.line_coords) < 2:
            return 0

        p1, p2 = self.line_coords[0], self.line_coords[1]

        def side(point: Tuple[float, float]) -> float:
            """Calculate which side of line a point is on using cross product."""
            return ((p2[0] - p1[0]) * (point[1] - p1[1]) -
                    (p2[1] - p1[1]) * (point[0] - p1[0]))

        prev_side = side(prev_pos)
        curr_side = side(curr_pos)

        # Check if they are on different sides of the line
        if prev_side * curr_side < 0:
            # Crossed the line
            return 1 if curr_side > 0 else -1

        return 0

    def _print_statistics(self):
        """Print processing statistics."""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print("-" * 80)
        print(f"[{time.strftime('%H:%M:%S')}] Statistics (runtime: {elapsed:.1f}s)")
        print(f"  Frames processed:   {self.frame_count} ({fps:.1f} fps)")
        print(f"  Cross-line events:  {self.event_count} ({self.count_in} IN, {self.count_out} OUT)")
        print(f"  Active tracks:      {len(self.track_positions)}")

        if self.sei_pipeline:
            try:
                stats = self.sei_pipeline.get_statistics()
                sei_fps = stats.frames_pushed / elapsed if elapsed > 0 else 0
                print(f"  SEI streaming:      OK ({stats.frames_pushed} frames, {sei_fps:.1f} fps)")
                if self.dropped_frames > 0:
                    print(f"  Dropped frames:     {self.dropped_frames}")
                print(f"  Recordings:         {stats.recordings_completed} completed")
            except Exception:
                pass

        print("-" * 80)

    def cleanup(self):
        """Clean up all resources in reverse order."""
        print("\nCleaning up resources...")

        # 1. Stop SEI pipeline
        if self.sei_pipeline:
            print("  Stopping SEI pipeline...")
            try:
                self.sei_pipeline.stop()
                stats = self.sei_pipeline.get_statistics()
                print(f"    Frames pushed: {stats.frames_pushed}")
                print(f"    Recordings: {stats.recordings_completed}")
            except Exception as e:
                print(f"    Error stopping pipeline: {e}")
            finally:
                self.sei_pipeline = None

        # 2. Close stream loader
        if self.stream_loader:
            print("  Closing stream...")
            try:
                self.stream_loader.close()
            except Exception as e:
                print(f"    Error closing stream: {e}")
            finally:
                self.stream_loader = None

        # 3. Reset tracker
        if self.tracker:
            try:
                self.tracker.reset()
            except Exception:
                pass
            finally:
                self.tracker = None

        print("Cleanup complete.")

        # Print final summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"Runtime:              {elapsed:.1f} seconds")
        print(f"Total frames:         {self.frame_count}")
        print(f"Frame rate:           {fps:.1f} fps")
        print(f"Cross-line events:    {self.event_count} ({self.count_in} IN, {self.count_out} OUT)")
        print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Cross-line detection end-to-end test with optional SEI streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with default RTSP
  python tests/test_cross_line.py

  # Custom model and stream
  python tests/test_cross_line.py -m weights/yolo11n.pt -r rtsp://cam/stream

  # Enable SEI streaming and recording
  python tests/test_cross_line.py --enable-sei-streaming --enable-recording

  # Full configuration
  python tests/test_cross_line.py \\
    -m weights/yolo11m.pt \\
    -r rtsp://192.168.2.71:8554/mystream3 \\
    --enable-sei-streaming \\
    --enable-recording \\
    --output-dir output/recordings
        """
    )

    parser.add_argument(
        "-m", "--model",
        default="weights/yolo11m.pt",
        help="Path to YOLO model weights (default: weights/yolo11m.pt)"
    )

    parser.add_argument(
        "-r", "--rtsp-url",
        default="rtsp://192.168.2.71:8554/mystream3",
        help="RTSP stream URL (default: rtsp://192.168.2.71:8554/mystream3)"
    )

    parser.add_argument(
        "--enable-sei-streaming",
        action="store_true",
        help="Enable SEI data injection and RTMP streaming"
    )

    parser.add_argument(
        "--enable-recording",
        action="store_true",
        help="Enable event-triggered video recording"
    )

    parser.add_argument(
        "--rtmp-url",
        default="rtmp://192.168.2.234:1935/cross_line/sei_cross_line_001",
        help="RTMP server URL for streaming (default: rtmp://192.168.2.234:1935/cross_line/sei_cross_line_001)"
    )

    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Output directory for recorded videos (default: recordings)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device: cuda, cpu, etc. (default: cuda)"
    )

    parser.add_argument(
        "--sei-frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame for SEI streaming (default: 3 for ~8-10 FPS output)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create configuration
    config = CrossLineTestConfig(
        model_path=args.model,
        rtsp_url=args.rtsp_url,
        enable_sei_streaming=args.enable_sei_streaming,
        enable_recording=args.enable_recording,
        rtmp_url=args.rtmp_url,
        output_dir=args.output_dir,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        sei_frame_skip=args.sei_frame_skip
    )

    # Create and run test
    test = CrossLineTest(config)
    test.run()


if __name__ == "__main__":
    main()
