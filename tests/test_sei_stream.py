# -*- coding: utf-8 -*-
"""
SEI Stream Integration Test Script

This module provides integration tests for the SEI (Supplemental Enhancement
Information) streaming pipeline, validating the end-to-end workflow of:
- Reading video frames from an RTSP stream
- Injecting custom test SEI data into video frames
- Optionally pushing the stream with SEI data to RTMP
- Optionally saving the processed video to disk

Test Configuration:
<<<<<<< HEAD
    - Default RTSP source: rtsp://192.168.2.71:8554/mystream3
=======
    - Default RTSP source: rtsp://192.168.2.71:8554/mystream1
>>>>>>> 07331326 (feat: build video analytics task management system)
    - Default RTMP target: rtmp://192.168.2.234:1935/live/sei_test_001
    - Default output directory: recordings/

Usage:
    # Run as pytest (tests are skipped by default due to resource requirements)
    pytest tests/test_sei_stream.py -v

    # Run as standalone script for manual testing
    python tests/test_sei_stream.py --enable-recording --max-frames 50

    # Full test with RTMP streaming
    python tests/test_sei_stream.py --enable-streaming --enable-recording --max-frames 100

    # Test with frame skipping (process every 2nd frame)
    python tests/test_sei_stream.py --enable-recording --frame-skip 2 --max-frames 50
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Configuration Constants
# =============================================================================

<<<<<<< HEAD
TEST_RTSP_SOURCE = "rtsp://192.168.2.71:8554/mystream3"
TEST_RTMP_TARGET = "rtmp://192.168.2.234:1935/live/sei_test_001"
TEST_OUTPUT_DIR = PROJECT_ROOT / "recordings"
DEFAULT_FRAME_SKIP = 1  # Process every frame
DEFAULT_MAX_FRAMES = 1000  # Number of frames to process
=======
TEST_RTSP_SOURCE = "rtsp://192.168.2.71:8554/mystream1"
TEST_RTMP_TARGET = "rtmp://192.168.2.234:1935/live/sei_test_001"
TEST_OUTPUT_DIR = PROJECT_ROOT / "recordings"
DEFAULT_FRAME_SKIP = 1  # Process every frame
DEFAULT_MAX_FRAMES = 10000  # Number of frames to process
>>>>>>> 07331326 (feat: build video analytics task management system)
DEFAULT_TRIGGER_FRAME = 30  # Frame number to trigger recording event
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 10.0


# =============================================================================
# Helper Functions
# =============================================================================

def create_test_sei_data(frame_index: int) -> Dict[str, Any]:
    """
    Generate simple test SEI payload data.

    Creates a test payload with basic markers for validation,
    without mock detection data.

    Args:
        frame_index: Current frame number

    Returns:
        Dictionary containing test SEI data
    """
    return {
        "test_id": "sei_test_001",
        "frame_index": frame_index,
        "timestamp": time.time(),
        "marker": "SEI_INJECTION_TEST"
    }


def log_progress(frame_count: int, stats: Any, interval: int = 10) -> None:
    """
    Log processing progress at specified intervals.

    Args:
        frame_count: Current frame count
        stats: Pipeline statistics object
        interval: Logging interval in frames
    """
    if frame_count % interval == 0:
        print(f"Frame {frame_count}: "
              f"encoded={stats.frames_encoded}, "
              f"sei={stats.frames_with_sei}, "
              f"fps={stats.fps:.2f}")


# =============================================================================
# Core Test Function
# =============================================================================

def run_sei_stream_test(
    rtsp_source: str = TEST_RTSP_SOURCE,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    enable_streaming: bool = False,
    rtmp_url: str = TEST_RTMP_TARGET,
    enable_recording: bool = True,
    output_dir: str = str(TEST_OUTPUT_DIR),
    max_frames: int = DEFAULT_MAX_FRAMES,
    trigger_frame: int = DEFAULT_TRIGGER_FRAME
) -> Dict[str, Any]:
    """
    Run the SEI stream integration test.

    This function demonstrates the end-to-end SEI injection and streaming
    workflow by:
    1. Reading frames from an RTSP stream
    2. Injecting custom SEI data into each frame
    3. Optionally pushing to RTMP server
    4. Optionally recording to MP4 file
    5. Triggering a recording event at a specified frame

    Args:
        rtsp_source: RTSP URL for video input
        frame_skip: Process every Nth frame (1 = every frame)
        enable_streaming: Enable RTMP streaming output
        rtmp_url: RTMP server URL for streaming
        enable_recording: Enable MP4 recording output
        output_dir: Directory for recording output
        max_frames: Maximum frames to process
        trigger_frame: Frame number at which to trigger recording event

    Returns:
        Dictionary containing test results and statistics
    """
    from stream import StreamLoader
    from stream.exceptions import StreamConnectionError
    from sei import (
        SeiStreamingPipeline,
        SeiConfig,
        StreamConfig,
        OutputConfig,
        RecorderConfig,
        BufferConfig
    )

    print("=" * 60)
    print("SEI Stream Integration Test")
    print("=" * 60)
    print(f"RTSP Source: {rtsp_source}")
    print(f"Frame Skip: {frame_skip}")
    print(f"Streaming: {enable_streaming} -> {rtmp_url if enable_streaming else 'N/A'}")
    print(f"Recording: {enable_recording} -> {output_dir if enable_recording else 'N/A'}")
    print(f"Max Frames: {max_frames}")
    print(f"Event Trigger Frame: {trigger_frame}")
    print("=" * 60)

    results = {
        "success": False,
        "frames_processed": 0,
        "frames_encoded": 0,
        "frames_with_sei": 0,
        "event_triggered": False,
        "recording_path": None,
        "error": None
    }

    loader = None
    pipeline = None

    try:
        # Initialize StreamLoader
        print("\n[1/5] Initializing stream loader...")
        try:
            loader = StreamLoader(
                sources=rtsp_source,
                vid_stride=frame_skip,
                buffer=False  # Keep only latest frame for real-time processing
            )
        except StreamConnectionError as e:
            results["error"] = f"Failed to connect to RTSP stream: {e}"
            print(f"ERROR: {results['error']}")
            return results

        # Get stream properties
        props = loader.get_stream_properties()
        width = props.get('width', DEFAULT_WIDTH)
        height = props.get('height', DEFAULT_HEIGHT)
        fps = props.get('fps', DEFAULT_FPS)

        # Adjust FPS for frame skipping
        effective_fps = fps / frame_skip if frame_skip > 1 else fps

        print(f"Stream properties: {width}x{height} @ {fps:.1f} FPS")
        print(f"Effective FPS (after skip): {effective_fps:.1f}")

        # Configure pipeline
        print("\n[2/5] Configuring SEI pipeline...")
        sei_config = SeiConfig(enable=True)
        stream_config = StreamConfig(
            width=width,
            height=height,
            fps=effective_fps,
            rtmp_url=rtmp_url if enable_streaming else ""
        )
        recorder_config = RecorderConfig(
            output_dir=output_dir,
            filename_pattern="{task_id}_{timestamp}_{event_type}.mp4",
            pre_event_frames=30,
            post_event_frames=50
        )
        buffer_config = BufferConfig(
            capacity=150,
            pre_event_frames=30,
            post_event_frames=50
        )
        output_config = OutputConfig(
            rtmp_enabled=enable_streaming,
            recording_enabled=enable_recording,
            stream_config=stream_config,
            recorder_config=recorder_config,
            buffer_config=buffer_config
        )

        # Create pipeline
        pipeline = SeiStreamingPipeline(
            sei_config=sei_config,
            stream_config=stream_config,
            output_config=output_config,
            log_func=lambda msg: print(f"[SEI] {msg}")
        )

        # Start pipeline
        print("\n[3/5] Starting pipeline...")
        rtmp_target = rtmp_url if enable_streaming else None
        if not pipeline.start(rtmp_target):
            results["error"] = "Failed to start pipeline"
            print(f"ERROR: {results['error']}")
            return results

        print("Pipeline started successfully")
        if enable_streaming:
            print(f"Streaming to: {rtmp_url}")
        if enable_recording:
            print(f"Recording enabled, output dir: {output_dir}")

        # Process frames
        print("\n[4/5] Processing frames...")
        task_id = "sei_test_001"
        frame_count = 0
        event_triggered = False
        event_id = None

        start_time = time.time()

        for sources, frames, _ in loader:
            for frame in frames:
                # Create test SEI data
                sei_data = create_test_sei_data(frame_count)

                # Push frame through pipeline
                result = pipeline.push_frame(
                    frame=frame,
                    inference_data=None,  # Simple test markers only
                    custom_data=sei_data,
                    task_id=task_id
                )

                # Trigger event at specified frame
                if (frame_count == trigger_frame and
                    enable_recording and
                    not event_triggered):
                    success = pipeline.trigger_event(
                        "test_event",
                        task_id,
                        {"trigger_type": "timed", "frame": frame_count}
                    )
                    if success:
                        event_triggered = True
                        results["event_triggered"] = True
                        print(f"\n>>> Recording event triggered at frame {frame_count}")
                    else:
                        print(f"\n>>> Failed to trigger event at frame {frame_count}")

                # Log progress
                if frame_count % 10 == 0:
                    stats = pipeline.get_statistics()
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame_count}: "
                          f"encoded={stats.frames_encoded}, "
                          f"sei={stats.frames_with_sei}, "
                          f"actual_fps={actual_fps:.2f}")

                frame_count += 1
                if frame_count >= max_frames:
                    break

            if frame_count >= max_frames:
                break

        # Get final statistics
        print("\n[5/5] Collecting final statistics...")
        stats = pipeline.get_statistics()
        elapsed = time.time() - start_time

        results["success"] = True
        results["frames_processed"] = frame_count
        results["frames_encoded"] = stats.frames_encoded
        results["frames_with_sei"] = stats.frames_with_sei

        # Print summary
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Frames Processed: {frame_count}")
        print(f"Frames Encoded: {stats.frames_encoded}")
        print(f"Frames with SEI: {stats.frames_with_sei}")
        print(f"Events Triggered: {stats.events_triggered}")
        print(f"Recordings Completed: {stats.recordings_completed}")
        print(f"Total Duration: {elapsed:.2f}s")
        print(f"Average FPS: {frame_count / elapsed if elapsed > 0 else 0:.2f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        results["error"] = "Interrupted by user"

    except Exception as e:
        results["error"] = str(e)
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")
        if pipeline is not None:
            try:
                pipeline.stop()
                print("Pipeline stopped")
            except Exception as e:
                print(f"Error stopping pipeline: {e}")

        if loader is not None:
            try:
                loader.close()
                print("Stream loader closed")
            except Exception as e:
                print(f"Error closing loader: {e}")

    return results


# =============================================================================
# pytest Test Cases
# =============================================================================

class TestSeiStream:
    """Integration tests for SEI streaming pipeline."""

    @pytest.mark.skip(reason="Requires RTSP stream")
    def test_stream_with_sei_injection(self):
        """Test basic SEI injection without streaming or recording."""
        results = run_sei_stream_test(
            enable_streaming=False,
            enable_recording=False,
            max_frames=30
        )
        assert results["success"]
        assert results["frames_processed"] > 0
        assert results["frames_with_sei"] > 0

    @pytest.mark.skip(reason="Requires RTSP stream and RTMP server")
    def test_stream_with_rtmp_push(self):
        """Test SEI streaming to RTMP server."""
        results = run_sei_stream_test(
            enable_streaming=True,
            enable_recording=False,
            max_frames=50
        )
        assert results["success"]
        assert results["frames_processed"] > 0

    @pytest.mark.skip(reason="Requires RTSP stream")
    def test_stream_with_recording(self):
        """Test video recording with SEI data and event trigger."""
        results = run_sei_stream_test(
            enable_streaming=False,
            enable_recording=True,
            max_frames=50,
            trigger_frame=20
        )
        assert results["success"]
        assert results["event_triggered"]

    @pytest.mark.skip(reason="Requires RTSP stream")
    def test_frame_skipping(self):
        """Test that frame skip parameter affects processing."""
        # Test with skip=1 (every frame)
        results1 = run_sei_stream_test(
            frame_skip=1,
            enable_streaming=False,
            enable_recording=False,
            max_frames=30
        )

        # Test with skip=2 (every other frame)
        results2 = run_sei_stream_test(
            frame_skip=2,
            enable_streaming=False,
            enable_recording=False,
            max_frames=30
        )

        assert results1["success"]
        assert results2["success"]

    @pytest.mark.skip(reason="Requires RTSP stream")
    def test_full_pipeline(self):
        """Test complete end-to-end workflow."""
        results = run_sei_stream_test(
            enable_streaming=False,  # Skip RTMP for basic test
            enable_recording=True,
            max_frames=100,
            trigger_frame=30
        )
        assert results["success"]
        assert results["frames_processed"] == 100
        assert results["frames_with_sei"] > 0
        assert results["event_triggered"]


# =============================================================================
# Standalone Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SEI Stream Integration Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with recording only
  python tests/test_sei_stream.py --enable-recording --max-frames 50

  # Test with RTMP streaming
  python tests/test_sei_stream.py --enable-streaming --max-frames 100

  # Full test with both streaming and recording
  python tests/test_sei_stream.py --enable-streaming --enable-recording

  # Test with frame skipping (process every 3rd frame)
  python tests/test_sei_stream.py --frame-skip 3 --enable-recording

  # Custom RTSP source
  python tests/test_sei_stream.py --rtsp rtsp://192.168.1.100:554/stream
        """
    )

    parser.add_argument(
        "--rtsp",
        default=TEST_RTSP_SOURCE,
        help=f"RTSP source URL (default: {TEST_RTSP_SOURCE})"
    )
    parser.add_argument(
        "--rtmp",
        default=TEST_RTMP_TARGET,
        help=f"RTMP target URL (default: {TEST_RTMP_TARGET})"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help=f"Process every Nth frame (default: {DEFAULT_FRAME_SKIP})"
    )
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable RTMP streaming output"
    )
    parser.add_argument(
        "--enable-recording",
        action="store_true",
        help="Enable MP4 recording output"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Maximum frames to process (default: {DEFAULT_MAX_FRAMES})"
    )
    parser.add_argument(
        "--output-dir",
        default=str(TEST_OUTPUT_DIR),
        help=f"Recording output directory (default: {TEST_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--trigger-frame",
        type=int,
        default=DEFAULT_TRIGGER_FRAME,
        help=f"Frame number to trigger recording event (default: {DEFAULT_TRIGGER_FRAME})"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run test
    results = run_sei_stream_test(
        rtsp_source=args.rtsp,
        frame_skip=args.frame_skip,
        enable_streaming=args.enable_streaming,
        rtmp_url=args.rtmp,
        enable_recording=args.enable_recording,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        trigger_frame=args.trigger_frame
    )

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)
