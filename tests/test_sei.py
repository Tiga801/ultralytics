# -*- coding: utf-8 -*-
"""
SEI Video Streaming Module Tests

This module provides comprehensive tests for the SEI video streaming pipeline,
including unit tests for individual components and integration tests for the
complete workflow.

Test Configuration:
    - Test video stream: rtsp://192.168.2.71:8554/mystream3
    - Test model: weights/yolo11m.pt
    - Output directory: results/

Test Categories:
    1. Unit Tests: Individual component functionality
    2. Integration Tests: Component interaction
    3. End-to-End Tests: Complete workflow with RTSP/inference

Usage:
    # Run all tests
    pytest tests/test_sei.py -v

    # Run specific test class
    pytest tests/test_sei.py::TestNalUtils -v

    # Run with coverage
    pytest tests/test_sei.py --cov=sei --cov-report=html
"""

import sys
import time
import json
import tempfile
import threading
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Configuration
# =============================================================================

TEST_RTSP_URL = "rtsp://192.168.2.71:8554/mystream3"
TEST_RTMP_URL = "rtmp://192.168.2.229/live/test"
TEST_MODEL_PATH = "weights/yolo11m.pt"
TEST_OUTPUT_DIR = PROJECT_ROOT / "results"
TEST_WIDTH = 640
TEST_HEIGHT = 480
TEST_FPS = 10.0


def create_test_frame(width: int = TEST_WIDTH, height: int = TEST_HEIGHT) -> np.ndarray:
    """Create a test frame (random BGR image)."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_test_h264_data() -> bytes:
    """Create minimal H.264 test data with SPS, PPS, IDR."""
    # Minimal H.264 NAL units for testing
    sps = b'\x00\x00\x00\x01\x67\x42\x00\x1e\x8d\x68\x10\x00'  # SPS
    pps = b'\x00\x00\x00\x01\x68\xce\x3c\x80'  # PPS
    idr = b'\x00\x00\x00\x01\x65' + bytes(100)  # IDR slice (simplified)
    return sps + pps + idr


def create_test_inference_data() -> dict:
    """Create test inference data."""
    return {
        "tracks": [
            {
                "track_id": 1,
                "class": "person",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 300]
            },
            {
                "track_id": 2,
                "class": "car",
                "confidence": 0.87,
                "bbox": [300, 200, 500, 400]
            }
        ],
        "events": [],
        "frame_count": 1
    }


# =============================================================================
# NAL Utilities Tests
# =============================================================================

class TestNalUtils:
    """Tests for NAL unit utilities."""

    def test_split_nalus_basic(self):
        """Test basic NAL unit splitting."""
        from sei.nalutils import split_nalus, NAL_START_CODE_4

        # Create test data with multiple NALs
        data = (
            NAL_START_CODE_4 + b'\x67' + bytes(10) +  # SPS
            NAL_START_CODE_4 + b'\x68' + bytes(5) +   # PPS
            NAL_START_CODE_4 + b'\x65' + bytes(50)    # IDR
        )

        nalus = split_nalus(data)

        assert len(nalus) == 3
        assert nalus[0].startswith(NAL_START_CODE_4)
        assert nalus[1].startswith(NAL_START_CODE_4)
        assert nalus[2].startswith(NAL_START_CODE_4)

    def test_split_nalus_3byte_start_code(self):
        """Test splitting with 3-byte start codes."""
        from sei.nalutils import split_nalus, NAL_START_CODE_3

        data = (
            NAL_START_CODE_3 + b'\x67' + bytes(10) +
            NAL_START_CODE_3 + b'\x68' + bytes(5)
        )

        nalus = split_nalus(data)
        assert len(nalus) == 2

    def test_split_nalus_empty(self):
        """Test splitting empty data."""
        from sei.nalutils import split_nalus

        nalus = split_nalus(b'')
        assert nalus == []

    def test_get_nal_type(self):
        """Test NAL type extraction."""
        from sei.nalutils import (
            get_nal_type, NAL_START_CODE_4,
            NAL_TYPE_SPS, NAL_TYPE_PPS, NAL_TYPE_IDR, NAL_TYPE_SEI
        )

        # Test SPS (type 7)
        sps = NAL_START_CODE_4 + b'\x67' + bytes(10)
        assert get_nal_type(sps) == NAL_TYPE_SPS

        # Test PPS (type 8)
        pps = NAL_START_CODE_4 + b'\x68' + bytes(5)
        assert get_nal_type(pps) == NAL_TYPE_PPS

        # Test IDR (type 5)
        idr = NAL_START_CODE_4 + b'\x65' + bytes(50)
        assert get_nal_type(idr) == NAL_TYPE_IDR

        # Test SEI (type 6)
        sei = NAL_START_CODE_4 + b'\x06' + bytes(20)
        assert get_nal_type(sei) == NAL_TYPE_SEI

    def test_is_video_frame_nal(self):
        """Test video frame NAL detection."""
        from sei.nalutils import is_video_frame_nal

        assert is_video_frame_nal(1) is True   # Non-IDR
        assert is_video_frame_nal(5) is True   # IDR
        assert is_video_frame_nal(6) is False  # SEI
        assert is_video_frame_nal(7) is False  # SPS

    def test_is_keyframe_nal(self):
        """Test keyframe detection."""
        from sei.nalutils import is_keyframe_nal

        assert is_keyframe_nal(5) is True   # IDR
        assert is_keyframe_nal(1) is False  # Non-IDR

    def test_make_sei_user_data_unregistered(self):
        """Test SEI NAL unit creation."""
        from sei.nalutils import (
            make_sei_user_data_unregistered,
            NAL_START_CODE_3, NAL_TYPE_SEI,
            get_nal_type
        )

        payload = b'{"frame": 1, "data": "test"}'
        sei_nalu = make_sei_user_data_unregistered(payload)

        # Check start code
        assert sei_nalu.startswith(NAL_START_CODE_3)

        # Check NAL type
        assert get_nal_type(sei_nalu) == NAL_TYPE_SEI

        # Check payload is embedded
        assert payload in sei_nalu

    def test_make_sei_with_custom_uuid(self):
        """Test SEI creation with custom UUID."""
        from sei.nalutils import make_sei_user_data_unregistered

        payload = b'test'
        custom_uuid = b'CUSTOM_UUID_TEST'

        sei_nalu = make_sei_user_data_unregistered(payload, custom_uuid)

        assert custom_uuid in sei_nalu

    def test_inject_sei_into_h264_data(self):
        """Test SEI injection into H.264 stream."""
        from sei.nalutils import (
            inject_sei_into_h264_data, split_nalus,
            get_nal_type, NAL_TYPE_SEI
        )

        h264_data = create_test_h264_data()
        payload = b'{"test": "data"}'

        enhanced = inject_sei_into_h264_data(h264_data, payload)

        # Should be larger (SEI added)
        assert len(enhanced) > len(h264_data)

        # Should contain SEI NAL
        nalus = split_nalus(enhanced)
        sei_found = any(get_nal_type(n) == NAL_TYPE_SEI for n in nalus)
        assert sei_found

    def test_parse_sei_payload(self):
        """Test SEI payload parsing."""
        from sei.nalutils import (
            make_sei_user_data_unregistered,
            parse_sei_payload
        )

        original_payload = b'{"frame": 123, "data": "test"}'
        sei_nalu = make_sei_user_data_unregistered(original_payload)

        parsed = parse_sei_payload(sei_nalu)

        assert parsed is not None
        assert original_payload in parsed or parsed == original_payload


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfig:
    """Tests for configuration classes."""

    def test_encoder_config_defaults(self):
        """Test encoder config default values."""
        from sei.config import EncoderConfig

        config = EncoderConfig()

        assert config.preset == "ultrafast"
        assert config.profile == "baseline"
        assert config.crf == 23
        assert config.tune == "zerolatency"

    def test_encoder_config_to_ffmpeg_args(self):
        """Test FFmpeg argument generation."""
        from sei.config import EncoderConfig

        config = EncoderConfig(preset="fast", crf=20)
        args = config.to_ffmpeg_args()

        assert "-preset" in args
        assert "fast" in args
        assert "-crf" in args
        assert "20" in args

    def test_sei_config_defaults(self):
        """Test SEI config default values."""
        from sei.config import SeiConfig, DEFAULT_SEI_UUID

        config = SeiConfig()

        assert config.enable is True
        assert config.uuid == DEFAULT_SEI_UUID
        assert config.insert_interval == 1

    def test_sei_config_custom_uuid(self):
        """Test SEI config with custom UUID string."""
        from sei.config import SeiConfig

        config = SeiConfig(custom_uuid_string="MY_CUSTOM_UUID")

        assert len(config.uuid) == 16
        assert config.uuid.startswith(b'MY_CUSTOM_UUID')

    def test_stream_config_defaults(self):
        """Test stream config default values."""
        from sei.config import StreamConfig

        config = StreamConfig()

        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 10.0
        assert config.ffmpeg_path == "ffmpeg"
    
    def test_stream_config_frame_interval(self):
        """Test frame interval calculation."""
        from sei.config import StreamConfig

        config = StreamConfig(fps=10.0)
        assert config.frame_interval == 0.1

        config = StreamConfig(fps=30.0)
        assert abs(config.frame_interval - 0.0333) < 0.001
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        from sei.config import StreamConfig

        config_dict = {
            "width": 1280,
            "height": 720,
            "fps": 30.0,
            "rtmp_url": "rtmp://test/live"
        }

        config = StreamConfig.from_dict(config_dict)

        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30.0

    def test_buffer_config_validation(self):
        """Test buffer config validation."""
        from sei.config import BufferConfig

        # Should auto-adjust capacity if too small
        config = BufferConfig(capacity=50, pre_event_frames=50, post_event_frames=50)
        assert config.capacity >= 100

    def test_recorder_config_defaults(self):
        """Test recorder config defaults."""
        from sei.config import RecorderConfig

        config = RecorderConfig()

        assert config.output_dir == "recordings"
        assert config.container == "mp4"
        assert config.codec == "libx264"
        assert config.pre_event_frames == 50
        assert config.post_event_frames == 50


# =============================================================================
# Interface Tests
# =============================================================================

class TestInterfaces:
    """Tests for interface data classes."""

    def test_sei_payload_creation(self):
        """Test SEI payload creation."""
        from sei.interfaces import SeiPayload

        payload = SeiPayload(
            frame_timestamp=time.time(),
            inference_data={"tracks": []},
            frame_count=1
        )

        assert payload.injection_time is not None
        assert payload.frame_count == 1

    def test_sei_payload_to_json(self):
        """Test SEI payload JSON serialization."""
        from sei.interfaces import SeiPayload

        payload = SeiPayload(
            frame_timestamp=12345.67,
            inference_data={"test": "data"},
            frame_count=10
        )

        json_str = payload.to_json()
        data = json.loads(json_str)

        assert "ts" in data  # frame_timestamp
        assert "fc" in data  # frame_count
        assert data["fc"] == 10

    def test_sei_payload_to_bytes(self):
        """Test SEI payload byte conversion."""
        from sei.interfaces import SeiPayload

        payload = SeiPayload(frame_timestamp=time.time())
        payload_bytes = payload.to_bytes()

        assert isinstance(payload_bytes, bytes)
        assert len(payload_bytes) > 0

    def test_encoded_frame_creation(self):
        """Test encoded frame creation."""
        from sei.interfaces import EncodedFrame

        frame = EncodedFrame(
            h264_data=b'\x00\x00\x00\x01\x65' + bytes(100),
            timestamp=time.time(),
            frame_index=0,
            is_keyframe=True
        )

        assert frame.size == 105
        assert frame.is_keyframe is True
        assert frame.has_sei() is False

    def test_streaming_statistics(self):
        """Test streaming statistics."""
        from sei.interfaces import StreamingStatistics

        stats = StreamingStatistics(
            frames_pushed=100,
            frames_failed=5,
            start_time=time.time() - 10,
            last_frame_time=time.time()
        )

        assert stats.fps > 0
        assert stats.success_rate > 0.9


# =============================================================================
# Buffer Tests
# =============================================================================

class TestBuffer:
    """Tests for ring buffer."""

    def test_buffer_creation(self):
        """Test buffer creation."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig

        config = BufferConfig(capacity=100)
        buffer = FrameRingBuffer(config)

        assert len(buffer) == 0
        assert buffer.capacity == 100

    def test_buffer_push_pop(self):
        """Test buffer push and pop operations."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig
        from sei.interfaces import EncodedFrame

        buffer = FrameRingBuffer(BufferConfig(capacity=10))

        # Push frames
        for i in range(5):
            frame = EncodedFrame(
                h264_data=bytes(100),
                timestamp=float(i),
                frame_index=i
            )
            buffer.push(frame)

        assert len(buffer) == 5

        # Pop frame
        popped = buffer.pop()
        assert popped is not None
        assert popped.frame_index == 0
        assert len(buffer) == 4

    def test_buffer_overflow(self):
        """Test buffer overflow handling."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig
        from sei.interfaces import EncodedFrame

        buffer = FrameRingBuffer(BufferConfig(capacity=5, pre_event_frames=0, post_event_frames=0))

        # Push more than capacity
        for i in range(10):
            frame = EncodedFrame(
                h264_data=bytes(10),
                timestamp=float(i),
                frame_index=i
            )
            buffer.push(frame)

        # Should be at capacity
        assert len(buffer) == 5

        # Oldest should be frame 5 (0-4 dropped)
        oldest = buffer.peek_oldest()
        assert oldest.frame_index == 5

    def test_buffer_get_recent(self):
        """Test getting recent frames."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig
        from sei.interfaces import EncodedFrame

        buffer = FrameRingBuffer(BufferConfig(capacity=100, pre_event_frames=5))

        # Push frames
        for i in range(20):
            frame = EncodedFrame(
                h264_data=bytes(10),
                timestamp=float(i),
                frame_index=i
            )
            buffer.push(frame)

        # Get recent
        recent = buffer.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].frame_index == 19

    def test_buffer_pre_event_frames(self):
        """Test pre-event frame retrieval."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig
        from sei.interfaces import EncodedFrame

        buffer = FrameRingBuffer(BufferConfig(
            capacity=100,
            pre_event_frames=10
        ))

        # Push frames
        for i in range(50):
            frame = EncodedFrame(
                h264_data=bytes(10),
                timestamp=float(i),
                frame_index=i
            )
            buffer.push(frame)

        # Get pre-event frames
        pre_event = buffer.get_pre_event_frames()

        assert len(pre_event) == 10
        assert pre_event[0].frame_index == 40  # 50 - 10

    def test_buffer_thread_safety(self):
        """Test buffer thread safety."""
        from sei.buffer import FrameRingBuffer
        from sei.config import BufferConfig
        from sei.interfaces import EncodedFrame

        buffer = FrameRingBuffer(BufferConfig(capacity=100))
        errors = []

        def writer():
            try:
                for i in range(100):
                    frame = EncodedFrame(
                        h264_data=bytes(10),
                        timestamp=float(i),
                        frame_index=i
                    )
                    buffer.push(frame)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    buffer.get_recent(10)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Event System Tests
# =============================================================================

class TestEvents:
    """Tests for event system."""

    def test_sei_event_creation(self):
        """Test SEI event creation."""
        from sei.events import SeiEvent

        event = SeiEvent.create(
            event_type="cross_line",
            task_id="task001",
            metadata={"line_id": "line1"}
        )

        assert event.event_id.startswith("evt_")
        assert event.event_type == "cross_line"
        assert event.task_id == "task001"
        assert event.timestamp > 0

    def test_sei_event_to_dict(self):
        """Test event dictionary conversion."""
        from sei.events import SeiEvent

        event = SeiEvent.create("detection", "task001")
        event_dict = event.to_dict()

        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert event_dict["event_type"] == "detection"

    def test_event_trigger_detection(self):
        """Test detection-based event trigger."""
        from sei.events import EventTrigger

        config = {
            "type": "detection",
            "classes": ["person"],
            "min_confidence": 0.7,
            "cooldown": 0
        }
        trigger = EventTrigger(config)

        inference_data = {
            "detections": [
                {"class": "person", "confidence": 0.95},
                {"class": "car", "confidence": 0.85}
            ]
        }

        event = trigger.evaluate(inference_data, "task001")

        assert event is not None
        assert event.event_type == "detection"

    def test_event_trigger_count(self):
        """Test count-based event trigger."""
        from sei.events import EventTrigger

        config = {
            "type": "count",
            "classes": ["person"],
            "threshold": 3,
            "cooldown": 0
        }
        trigger = EventTrigger(config)

        # Not enough detections
        inference_data = {
            "detections": [
                {"class": "person", "confidence": 0.9},
                {"class": "person", "confidence": 0.8}
            ]
        }
        event = trigger.evaluate(inference_data, "task001")
        assert event is None

        # Enough detections
        inference_data["detections"].append({"class": "person", "confidence": 0.85})
        event = trigger.evaluate(inference_data, "task001")
        assert event is not None

    def test_event_trigger_cooldown(self):
        """Test event trigger cooldown."""
        from sei.events import EventTrigger

        config = {
            "type": "detection",
            "classes": ["person"],
            "cooldown": 5.0
        }
        trigger = EventTrigger(config)

        inference_data = {"detections": [{"class": "person", "confidence": 0.9}]}

        # First trigger
        event1 = trigger.evaluate(inference_data, "task001")
        assert event1 is not None

        # Should be blocked by cooldown
        event2 = trigger.evaluate(inference_data, "task001")
        assert event2 is None

    def test_event_trigger_disabled(self):
        """Test disabled event trigger."""
        from sei.events import EventTrigger

        config = {"type": "detection", "enabled": False}
        trigger = EventTrigger(config)

        inference_data = {"detections": [{"class": "person", "confidence": 0.9}]}
        event = trigger.evaluate(inference_data, "task001")

        assert event is None


# =============================================================================
# Injector Tests
# =============================================================================

class TestInjector:
    """Tests for SEI injector."""

    def test_injector_creation(self):
        """Test injector creation."""
        from sei.injector import SeiInjector
        from sei.config import SeiConfig

        config = SeiConfig(enable=True)
        injector = SeiInjector(config)

        assert injector.is_enabled() is True
        assert injector.injection_count == 0

    def test_injector_inject(self):
        """Test SEI injection."""
        from sei.injector import SeiInjector
        from sei.interfaces import SeiPayload

        injector = SeiInjector()
        h264_data = create_test_h264_data()
        payload = SeiPayload(
            frame_timestamp=time.time(),
            inference_data={"test": "data"},
            frame_count=1
        )

        enhanced = injector.inject(h264_data, payload)

        assert len(enhanced) > len(h264_data)
        assert injector.injection_count == 1

    def test_injector_disabled(self):
        """Test injector when disabled."""
        from sei.injector import SeiInjector
        from sei.config import SeiConfig
        from sei.interfaces import SeiPayload

        config = SeiConfig(enable=False)
        injector = SeiInjector(config)

        h264_data = create_test_h264_data()
        payload = SeiPayload(frame_timestamp=time.time())

        result = injector.inject(h264_data, payload)

        assert result == h264_data  # Unchanged
        assert injector.injection_count == 0

    def test_injector_set_uuid(self):
        """Test UUID setting."""
        from sei.injector import SeiInjector

        injector = SeiInjector()
        injector.set_uuid(b'NEW_UUID_BYTES')

        assert injector.uuid.startswith(b'NEW_UUID_BYTES')


# =============================================================================
# Recorder Tests
# =============================================================================

class TestRecorder:
    """Tests for MP4 recorder."""

    def test_recorder_creation(self):
        """Test recorder creation."""
        from sei.recorder import Mp4Recorder
        from sei.config import RecorderConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecorderConfig(output_dir=tmpdir)
            recorder = Mp4Recorder(config)

            assert recorder.is_recording() is False

    def test_recorder_output_path_generation(self):
        """Test output path generation."""
        from sei.recorder import Mp4Recorder
        from sei.config import RecorderConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecorderConfig(
                output_dir=tmpdir,
                filename_pattern="{task_id}_{event_type}.mp4"
            )
            recorder = Mp4Recorder(config)

            path = recorder._generate_output_path("evt001", "cross_line", "task001")

            assert "task001" in path
            assert "cross_line" in path
            assert path.endswith(".mp4")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for SEI pipeline components."""

    def test_encoder_injector_integration(self):
        """Test encoder + injector integration."""
        from sei.encoder import SimpleH264Encoder
        from sei.injector import SeiInjector
        from sei.config import StreamConfig
        from sei.interfaces import SeiPayload

        # Skip if FFmpeg not available
        pytest.importorskip("subprocess")

        config = StreamConfig(width=TEST_WIDTH, height=TEST_HEIGHT, fps=TEST_FPS)
        encoder = SimpleH264Encoder(config)
        injector = SeiInjector()

        encoder.start()

        try:
            frame = create_test_frame()
            h264_data = encoder.encode(frame)

            if h264_data:  # FFmpeg may not be available
                payload = SeiPayload(
                    frame_timestamp=time.time(),
                    inference_data=create_test_inference_data()
                )
                enhanced = injector.inject(h264_data, payload)

                assert len(enhanced) > len(h264_data)
        finally:
            encoder.stop()

    def test_buffer_recorder_integration(self):
        """Test buffer + recorder integration."""
        from sei.buffer import FrameRingBuffer
        from sei.recorder import Mp4Recorder
        from sei.config import BufferConfig, RecorderConfig
        from sei.interfaces import EncodedFrame

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create buffer and recorder
            buffer_config = BufferConfig(capacity=100, pre_event_frames=10)
            recorder_config = RecorderConfig(output_dir=tmpdir)

            buffer = FrameRingBuffer(buffer_config)
            recorder = Mp4Recorder(recorder_config)

            # Fill buffer with frames
            h264_data = create_test_h264_data()
            for i in range(20):
                frame = EncodedFrame(
                    h264_data=h264_data,
                    timestamp=float(i),
                    frame_index=i,
                    is_keyframe=(i % 10 == 0)
                )
                buffer.push(frame)

            # Get pre-event frames
            pre_event = buffer.get_pre_event_frames()
            assert len(pre_event) == 10

            # Note: Actual recording test would require FFmpeg


# =============================================================================
# End-to-End Tests (require RTSP stream)
# =============================================================================

class TestEndToEnd:
    """End-to-end tests requiring external resources."""

    @pytest.mark.skip(reason="Requires RTSP stream")
    def test_full_pipeline_rtsp(self):
        """Test complete pipeline with RTSP input."""
        import cv2
        from sei import (
            SeiStreamingPipeline,
            SeiConfig,
            StreamConfig
        )

        # Open RTSP stream
        cap = cv2.VideoCapture(TEST_RTSP_URL)
        if not cap.isOpened():
            pytest.skip("Cannot open RTSP stream")

        try:
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

            # Configure pipeline
            sei_config = SeiConfig(enable=True)
            stream_config = StreamConfig(width=width, height=height, fps=fps)

            pipeline = SeiStreamingPipeline(
                sei_config=sei_config,
                stream_config=stream_config
            )

            # Start pipeline (without RTMP for this test)
            pipeline.start()

            # Process frames
            for _ in range(50):
                ret, frame = cap.read()
                if not ret:
                    break

                inference_data = create_test_inference_data()
                result = pipeline.push_frame(frame, inference_data, task_id="test_task")

                assert result.encoded

            # Check statistics
            stats = pipeline.get_statistics()
            assert stats.frames_encoded > 0

            pipeline.stop()

        finally:
            cap.release()

    @pytest.mark.skip(reason="Requires RTSP stream and model")
    def test_full_workflow_with_inference(self):
        """Test complete workflow with model inference."""
        import cv2

        # This test would require:
        # 1. RTSP stream availability
        # 2. YOLO model at TEST_MODEL_PATH
        # 3. RTMP server for streaming
        pass


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
