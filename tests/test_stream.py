"""Test script for standalone stream module.

This script tests the StreamLoader class with various video sources
including RTSP streams and local video files.

Usage:
    # Run all tests
    pytest tests/test_stream.py -v

    # Run specific test
    pytest tests/test_stream.py::test_video_file -v

    # Run with stream tests (requires network access)
    pytest tests/test_stream.py -v -m "not slow"

    # Run as standalone script
    python tests/test_stream.py
"""

from __future__ import annotations

import numpy as np
import pytest
import sys

from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stream import (
    StreamLoader,
    StreamConnectionError,
    StreamReadError,
    StreamError,
    VID_FORMATS,
    STREAM_PROTOCOLS,
    is_stream_url,
    is_video_file,
    is_webcam,
    clean_str,
)

# Test constants
RTSP_STREAMS = [
    # "rtsp://admin:easyair@2025@192.168.2.137:554/LiveMedia/ch1/Media1",
    "rtsp://192.168.2.71:8554/mystream3"
]
VIDEO_FILE = Path("/home/easyair/ljwork/data/test_video/stream3.mp4")


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_clean_str(self):
        """Test clean_str removes special characters."""
        assert clean_str("test@url#123") == "test_url_123"
        assert clean_str("normal_string") == "normal_string"
        assert clean_str("rtsp://192.168.1.1:554/stream") == "rtsp___192.168.1.1_554_stream"

    def test_is_stream_url(self):
        """Test stream URL detection."""
        assert is_stream_url("rtsp://192.168.1.1:554/stream") is True
        assert is_stream_url("rtmp://server.com/live") is True
        assert is_stream_url("http://example.com/video.mp4") is True
        assert is_stream_url("https://example.com/stream") is True
        assert is_stream_url("tcp://192.168.1.1:5000") is True
        assert is_stream_url("/path/to/video.mp4") is False
        assert is_stream_url("video.mp4") is False
        assert is_stream_url("0") is False

    def test_is_video_file(self):
        """Test video file detection."""
        assert is_video_file("video.mp4") is True
        assert is_video_file("/path/to/video.avi") is True
        assert is_video_file("movie.mkv") is True
        assert is_video_file("image.jpg") is False
        assert is_video_file("document.txt") is False
        assert is_video_file("rtsp://stream.url") is False

    def test_is_webcam(self):
        """Test webcam detection."""
        assert is_webcam("0") is True
        assert is_webcam("1") is True
        assert is_webcam(0) is True
        assert is_webcam("video.mp4") is False
        assert is_webcam("rtsp://stream") is False


class TestConstants:
    """Test module constants."""

    def test_vid_formats(self):
        """Test video format set."""
        assert "mp4" in VID_FORMATS
        assert "avi" in VID_FORMATS
        assert "mkv" in VID_FORMATS
        assert "webm" in VID_FORMATS

    def test_stream_protocols(self):
        """Test stream protocol tuple."""
        assert "rtsp://" in STREAM_PROTOCOLS
        assert "rtmp://" in STREAM_PROTOCOLS
        assert "http://" in STREAM_PROTOCOLS
        assert "https://" in STREAM_PROTOCOLS
        assert "tcp://" in STREAM_PROTOCOLS


# =============================================================================
# Video File Tests
# =============================================================================


class TestVideoFile:
    """Test StreamLoader with local video files."""

    @pytest.fixture
    def video_loader(self):
        """Create loader for video file."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")
        loader = StreamLoader(str(VIDEO_FILE), vid_stride=1, buffer=False)
        yield loader
        loader.close()

    def test_video_file_loading(self, video_loader):
        """Test loading video file."""
        assert video_loader.bs == 1
        assert video_loader.mode == "stream"
        assert len(video_loader.fps) == 1
        assert video_loader.fps[0] > 0

    def test_video_file_iteration(self, video_loader):
        """Test iterating over video frames."""
        frame_count = 0
        for sources, images, info in video_loader:
            assert len(sources) == 1
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            assert images[0].ndim == 3  # H, W, C
            frame_count += 1
            if frame_count >= 30:  # Read 30 frames
                break

        assert frame_count == 30
        print(f"Successfully read {frame_count} frames from video file")

    def test_video_file_frame_shape(self, video_loader):
        """Test frame shape consistency."""
        sources, images, info = next(iter(video_loader))
        shape = images[0].shape
        assert len(shape) == 3
        assert shape[2] == 3  # RGB channels

    def test_video_file_context_manager(self):
        """Test context manager usage with video file."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE)) as loader:
            sources, images, info = next(iter(loader))
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)

        # After context exit, loader should be closed
        assert not loader.running


# =============================================================================
# RTSP Stream Tests
# =============================================================================


class TestRTSPStreams:
    """Test StreamLoader with RTSP streams."""

    @pytest.fixture
    def rtsp_loader(self):
        """Create loader for single RTSP stream."""
        try:
            loader = StreamLoader(RTSP_STREAMS[0], vid_stride=1, buffer=False)
            yield loader
            loader.close()
        except StreamConnectionError:
            pytest.skip(f"Cannot connect to RTSP stream: {RTSP_STREAMS[0]}")

    def test_single_rtsp_stream(self, rtsp_loader):
        """Test loading single RTSP stream."""
        assert rtsp_loader.bs == 1
        assert rtsp_loader.mode == "stream"
        assert rtsp_loader.running is True

    def test_rtsp_frame_reading(self, rtsp_loader):
        """Test reading frames from RTSP stream."""
        frame_count = 0
        for sources, images, info in rtsp_loader:
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            assert images[0].ndim == 3
            frame_count += 1
            if frame_count >= 10:
                break

        assert frame_count == 10
        print(f"Successfully read {frame_count} frames from RTSP stream")

    def test_rtsp_frame_shape(self, rtsp_loader):
        """Test RTSP frame shape."""
        sources, images, info = next(iter(rtsp_loader))
        shape = images[0].shape
        print(f"Frame shape: {shape}")
        assert len(shape) == 3
        assert shape[0] > 0  # Height
        assert shape[1] > 0  # Width
        assert shape[2] in (1, 3)  # Grayscale or RGB

    def test_multiple_rtsp_streams(self):
        """Test loading multiple RTSP streams simultaneously."""
        try:
            loader = StreamLoader(RTSP_STREAMS, vid_stride=1, buffer=False)
        except StreamConnectionError:
            pytest.skip("Cannot connect to RTSP streams")

        try:
            assert loader.bs == 2
            sources, images, info = next(iter(loader))
            assert len(images) == 2
            for img in images:
                assert isinstance(img, np.ndarray)
                assert img.ndim == 3
            print(f"Successfully loaded {loader.bs} streams")
        finally:
            loader.close()


# =============================================================================
# Buffer Mode Tests
# =============================================================================


class TestBufferModes:
    """Test different buffer modes."""

    def test_buffer_mode_fifo(self):
        """Test FIFO buffer mode."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE), buffer=True, vid_stride=1) as loader:
            # Read a few frames
            for i, (sources, images, info) in enumerate(loader):
                assert len(images) == 1
                if i >= 5:
                    break

    def test_buffer_mode_latest(self):
        """Test latest-only buffer mode (default)."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE), buffer=False, vid_stride=1) as loader:
            for i, (sources, images, info) in enumerate(loader):
                assert len(images) == 1
                if i >= 5:
                    break


# =============================================================================
# Vid Stride Tests
# =============================================================================


class TestVidStride:
    """Test video stride functionality."""

    def test_vid_stride_2(self):
        """Test processing every 2nd frame."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE), vid_stride=2) as loader:
            frames = []
            for i, (sources, images, info) in enumerate(loader):
                frames.append(images[0])
                if i >= 10:
                    break

            assert len(frames) == 11

    def test_vid_stride_5(self):
        """Test processing every 5th frame."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE), vid_stride=5) as loader:
            frames = []
            for i, (sources, images, info) in enumerate(loader):
                frames.append(images[0])
                if i >= 5:
                    break

            assert len(frames) == 6


# =============================================================================
# Grayscale Mode Tests
# =============================================================================


class TestGrayscaleMode:
    """Test grayscale frame loading."""

    def test_grayscale_video(self):
        """Test loading video in grayscale mode."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE), channels=1) as loader:
            sources, images, info = next(iter(loader))
            assert images[0].shape[2] == 1  # Single channel
            print(f"Grayscale frame shape: {images[0].shape}")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_source(self):
        """Test handling of invalid source."""
        with pytest.raises(StreamConnectionError):
            StreamLoader("rtsp://invalid.nonexistent.url:554/stream")

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        with pytest.raises(StreamConnectionError):
            StreamLoader("/nonexistent/path/video.mp4")


# =============================================================================
# Get Frame Method Tests
# =============================================================================


class TestGetFrame:
    """Test get_frame method."""

    def test_get_frame_basic(self):
        """Test basic get_frame functionality."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE)) as loader:
            frame = loader.get_frame(0, timeout=5.0)
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3

    def test_get_frame_timeout(self):
        """Test get_frame with timeout."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE)) as loader:
            # Get multiple frames
            for _ in range(5):
                frame = loader.get_frame(0, timeout=1.0)
                assert frame is not None

    def test_get_frame_invalid_index(self):
        """Test get_frame with invalid index."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        with StreamLoader(str(VIDEO_FILE)) as loader:
            with pytest.raises(IndexError):
                loader.get_frame(99)  # Invalid index


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with inference-like usage."""

    def test_inference_loop_pattern(self):
        """Test typical inference loop pattern."""
        if not VIDEO_FILE.exists():
            pytest.skip(f"Video file not found: {VIDEO_FILE}")

        processed_frames = 0

        with StreamLoader(str(VIDEO_FILE), vid_stride=1, buffer=False) as loader:
            for sources, images, info_strings in loader:
                # Simulate inference
                for img in images:
                    # Check frame is valid for inference
                    assert isinstance(img, np.ndarray)
                    assert img.dtype == np.uint8
                    assert img.shape[2] == 3  # RGB

                    # Simulate some processing
                    _ = img.mean()
                    processed_frames += 1

                if processed_frames >= 50:
                    break

        print(f"Processed {processed_frames} frames in inference loop")
        assert processed_frames == 50

    def test_multi_stream_inference(self):
        """Test inference with multiple streams."""
        try:
            loader = StreamLoader(RTSP_STREAMS, vid_stride=1, buffer=False)
        except StreamConnectionError:
            pytest.skip("Cannot connect to RTSP streams")

        try:
            processed_batches = 0
            for sources, images, info in loader:
                assert len(images) == 2  # Two streams

                # Process each stream
                for i, img in enumerate(images):
                    assert isinstance(img, np.ndarray)
                    _ = img.mean()

                processed_batches += 1
                if processed_batches >= 10:
                    break

            print(f"Processed {processed_batches} batches from {len(RTSP_STREAMS)} streams")
        finally:
            loader.close()


# =============================================================================
# Standalone Test Functions
# =============================================================================


def test_quick_video():
    """Quick test for video file loading."""
    if not VIDEO_FILE.exists():
        print(f"SKIP: Video file not found: {VIDEO_FILE}")
        return

    print(f"Testing video file: {VIDEO_FILE}")
    loader = StreamLoader(str(VIDEO_FILE), vid_stride=1, buffer=False)

    try:
        frame_count = 0
        for sources, images, info in loader:
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            frame_count += 1
            if frame_count >= 30:
                break
        print(f"SUCCESS: Read {frame_count} frames from video file")
        print(f"  Frame shape: {images[0].shape}")
        print(f"  FPS: {loader.fps[0]}")
    finally:
        loader.close()


def test_quick_rtsp():
    """Quick test for RTSP stream loading."""
    print(f"Testing RTSP stream: {RTSP_STREAMS[0]}")

    try:
        loader = StreamLoader(RTSP_STREAMS[0], vid_stride=1, buffer=False)
    except StreamConnectionError as e:
        print(f"SKIP: Cannot connect to RTSP stream: {e}")
        return

    try:
        frame_count = 0
        for sources, images, info in loader:
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            frame_count += 1
            print(f"  Frame {frame_count}: shape={images[0].shape}")
            if frame_count >= 1000:
                break
        print(f"SUCCESS: Read {frame_count} frames from RTSP stream")
    finally:
        loader.close()


def test_quick_multi_rtsp():
    """Quick test for multiple RTSP streams."""
    print(f"Testing multiple RTSP streams: {RTSP_STREAMS}")

    try:
        loader = StreamLoader(RTSP_STREAMS, vid_stride=1, buffer=False)
    except StreamConnectionError as e:
        print(f"SKIP: Cannot connect to RTSP streams: {e}")
        return

    try:
        frame_count = 0
        for sources, images, info in loader:
            assert len(images) == 2
            for i, img in enumerate(images):
                assert isinstance(img, np.ndarray)
            frame_count += 1
            if frame_count >= 5:
                break
        print(f"SUCCESS: Read {frame_count} batches from {len(RTSP_STREAMS)} streams")
        for i, shape in enumerate(loader.shape):
            print(f"  Stream {i}: shape={shape}, fps={loader.fps[i]}")
    finally:
        loader.close()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Standalone Stream Module Tests")
    print("=" * 60)

    # print("\n--- Video File Test ---")
    # test_quick_video()

    print("\n--- Single RTSP Stream Test ---")
    test_quick_rtsp()

    # print("\n--- Multiple RTSP Streams Test ---")
    # test_quick_multi_rtsp()

    print("\n" + "=" * 60)
    print("All standalone tests completed!")
    print("=" * 60)
