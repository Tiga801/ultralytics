"""Tests for CameraEngine.

This module tests the camera engine including stream management
and task lifecycle.
"""

import queue
import time
import pytest
import numpy as np
import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task import TaskConfigManager


class TestCameraEngine:
    """Test cases for CameraEngine."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        TaskConfigManager().clear_all()
        yield
        TaskConfigManager().clear_all()

    @pytest.fixture
    def camera_engine(self):
        """Create a test camera engine."""
        from engine.camera_engine import CameraEngine
        engine = CameraEngine(
            camera_id="cam_001",
            camera_name="Test Camera",
            stand_name="test_stand"
        )
        yield engine
        engine.stop()

    def test_camera_engine_initialization(self, camera_engine):
        """Test camera engine initializes correctly."""
        assert camera_engine.camera_id == "cam_001"
        assert camera_engine.camera_name == "Test Camera"
        assert camera_engine.stand_name == "test_stand"
        assert not camera_engine.is_stream_running()
        assert camera_engine.get_task_count() == 0

    def test_get_status(self, camera_engine):
        """Test getting camera engine status."""
        status = camera_engine.get_status()
        assert status["camera_id"] == "cam_001"
        assert status["camera_name"] == "Test Camera"
        assert status["stream_running"] is False
        assert status["task_count"] == 0

    def test_has_tasks_initially_false(self, camera_engine):
        """Test has_tasks returns False initially."""
        assert not camera_engine.has_tasks()

    def test_frame_queue_initialization(self, camera_engine):
        """Test frame queue is initialized."""
        assert camera_engine._frame_queue is not None
        assert isinstance(camera_engine._frame_queue, queue.Queue)

    def test_stream_not_started_without_tasks(self, camera_engine):
        """Test stream doesn't start without tasks."""
        camera_engine.start()
        time.sleep(0.1)
        assert not camera_engine.is_stream_running()

    @patch("engine.camera_engine.TaskConfigManager")
    def test_add_task_updates_stream_info(self, mock_config_manager, camera_engine):
        """Test adding task updates stream info."""
        # Setup mock config
        mock_config = MagicMock()
        mock_config.rtsp_url = "rtsp://test:554/stream"
        mock_config.get_resolution_width.return_value = 1920
        mock_config.get_resolution_height.return_value = 1080
        mock_config.fps = 25
        mock_config_manager.return_value.get_config.return_value = mock_config

        camera_engine._update_stream_info("task_001")

        assert camera_engine._rtsp_url == "rtsp://test:554/stream"
        assert camera_engine.camera_width == 1920
        assert camera_engine.camera_height == 1080
        assert camera_engine.fps == 25


class TestCameraEngineStreamLifecycle:
    """Test stream lifecycle management."""

    @pytest.fixture
    def mock_stream_loader(self):
        """Create a mock stream loader."""
        with patch("engine.camera_engine.StreamLoader") as mock:
            loader_instance = MagicMock()
            loader_instance.__iter__ = MagicMock(return_value=iter([]))
            mock.return_value = loader_instance
            yield mock

    def test_stream_starts_on_first_task(self, mock_stream_loader):
        """Test stream starts when first task is added."""
        # This would require more setup with mocked task manager
        pass

    def test_stream_stops_on_last_task_removal(self, mock_stream_loader):
        """Test stream stops when last task is removed."""
        # This would require more setup with mocked task manager
        pass


class TestFrameDistribution:
    """Test frame distribution to tasks."""

    def test_frame_queue_put_and_get(self):
        """Test frames can be queued and retrieved."""
        from engine.camera_engine import CameraEngine
        engine = CameraEngine("cam_001", "Test", "stand")

        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = time.time()

        # Put frame in queue
        engine._frame_queue.put(("Test", timestamp, frame))

        # Get frame from queue
        result = engine._frame_queue.get(timeout=1.0)
        camera_name, ts, f = result

        assert camera_name == "Test"
        assert ts == timestamp
        assert np.array_equal(f, frame)

        engine.stop()

    def test_frame_queue_overflow_handling(self):
        """Test queue handles overflow gracefully."""
        from engine.camera_engine import CameraEngine
        engine = CameraEngine("cam_001", "Test", "stand")

        # Fill queue beyond capacity
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(engine.QUEUE_MAXSIZE + 10):
            try:
                engine._frame_queue.put_nowait(("Test", time.time(), frame))
            except queue.Full:
                # Expected for some puts
                pass

        # Queue should be at max size or less
        assert engine._frame_queue.qsize() <= engine.QUEUE_MAXSIZE

        engine.stop()


class TestCameraEngineRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ returns expected format."""
        from engine.camera_engine import CameraEngine
        engine = CameraEngine("cam_001", "Test", "stand")
        repr_str = repr(engine)
        assert "CameraEngine" in repr_str
        assert "cam_001" in repr_str
        engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
