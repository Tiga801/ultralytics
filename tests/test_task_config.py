"""Tests for TaskConfig and TaskConfigManager.

This module tests task configuration parsing, validation, and management.
"""

import pytest
import sys

from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task.config import TaskConfig, TaskConfigManager


class TestTaskConfig:
    """Test cases for TaskConfig dataclass."""

    def test_basic_config_creation(self):
        """Test creating a basic task config."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test Task",
            task_type="cross_line",
            rtsp_url="rtsp://localhost:554/stream",
        )
        assert config.task_id == "task_001"
        assert config.task_name == "Test Task"
        assert config.task_type == "cross_line"
        assert config.rtsp_url == "rtsp://localhost:554/stream"

    def test_default_values(self):
        """Test default configuration values."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test",
            task_type="test",
        )
        assert config.fps == 25
        assert config.stand_name == "default"
        assert config.camera_id == ""
        assert config.areas_info == []

    def test_resolution_parsing(self):
        """Test resolution string parsing."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test",
            task_type="test",
            resolution="1920x1080",
        )
        assert config.get_resolution_width() == 1920
        assert config.get_resolution_height() == 1080

    def test_resolution_invalid(self):
        """Test invalid resolution handling."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test",
            task_type="test",
            resolution="invalid",
        )
        assert config.get_resolution_width() == 0
        assert config.get_resolution_height() == 0

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test Task",
            task_type="cross_line",
            rtsp_url="rtsp://localhost/stream",
        )
        d = config.to_dict()
        assert d["task_id"] == "task_001"
        assert d["task_name"] == "Test Task"
        assert d["task_type"] == "cross_line"

    def test_get_extra(self):
        """Test getting extra configuration values."""
        config = TaskConfig(
            task_id="task_001",
            task_name="Test",
            task_type="test",
            extra={"custom_key": "custom_value", "threshold": 0.5},
        )
        assert config.get_extra("custom_key") == "custom_value"
        assert config.get_extra("threshold") == 0.5
        assert config.get_extra("missing") is None
        assert config.get_extra("missing", "default") == "default"


class TestTaskConfigManager:
    """Test cases for TaskConfigManager singleton."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset manager state before each test."""
        manager = TaskConfigManager()
        manager.clear_all()
        yield
        manager.clear_all()

    def test_singleton_pattern(self):
        """Test TaskConfigManager is a singleton."""
        manager1 = TaskConfigManager()
        manager2 = TaskConfigManager()
        assert manager1 is manager2

    def test_add_config(self):
        """Test adding a configuration."""
        manager = TaskConfigManager()
        config = TaskConfig(
            task_id="task_001",
            task_name="Test",
            task_type="test",
        )
        manager.add_config(config)
        assert manager.get_config("task_001") is config

    def test_get_config_not_found(self):
        """Test getting non-existent configuration."""
        manager = TaskConfigManager()
        assert manager.get_config("nonexistent") is None

    def test_remove_config(self):
        """Test removing a configuration."""
        manager = TaskConfigManager()
        config = TaskConfig(task_id="task_001", task_name="Test", task_type="test")
        manager.add_config(config)
        result = manager.remove_config("task_001")
        assert result is True
        assert manager.get_config("task_001") is None

    def test_remove_nonexistent_config(self):
        """Test removing non-existent configuration."""
        manager = TaskConfigManager()
        result = manager.remove_config("nonexistent")
        assert result is False

    def test_build_config_from_dict(self):
        """Test building config from Algorithm Warehouse format."""
        manager = TaskConfigManager()
        data = {
            "taskID": "task_002",
            "taskName": "Built Task",
            "taskType": "region_intrusion",
            "rtspUrl": "rtsp://camera:554/stream",
            "standName": "stand_001",
            "cameraId": "cam_001",
            "cameraName": "Camera 1",
            "fps": 30,
            "resolution": "1280x720",
            "areasInfo": [
                {
                    "id": "area_1",
                    "points": [
                        {"x": 100, "y": 100},
                        {"x": 200, "y": 100},
                        {"x": 200, "y": 200},
                        {"x": 100, "y": 200},
                    ]
                }
            ],
        }
        config = manager.build_config(data)
        assert config.task_id == "task_002"
        assert config.task_name == "Built Task"
        assert config.task_type == "region_intrusion"
        assert config.rtsp_url == "rtsp://camera:554/stream"
        assert config.stand_name == "stand_001"
        assert config.camera_id == "cam_001"
        assert config.fps == 30
        assert len(config.areas_info) == 1

    def test_build_config_auto_adds(self):
        """Test build_config automatically adds to manager."""
        manager = TaskConfigManager()
        data = {
            "taskID": "task_003",
            "taskName": "Auto Added",
            "taskType": "test",
        }
        config = manager.build_config(data)
        retrieved = manager.get_config("task_003")
        assert retrieved is config

    def test_get_all_task_ids(self):
        """Test getting all task IDs."""
        manager = TaskConfigManager()
        for i in range(3):
            config = TaskConfig(
                task_id=f"task_{i}",
                task_name=f"Task {i}",
                task_type="test",
            )
            manager.add_config(config)
        ids = manager.get_all_task_ids()
        assert len(ids) == 3
        assert "task_0" in ids
        assert "task_1" in ids
        assert "task_2" in ids

    def test_count(self):
        """Test counting configurations."""
        manager = TaskConfigManager()
        assert manager.count() == 0
        for i in range(5):
            config = TaskConfig(
                task_id=f"task_{i}",
                task_name=f"Task {i}",
                task_type="test",
            )
            manager.add_config(config)
        assert manager.count() == 5

    def test_clear_all(self):
        """Test clearing all configurations."""
        manager = TaskConfigManager()
        for i in range(3):
            config = TaskConfig(
                task_id=f"task_{i}",
                task_name=f"Task {i}",
                task_type="test",
            )
            manager.add_config(config)
        manager.clear_all()
        assert manager.count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
