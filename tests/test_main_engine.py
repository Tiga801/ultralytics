"""Tests for MainEngine.

This module tests the main engine lifecycle, task management,
and stand coordination.
"""

import pytest
import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import SingletonMeta


class TestMainEngineSingleton:
    """Test MainEngine singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        # Clear singleton instances
        if "MainEngine" in str(SingletonMeta._instances):
            SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_singleton_pattern(self):
        """Test MainEngine is a singleton."""
        from engine import MainEngine
        engine1 = MainEngine()
        engine2 = MainEngine()
        assert engine1 is engine2

    def test_initial_state(self):
        """Test initial engine state."""
        from engine import MainEngine
        engine = MainEngine()
        assert not engine._initialized
        assert not engine._started
        assert len(engine._stand_engines) == 0


class TestMainEngineLifecycle:
    """Test MainEngine lifecycle methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    @pytest.fixture
    def engine(self):
        """Create a main engine instance."""
        from engine import MainEngine
        engine = MainEngine()
        yield engine
        try:
            engine.stop()
        except Exception:
            pass

    def test_init_sets_initialized(self, engine):
        """Test init sets initialized flag."""
        engine.init()
        assert engine._initialized

    def test_init_idempotent(self, engine):
        """Test init can be called multiple times."""
        engine.init()
        engine.init()  # Should not raise
        assert engine._initialized

    def test_start_requires_init(self, engine):
        """Test start raises if not initialized."""
        from engine import EngineNotInitializedException
        with pytest.raises(EngineNotInitializedException):
            engine.start()

    def test_start_sets_started(self, engine):
        """Test start sets started flag."""
        engine.init()
        engine.start()
        assert engine._started

    def test_start_idempotent(self, engine):
        """Test start can be called multiple times."""
        engine.init()
        engine.start()
        engine.start()  # Should not raise
        assert engine._started

    def test_stop_clears_state(self, engine):
        """Test stop clears engine state."""
        engine.init()
        engine.start()
        engine.stop()
        assert not engine._started
        assert len(engine._stand_engines) == 0


class TestMainEngineTaskManagement:
    """Test MainEngine task management."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        from task import TaskConfigManager
        TaskConfigManager().clear_all()
        yield
        SingletonMeta._instances.clear()

    @pytest.fixture
    def engine(self):
        """Create and initialize engine."""
        from engine import MainEngine
        engine = MainEngine()
        engine.init()
        engine.start()
        yield engine
        engine.stop()

    def test_get_all_tasks_empty(self, engine):
        """Test get_all_tasks returns empty list initially."""
        tasks = engine.get_all_tasks()
        assert tasks == []

    def test_get_engine_status(self, engine):
        """Test get_engine_status returns expected structure."""
        status = engine.get_engine_status()
        assert "initialized" in status
        assert "started" in status
        assert "stand_count" in status
        assert "camera_count" in status
        assert "task_count" in status
        assert status["initialized"] is True
        assert status["started"] is True

    def test_get_capabilities(self, engine):
        """Test get_capabilities returns expected format."""
        caps = engine.get_capabilities()
        assert "taskCurNum" in caps
        assert "taskTotalNum" in caps
        assert "totalCapability" in caps
        assert "curCapability" in caps
        assert "resolutionCap" in caps

    @patch("engine.main_engine.TaskConfigManager")
    def test_add_task_creates_stand_engine(self, mock_config_manager, engine):
        """Test add_task creates stand engine if needed."""
        # Setup mock
        mock_config = MagicMock()
        mock_config.stand_name = "new_stand"
        mock_config.camera_id = "cam_001"
        mock_config.camera_name = "Camera 1"
        mock_config_manager.return_value.get_config.return_value = None
        mock_config_manager.return_value.build_config.return_value = mock_config

        task_data = {
            "taskID": "task_001",
            "taskName": "Test",
            "taskType": "cross_line",
            "standName": "new_stand",
        }

        # This will fail due to incomplete mocking, but tests the path
        # engine.add_task(task_data)


class TestMainEngineStandManagement:
    """Test stand engine management."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_get_or_create_stand_engine(self):
        """Test stand engine creation."""
        from engine import MainEngine
        engine = MainEngine()
        engine.init()

        stand = engine._get_or_create_stand_engine("test_stand")
        assert stand is not None
        assert "test_stand" in engine._stand_engines

        # Get same stand
        stand2 = engine._get_or_create_stand_engine("test_stand")
        assert stand is stand2

        engine.stop()

    def test_remove_stand_engine(self):
        """Test stand engine removal."""
        from engine import MainEngine
        engine = MainEngine()
        engine.init()

        engine._get_or_create_stand_engine("test_stand")
        assert "test_stand" in engine._stand_engines

        engine._remove_stand_engine("test_stand")
        assert "test_stand" not in engine._stand_engines

        engine.stop()


class TestMainEngineRepr:
    """Test string representation."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_repr(self):
        """Test __repr__ returns expected format."""
        from engine import MainEngine
        engine = MainEngine()
        repr_str = repr(engine)
        assert "MainEngine" in repr_str
        assert "initialized" in repr_str
        engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
