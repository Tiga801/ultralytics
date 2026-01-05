"""Integration tests for the task management system.

This module provides end-to-end integration tests that verify
the complete pipeline from API to task execution.
"""

import json
import pytest
import sys

from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import SingletonMeta


class TestFullPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset all singleton state."""
        SingletonMeta._instances.clear()
        from task import TaskConfigManager
        TaskConfigManager().clear_all()
        yield
        SingletonMeta._instances.clear()

    @pytest.fixture
    def initialized_engine(self):
        """Create and initialize the main engine."""
        from engine import MainEngine, init_engine_config
        init_engine_config()
        engine = MainEngine()
        engine.init()
        engine.start()
        yield engine
        engine.stop()

    def test_engine_lifecycle(self, initialized_engine):
        """Test complete engine lifecycle."""
        engine = initialized_engine

        # Verify initialized state
        status = engine.get_engine_status()
        assert status["initialized"] is True
        assert status["started"] is True
        assert status["task_count"] == 0

    def test_stand_engine_creation_and_removal(self, initialized_engine):
        """Test stand engine management."""
        engine = initialized_engine

        # Create stand engine
        stand = engine._get_or_create_stand_engine("test_stand")
        assert stand is not None
        assert "test_stand" in engine._stand_engines

        # Verify status reflects stand
        status = engine.get_engine_status()
        assert status["stand_count"] == 1

        # Remove stand engine
        engine._remove_stand_engine("test_stand")
        status = engine.get_engine_status()
        assert status["stand_count"] == 0


class TestApiIntegration:
    """Integration tests for API with engine."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        from task import TaskConfigManager
        TaskConfigManager().clear_all()
        yield
        SingletonMeta._instances.clear()

    @pytest.fixture
    def app_with_engine(self):
        """Create app with initialized engine."""
        from engine import MainEngine, init_engine_config
        from api import create_app

        init_engine_config()
        engine = MainEngine()
        engine.init()
        engine.start()

        app = create_app({"TESTING": True})
        yield app, engine

        engine.stop()

    def test_status_reflects_engine_state(self, app_with_engine):
        """Test status endpoint reflects engine state."""
        app, engine = app_with_engine

        with app.test_client() as client:
            response = client.get("/IAP/engineStatus")
            data = json.loads(response.data)

            assert data["code"] == 0
            assert data["data"]["initialized"] is True
            assert data["data"]["started"] is True

    def test_capabilities_endpoint(self, app_with_engine):
        """Test capabilities reflect current state."""
        app, engine = app_with_engine

        with app.test_client() as client:
            response = client.get("/IAP/capabilities")
            data = json.loads(response.data)

            assert data["code"] == 0
            assert "taskCurNum" in data["data"]
            assert "taskTotalNum" in data["data"]
            assert data["data"]["taskCurNum"] == 0


class TestTaskConfigIntegration:
    """Integration tests for task configuration."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        from task import TaskConfigManager
        TaskConfigManager().clear_all()
        yield
        TaskConfigManager().clear_all()

    def test_config_manager_build_and_retrieve(self):
        """Test building and retrieving configs."""
        from task import TaskConfigManager

        manager = TaskConfigManager()

        # Build config from Algorithm Warehouse format
        data = {
            "taskID": "integration_task_001",
            "taskName": "Integration Test Task",
            "taskType": "cross_line",
            "rtspUrl": "rtsp://test:554/stream",
            "standName": "test_stand",
            "cameraId": "cam_001",
            "cameraName": "Test Camera",
            "fps": 25,
            "resolution": "1920x1080",
            "areasInfo": [
                {
                    "id": "line_1",
                    "line": [
                        {"x": 0, "y": 540},
                        {"x": 1920, "y": 540}
                    ]
                }
            ]
        }

        config = manager.build_config(data)

        # Verify config was created correctly
        assert config.task_id == "integration_task_001"
        assert config.task_name == "Integration Test Task"
        assert config.task_type == "cross_line"
        assert config.stand_name == "test_stand"
        assert config.get_resolution_width() == 1920
        assert config.get_resolution_height() == 1080

        # Verify it's in the manager
        retrieved = manager.get_config("integration_task_001")
        assert retrieved is config


class TestTaskRegistryIntegration:
    """Integration tests for task registry."""

    def test_solutions_registered(self):
        """Test solution tasks are registered with registry."""
        from task import TaskRegistry

        # Import solutions to trigger registration
        import solutions

        # Verify tasks are registered
        assert TaskRegistry.is_registered("cross_line")
        assert TaskRegistry.is_registered("region_intrusion")
        assert TaskRegistry.is_registered("face_detection")
        assert TaskRegistry.is_registered("crowd_density")

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        from task import TaskRegistry
        import solutions

        types = TaskRegistry.get_supported_types()
        assert "cross_line" in types
        assert "region_intrusion" in types


class TestThreadSafety:
    """Integration tests for thread safety."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        from task import TaskConfigManager
        TaskConfigManager().clear_all()
        yield
        SingletonMeta._instances.clear()

    def test_concurrent_config_access(self):
        """Test concurrent access to config manager."""
        import threading
        from task import TaskConfigManager, TaskConfig

        manager = TaskConfigManager()
        errors = []

        def add_configs(thread_id):
            try:
                for i in range(50):
                    config = TaskConfig(
                        task_id=f"task_{thread_id}_{i}",
                        task_name=f"Task {thread_id}-{i}",
                        task_type="test",
                    )
                    manager.add_config(config)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_configs, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have 250 configs (5 threads * 50 configs each)
        assert manager.count() == 250

    def test_concurrent_engine_status(self):
        """Test concurrent engine status queries."""
        import threading
        from engine import MainEngine, init_engine_config

        init_engine_config()
        engine = MainEngine()
        engine.init()
        engine.start()

        errors = []
        results = []

        def query_status():
            try:
                for _ in range(100):
                    status = engine.get_engine_status()
                    results.append(status)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=query_status)
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        engine.stop()

        assert len(errors) == 0
        assert len(results) == 500


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_engine_not_initialized_error(self):
        """Test error when starting uninitialized engine."""
        from engine import MainEngine, EngineNotInitializedException

        engine = MainEngine()
        with pytest.raises(EngineNotInitializedException):
            engine.start()

    def test_invalid_task_type_error(self):
        """Test error with invalid task type."""
        from task import TaskRegistry

        with pytest.raises(ValueError):
            TaskRegistry.create("nonexistent_type", "task_001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
