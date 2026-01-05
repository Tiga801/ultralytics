"""Tests for API endpoints.

This module tests the Flask API endpoints for task management.
"""

import json
import pytest
import sys

from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import SingletonMeta


class TestApiSetup:
    """Test API application setup."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_create_app(self):
        """Test Flask app creation."""
        from api import create_app
        app = create_app()
        assert app is not None

    def test_app_has_routes(self):
        """Test app has expected routes."""
        from api import create_app
        app = create_app()
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/IAP/Status" in rules
        assert "/IAP/runTask" in rules
        assert "/IAP/deleteTask" in rules


class TestHealthEndpoints:
    """Test health and status endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api import create_app
        app = create_app({"TESTING": True})
        with app.test_client() as client:
            yield client

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    def test_root_endpoint(self, client):
        """Test root endpoint returns ok."""
        response = client.get("/")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    @patch("api.controllers.status_controller.ResourceService")
    def test_status_endpoint(self, mock_service, client):
        """Test /IAP/Status endpoint."""
        mock_service.return_value.get_health_status.return_value = {
            "status": "healthy",
            "initialized": True,
            "started": True,
        }
        response = client.get("/IAP/Status")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.status_controller.ResourceService")
    def test_capabilities_endpoint(self, mock_service, client):
        """Test /IAP/capabilities endpoint."""
        mock_service.return_value.get_capabilities.return_value = {
            "taskCurNum": 0,
            "taskTotalNum": 10,
            "totalCapability": 100,
            "curCapability": 100,
        }
        response = client.get("/IAP/capabilities")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0


class TestTaskEndpoints:
    """Test task management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api import create_app
        app = create_app({"TESTING": True})
        with app.test_client() as client:
            yield client

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset state before each test."""
        SingletonMeta._instances.clear()
        yield
        SingletonMeta._instances.clear()

    @patch("api.controllers.task_controller.TaskService")
    def test_run_task_empty_request(self, mock_service, client):
        """Test run task with empty request."""
        response = client.post(
            "/IAP/runTask",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 400  # Bad request - no tasks

    @patch("api.controllers.task_controller.TaskService")
    def test_run_task_success(self, mock_service, client):
        """Test successful run task."""
        mock_service.return_value.run_tasks.return_value = {
            "total": 1,
            "success": 1,
            "failed": 0,
            "task_ids": ["task_001"],
            "errors": [],
        }

        response = client.post(
            "/IAP/runTask",
            data=json.dumps({
                "analyseConditions": [
                    {
                        "taskID": "task_001",
                        "taskName": "Test",
                        "taskType": "cross_line",
                    }
                ]
            }),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.task_controller.TaskService")
    def test_delete_task_empty_request(self, mock_service, client):
        """Test delete task with empty request."""
        response = client.post(
            "/IAP/deleteTask",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 400

    @patch("api.controllers.task_controller.TaskService")
    def test_delete_task_success(self, mock_service, client):
        """Test successful delete task."""
        mock_service.return_value.delete_tasks.return_value = {
            "total": 1,
            "success": 1,
            "failed": 0,
            "deleted_ids": ["task_001"],
            "errors": [],
        }

        response = client.post(
            "/IAP/deleteTask",
            data=json.dumps({"taskIds": ["task_001"]}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.task_controller.TaskService")
    def test_pause_task_no_id(self, mock_service, client):
        """Test pause task without ID."""
        response = client.post(
            "/IAP/pauseTask",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 400

    @patch("api.controllers.task_controller.TaskService")
    def test_pause_task_success(self, mock_service, client):
        """Test successful pause task."""
        mock_service.return_value.pause_task.return_value = True

        response = client.post(
            "/IAP/pauseTask",
            data=json.dumps({"taskId": "task_001"}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.task_controller.TaskService")
    def test_resume_task_success(self, mock_service, client):
        """Test successful resume task."""
        mock_service.return_value.resume_task.return_value = True

        response = client.post(
            "/IAP/resumeTask",
            data=json.dumps({"taskId": "task_001"}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.task_controller.TaskService")
    def test_task_status_get(self, mock_service, client):
        """Test get all task status."""
        mock_service.return_value.get_all_task_status.return_value = []

        response = client.get("/IAP/taskStatus")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0

    @patch("api.controllers.task_controller.TaskService")
    def test_task_status_post(self, mock_service, client):
        """Test get specific task status."""
        mock_service.return_value.get_all_task_status.return_value = [
            {"taskId": "task_001", "isRunning": True}
        ]

        response = client.post(
            "/IAP/taskStatus",
            data=json.dumps({"taskIds": ["task_001"]}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["code"] == 0


class TestApiModels:
    """Test API request/response models."""

    def test_api_response_success(self):
        """Test ApiResponse.success factory."""
        from api.models import ApiResponse
        response = ApiResponse.success(data={"test": "data"})
        assert response.code == 0
        assert response.message == "success"
        assert response.data == {"test": "data"}

    def test_api_response_error(self):
        """Test ApiResponse.error factory."""
        from api.models import ApiResponse
        response = ApiResponse.error(code=500, message="Error")
        assert response.code == 500
        assert response.message == "Error"

    def test_task_run_request_from_dict(self):
        """Test TaskRunRequest.from_dict."""
        from api.models import TaskRunRequest
        data = {
            "analyseConditions": [
                {"taskID": "task_001"}
            ]
        }
        request = TaskRunRequest.from_dict(data)
        assert len(request.tasks) == 1
        assert request.tasks[0]["taskID"] == "task_001"

    def test_task_delete_request_from_dict(self):
        """Test TaskDeleteRequest.from_dict."""
        from api.models import TaskDeleteRequest
        data = {"taskIds": ["task_001", "task_002"]}
        request = TaskDeleteRequest.from_dict(data)
        assert len(request.task_ids) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
