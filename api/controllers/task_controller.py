"""Task lifecycle controller.

This module provides the TaskController class that handles
task-related API requests.
"""

import json
from typing import Any, Dict

from utils import Logger
from ..models import (
    ApiResponse,
    TaskRunRequest,
    TaskDeleteRequest,
    TaskPauseRequest,
    TaskResumeRequest,
    TaskStatusRequest,
)
from ..services import TaskService


# Module-level request logger (logs to requests.log and main service log)
_request_log = Logger.get_request_logger()


class TaskController:
    """Controller for task lifecycle operations.

    This controller handles API requests for task management:
    - Run/start tasks
    - Delete/stop tasks
    - Pause tasks
    - Resume tasks
    - Query task status
    """

    def __init__(self):
        """Initialize task controller."""
        self._service = TaskService()
        self.log = Logger.get_logging_method("TASK-CTL")

    def run_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run task request.

        Args:
            request_data: Request body data.

        Returns:
            API response dictionary.
        """
        try:
            request = TaskRunRequest.from_dict(request_data)

            # Log the request
            task_ids = [t.get("taskID", "unknown") for t in request.tasks] if request.tasks else []
            _request_log(f"START_TASK: {task_ids}")
            _request_log(f"REQUEST_DATA:\n{json.dumps(request_data, indent=2, ensure_ascii=False)}")

            if not request.tasks:
                return ApiResponse.error(
                    code=400,
                    message="No tasks provided"
                ).to_dict()

            result = self._service.run_tasks(request.tasks)

            if result["failed"] == 0:
                return ApiResponse.success(
                    data={
                        "taskIds": result["task_ids"],
                        "count": result["success"],
                    },
                    message=f"Successfully started {result['success']} task(s)"
                ).to_dict()
            else:
                return ApiResponse.error(
                    code=207,  # Multi-status
                    message=f"Started {result['success']}, failed {result['failed']}",
                    data={
                        "taskIds": result["task_ids"],
                        "errors": result["errors"],
                    }
                ).to_dict()

        except Exception as e:
            self.log(f"Error in run_task: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Internal error: {str(e)}"
            ).to_dict()

    def delete_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete task request.

        Args:
            request_data: Request body data.

        Returns:
            API response dictionary.
        """
        try:
            request = TaskDeleteRequest.from_dict(request_data)

            # Log the request
            _request_log(f"DELETE_TASK: {request.task_ids}")

            if not request.task_ids:
                return ApiResponse.error(
                    code=400,
                    message="No task IDs provided"
                ).to_dict()

            result = self._service.delete_tasks(request.task_ids)

            if result["failed"] == 0:
                return ApiResponse.success(
                    data={
                        "deletedIds": result["deleted_ids"],
                        "count": result["success"],
                    },
                    message=f"Successfully deleted {result['success']} task(s)"
                ).to_dict()
            else:
                return ApiResponse.error(
                    code=207,
                    message=f"Deleted {result['success']}, failed {result['failed']}",
                    data={
                        "deletedIds": result["deleted_ids"],
                        "errors": result["errors"],
                    }
                ).to_dict()

        except Exception as e:
            self.log(f"Error in delete_task: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Internal error: {str(e)}"
            ).to_dict()

    def pause_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pause task request.

        Args:
            request_data: Request body data.

        Returns:
            API response dictionary.
        """
        try:
            request = TaskPauseRequest.from_dict(request_data)

            # Log the request
            _request_log(f"PAUSE_TASK: {request.task_id}")

            if not request.task_id:
                return ApiResponse.error(
                    code=400,
                    message="Task ID required"
                ).to_dict()

            success = self._service.pause_task(request.task_id)

            if success:
                return ApiResponse.success(
                    message=f"Task {request.task_id} paused"
                ).to_dict()
            else:
                return ApiResponse.error(
                    code=404,
                    message=f"Failed to pause task {request.task_id}"
                ).to_dict()

        except Exception as e:
            self.log(f"Error in pause_task: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Internal error: {str(e)}"
            ).to_dict()

    def resume_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resume task request.

        Args:
            request_data: Request body data.

        Returns:
            API response dictionary.
        """
        try:
            request = TaskResumeRequest.from_dict(request_data)

            # Log the request
            _request_log(f"RESUME_TASK: {request.task_id}")

            if not request.task_id:
                return ApiResponse.error(
                    code=400,
                    message="Task ID required"
                ).to_dict()

            success = self._service.resume_task(request.task_id)

            if success:
                return ApiResponse.success(
                    message=f"Task {request.task_id} resumed"
                ).to_dict()
            else:
                return ApiResponse.error(
                    code=404,
                    message=f"Failed to resume task {request.task_id}"
                ).to_dict()

        except Exception as e:
            self.log(f"Error in resume_task: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Internal error: {str(e)}"
            ).to_dict()

    def get_task_status(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task status query request.

        Args:
            request_data: Request body data.

        Returns:
            API response dictionary.
        """
        try:
            request = TaskStatusRequest.from_dict(request_data)

            tasks = self._service.get_all_task_status(request.task_ids)

            return ApiResponse.success(
                data={
                    "tasks": tasks,
                    "count": len(tasks),
                }
            ).to_dict()

        except Exception as e:
            self.log(f"Error in get_task_status: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Internal error: {str(e)}"
            ).to_dict()
