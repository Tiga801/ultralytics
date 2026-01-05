"""Task management service.

This module provides the TaskService class that handles task
lifecycle operations through the MainEngine.
"""

from typing import Any, Dict, List, Optional

from utils import Logger
from engine import MainEngine


class TaskService:
    """Service for task lifecycle management.

    This service provides a high-level interface for task operations,
    delegating to the MainEngine singleton.
    """

    def __init__(self):
        """Initialize task service."""
        self._engine = MainEngine()
        self.log = Logger.get_logging_method("TASK-SERVICE")

    def run_tasks(self, task_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Start multiple tasks.

        Args:
            task_configs: List of task configurations.

        Returns:
            Result dictionary with success/failure counts.
        """
        results = {
            "total": len(task_configs),
            "success": 0,
            "failed": 0,
            "task_ids": [],
            "errors": [],
        }

        for config in task_configs:
            task_id = config.get("taskID", "")
            try:
                success = self._engine.add_task(config)
                if success:
                    results["success"] += 1
                    results["task_ids"].append(task_id)
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "taskId": task_id,
                        "error": "Failed to add task"
                    })
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "taskId": task_id,
                    "error": str(e)
                })
                self.log(f"Error adding task {task_id}: {e}")

        return results

    def delete_tasks(self, task_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple tasks.

        Args:
            task_ids: List of task IDs to delete.

        Returns:
            Result dictionary with success/failure counts.
        """
        results = {
            "total": len(task_ids),
            "success": 0,
            "failed": 0,
            "deleted_ids": [],
            "errors": [],
        }

        for task_id in task_ids:
            try:
                success = self._engine.remove_task(task_id)
                if success:
                    results["success"] += 1
                    results["deleted_ids"].append(task_id)
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "taskId": task_id,
                        "error": "Task not found"
                    })
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "taskId": task_id,
                    "error": str(e)
                })
                self.log(f"Error deleting task {task_id}: {e}")

        return results

    def pause_task(self, task_id: str) -> bool:
        """Pause a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if paused successfully.
        """
        try:
            return self._engine.pause_task(task_id)
        except Exception as e:
            self.log(f"Error pausing task {task_id}: {e}")
            return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if resumed successfully.
        """
        try:
            return self._engine.resume_task(task_id)
        except Exception as e:
            self.log(f"Error resuming task {task_id}: {e}")
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a single task.

        Args:
            task_id: Task identifier.

        Returns:
            Task status dictionary or None.
        """
        return self._engine.get_task_status(task_id)

    def get_all_task_status(
        self,
        task_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get status of multiple or all tasks.

        Args:
            task_ids: Optional list of task IDs to query.
                     If None, returns all tasks.

        Returns:
            List of task status dictionaries.
        """
        all_tasks = self._engine.get_all_tasks()

        if task_ids is None:
            return all_tasks

        # Filter by requested task IDs
        return [t for t in all_tasks if t.get("taskId") in task_ids]

    def get_task_count(self) -> int:
        """Get total number of active tasks.

        Returns:
            Number of tasks.
        """
        status = self._engine.get_engine_status()
        return status.get("task_count", 0)

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is running.

        Args:
            task_id: Task identifier.

        Returns:
            True if task is running.
        """
        status = self.get_task_status(task_id)
        return status is not None and status.get("isRunning", False)
