"""Task manager for per-camera task lifecycle management.

This module provides the TaskManager class that manages task instances
within a CameraEngine.
"""

import threading
from typing import Dict, List, Optional, TYPE_CHECKING

from utils import Logger
from .config import TaskConfig, TaskConfigManager
from .registry import TaskRegistry

if TYPE_CHECKING:
    from .base import TaskBase


class TaskManager:
    """Per-camera task container and lifecycle manager.

    This class manages all task instances for a single camera,
    handling creation, deletion, and lifecycle operations.

    Each TaskManager is owned by a CameraEngine and coordinates
    task processes running under that camera.
    """

    def __init__(self, camera_id: str = ""):
        """Initialize task manager.

        Args:
            camera_id: Camera identifier (for logging).
        """
        self.camera_id = camera_id
        self._tasks: Dict[str, "TaskBase"] = {}
        self._lock = threading.RLock()
        self.log = Logger.get_logging_method(f"TASKMGR-{camera_id[:18]}")

    def add_task(self, task_id: str) -> Optional["TaskBase"]:
        """Add a new task.

        Creates a task instance from the registered task type
        based on the task configuration.

        Args:
            task_id: Task identifier (config must exist in TaskConfigManager).

        Returns:
            Created task instance or None if failed.
        """
        with self._lock:
            if task_id in self._tasks:
                self.log(f"Task already exists: {task_id}")
                return self._tasks[task_id]

            # Get config
            config = TaskConfigManager().get_config(task_id)
            if config is None:
                self.log(f"Config not found for task: {task_id}")
                return None

            # Create task using registry
            try:
                task = TaskRegistry.create(config.task_type, task_id)
                self._tasks[task_id] = task
                self.log(f"Task added: {task_id} ({config.task_type})")
                return task

            except ValueError as e:
                self.log(f"Failed to create task: {e}")
                return None

    def remove_task(self, task_id: str) -> bool:
        """Remove and stop a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if removed successfully.
        """
        with self._lock:
            if task_id not in self._tasks:
                self.log(f"Task not found: {task_id}")
                return False

            task = self._tasks[task_id]

            # Stop the task first
            if not task.is_stopped():
                task.stop()

            # Remove from dict
            del self._tasks[task_id]
            self.log(f"Task removed: {task_id}")
            return True

    def get_task(self, task_id: str) -> Optional["TaskBase"]:
        """Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task instance or None.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def all_tasks(self) -> List["TaskBase"]:
        """Get all tasks.

        Returns:
            List of all task instances.
        """
        with self._lock:
            return list(self._tasks.values())

    def all_task_ids(self) -> List[str]:
        """Get all task IDs.

        Returns:
            List of task identifiers.
        """
        with self._lock:
            return list(self._tasks.keys())

    def count(self) -> int:
        """Get number of tasks.

        Returns:
            Task count.
        """
        with self._lock:
            return len(self._tasks)

    def is_empty(self) -> bool:
        """Check if manager has no tasks.

        Returns:
            True if no tasks.
        """
        with self._lock:
            return len(self._tasks) == 0

    # ==================== Lifecycle Operations ====================

    def start_task(self, task_id: str) -> bool:
        """Start a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            True if started successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                self.log(f"Task not found: {task_id}")
                return False
            return task.start()

    def stop_task(self, task_id: str) -> bool:
        """Stop a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            True if stopped successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                self.log(f"Task not found: {task_id}")
                return False
            return task.stop()

    def pause_task(self, task_id: str) -> bool:
        """Pause a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            True if paused successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                self.log(f"Task not found: {task_id}")
                return False
            return task.pause()

    def resume_task(self, task_id: str) -> bool:
        """Resume a specific task.

        Args:
            task_id: Task identifier.

        Returns:
            True if resumed successfully.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                self.log(f"Task not found: {task_id}")
                return False
            return task.resume()

    def start_all(self) -> None:
        """Start all tasks."""
        with self._lock:
            for task in self._tasks.values():
                if not task.is_running():
                    task.start()

    def stop_all(self) -> None:
        """Stop all tasks."""
        with self._lock:
            for task in self._tasks.values():
                if not task.is_stopped():
                    task.stop()

    def pause_all(self) -> None:
        """Pause all running tasks."""
        with self._lock:
            for task in self._tasks.values():
                if task.is_running():
                    task.pause()

    def resume_all(self) -> None:
        """Resume all paused tasks."""
        with self._lock:
            for task in self._tasks.values():
                if task.is_paused():
                    task.resume()

    # ==================== Query Methods ====================

    def get_running_tasks(self) -> List["TaskBase"]:
        """Get all running tasks.

        Returns:
            List of running task instances.
        """
        with self._lock:
            return [t for t in self._tasks.values() if t.is_running()]

    def get_tasks_requiring_stream(self) -> List["TaskBase"]:
        """Get all tasks that require video stream.

        Returns:
            List of tasks needing stream input.
        """
        with self._lock:
            return [t for t in self._tasks.values()
                    if t.requires_stream() and t.is_running()]

    def has_running_tasks(self) -> bool:
        """Check if any task is running.

        Returns:
            True if at least one task is running.
        """
        with self._lock:
            return any(t.is_running() for t in self._tasks.values())

    def has_tasks_requiring_stream(self) -> bool:
        """Check if any running task requires stream.

        Returns:
            True if at least one running task needs stream.
        """
        with self._lock:
            return any(t.requires_stream() and t.is_running()
                      for t in self._tasks.values())

    # ==================== Frame Distribution ====================

    def distribute_frame(
        self,
        camera_name: str,
        timestamp: float,
        frame
    ) -> int:
        """Distribute a frame to all running tasks.

        Args:
            camera_name: Source camera name.
            timestamp: Frame timestamp.
            frame: BGR image.

        Returns:
            Number of tasks that received the frame.
        """
        count = 0
        with self._lock:
            for task in self._tasks.values():
                if task.is_running() and task.requires_stream():
                    if task.put_frame(camera_name, timestamp, frame):
                        count += 1
        return count

    def __repr__(self) -> str:
        """String representation."""
        return f"TaskManager(camera={self.camera_id}, tasks={self.count()})"
