"""Main engine - global task scheduler and coordinator.

This module provides the MainEngine class that serves as the central
coordinator for the entire task management system.
"""

import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import SingletonMeta, Logger
from task import TaskConfig, TaskConfigManager
from .config import EngineConfig, get_engine_config
from .stand_engine import StandEngine
from .inference_manager import InferenceManager
from .exceptions import (
    TaskNotFoundException,
    TaskAlreadyExistsException,
    EngineNotInitializedException,
)


class MainEngine(metaclass=SingletonMeta):
    """Global task scheduler and resource coordinator (Singleton).

    This class is the top-level coordinator for the task management system,
    responsible for:
    - Receiving task configurations from API
    - Managing StandEngine instances
    - Dynamic task lifecycle (add/remove/pause/resume)
    - Resource coordination
    - Cleanup operations

    Engine Hierarchy:
        MainEngine (1 instance)
            └── StandEngine (1 per stand)
                    └── CameraEngine (1 per camera)
                            └── TaskManager
                                    └── Task instances (processes)
    """

    def __init__(self):
        """Initialize main engine."""
        self._stand_engines: Dict[str, StandEngine] = {}
        self._inference_manager: Optional[InferenceManager] = None
        self._config: Optional[EngineConfig] = None

        # Threading
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None

        # State
        self._initialized = False
        self._started = False

        # Logging
        self.log = Logger.get_logging_method("ENGINE")

    # ==================== Lifecycle ====================

    def init(self, config: Optional[EngineConfig] = None) -> None:
        """Initialize the engine.

        Args:
            config: Optional engine configuration.
        """
        if self._initialized:
            self.log("Engine already initialized")
            return

        self.log("Initializing MainEngine...")

        # Set configuration
        self._config = config or get_engine_config()

        # Initialize inference manager
        self._inference_manager = InferenceManager()

        # Start cleanup thread
        self._start_cleanup_thread()

        self._initialized = True
        self.log("MainEngine initialized")

    def start(self) -> None:
        """Start the engine and all stand engines."""
        if not self._initialized:
            raise EngineNotInitializedException()

        if self._started:
            self.log("Engine already started")
            return

        self.log("Starting MainEngine...")

        with self._lock:
            # Start all stand engines
            for stand_engine in self._stand_engines.values():
                stand_engine.start()

            self._started = True

        self.log("MainEngine started")

    def stop(self) -> None:
        """Stop the engine and all stand engines."""
        self.log("Stopping MainEngine...")

        self._stop_event.set()

        with self._lock:
            # Stop all stand engines
            for stand_engine in self._stand_engines.values():
                stand_engine.stop()

            self._stand_engines.clear()
            self._started = False

        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        self.log("MainEngine stopped")

    def run(self) -> None:
        """Run the engine (blocking).

        This method blocks until the engine is stopped.
        """
        if not self._initialized:
            self.init()

        self.start()

        Logger.title("EASYAIR Engine Running")

        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("Received shutdown signal")
        finally:
            self.stop()

    # ==================== Task Management ====================

    def add_task(self, task_config_data: Dict[str, Any]) -> bool:
        """Add a new task dynamically.

        Args:
            task_config_data: Task configuration (AnalyseCondition format).

        Returns:
            True if task added successfully.
        """
        task_id = task_config_data.get("taskID", "")
        self.log(f"Adding task: {task_id}")

        try:
            with self._lock:
                # Check if task already exists
                if TaskConfigManager().get_config(task_id):
                    self.log(f"Task already exists: {task_id}")
                    raise TaskAlreadyExistsException(task_id)

                # Build task config
                config = TaskConfigManager().build_config(task_config_data)

                # Get or create stand engine
                stand_engine = self._get_or_create_stand_engine(config.stand_name)

                # Add task to stand engine
                success = stand_engine.add_task(task_id)

                if success:
                    self.log(f"Task added successfully: {task_id}")
                else:
                    # Rollback config
                    TaskConfigManager().remove_config(task_id)
                    self.log(f"Failed to add task: {task_id}")

                return success

        except Exception as e:
            self.log(f"Error adding task: {e}")
            self.log(traceback.format_exc())
            return False

    def remove_task(self, task_id: str) -> bool:
        """Remove a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if removed successfully.
        """
        self.log(f"Removing task: {task_id}")

        try:
            with self._lock:
                # Get task config
                config = TaskConfigManager().get_config(task_id)
                if config is None:
                    self.log(f"Task not found: {task_id}")
                    return False

                # Find stand engine
                stand_engine = self._stand_engines.get(config.stand_name)
                if stand_engine is None:
                    self.log(f"Stand engine not found: {config.stand_name}")
                    return False

                # Remove task from stand
                success = stand_engine.remove_task(task_id)

                # Remove config
                TaskConfigManager().remove_config(task_id)

                # Clean up empty stand engines
                if stand_engine.get_task_count() == 0:
                    self._remove_stand_engine(config.stand_name)

                self.log(f"Task removed: {task_id}")
                return success

        except Exception as e:
            self.log(f"Error removing task: {e}")
            self.log(traceback.format_exc())
            return False

    def pause_task(self, task_id: str) -> bool:
        """Pause a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if paused successfully.
        """
        self.log(f"Pausing task: {task_id}")

        stand_engine = self._find_stand_for_task(task_id)
        if stand_engine:
            return stand_engine.pause_task(task_id)
        return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if resumed successfully.
        """
        self.log(f"Resuming task: {task_id}")

        stand_engine = self._find_stand_for_task(task_id)
        if stand_engine:
            return stand_engine.resume_task(task_id)
        return False

    # ==================== Stand Management ====================

    def _get_or_create_stand_engine(self, stand_name: str) -> StandEngine:
        """Get existing or create new stand engine.

        Args:
            stand_name: Stand name.

        Returns:
            StandEngine instance.
        """
        if stand_name not in self._stand_engines:
            engine = StandEngine(
                stand_id=stand_name,
                stand_name=stand_name
            )
            self._stand_engines[stand_name] = engine
            self.log(f"Created stand engine: {stand_name}")

            # Start if main engine is started
            if self._started:
                engine.start()

        return self._stand_engines[stand_name]

    def _remove_stand_engine(self, stand_name: str) -> None:
        """Remove a stand engine.

        Args:
            stand_name: Stand name.
        """
        if stand_name in self._stand_engines:
            engine = self._stand_engines[stand_name]
            engine.stop()
            del self._stand_engines[stand_name]
            self.log(f"Removed stand engine: {stand_name}")

    def _find_stand_for_task(self, task_id: str) -> Optional[StandEngine]:
        """Find the stand engine containing a task.

        Args:
            task_id: Task identifier.

        Returns:
            StandEngine or None.
        """
        config = TaskConfigManager().get_config(task_id)
        if config:
            return self._stand_engines.get(config.stand_name)
        return None

    # ==================== Cleanup ====================

    def _start_cleanup_thread(self) -> None:
        """Start the cleanup daemon thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CleanupThread"
        )
        self._cleanup_thread.start()
        self.log("Cleanup thread started")

    def _cleanup_loop(self) -> None:
        """Cleanup loop for removing old files."""
        while not self._stop_event.is_set():
            try:
                self._cleanup_old_files()
            except Exception as e:
                self.log(f"Cleanup error: {e}")

            # Wait for next cleanup
            self._stop_event.wait(self._config.cleanup_interval_hours * 3600)

    def _cleanup_old_files(self) -> None:
        """Clean up old files in logs and runs directories.

        Note: requests.log is excluded from cleanup (never deleted).
        """
        cutoff = datetime.now() - timedelta(days=self._config.file_expired_days)

        # Files to never delete (persistent logs)
        protected_files = {"requests.log"}

        for dir_name in [self._config.logs_dir, self._config.runs_dir]:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                continue

            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    # Skip protected files
                    if file_path.name in protected_files:
                        continue

                    try:
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff:
                            file_path.unlink()
                            self.log(f"Deleted old file: {file_path}")
                    except Exception as e:
                        self.log(f"Error deleting {file_path}: {e}")

    # ==================== Query Methods ====================

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a task.

        Args:
            task_id: Task identifier.

        Returns:
            Status dictionary or None.
        """
        config = TaskConfigManager().get_config(task_id)
        if config is None:
            return None

        stand_engine = self._stand_engines.get(config.stand_name)
        if stand_engine is None:
            return None

        camera_engine = stand_engine.get_camera_engine(config.camera_id)
        if camera_engine is None:
            return None

        task = camera_engine._task_manager.get_task(task_id)
        if task is None:
            return None

        return task.get_status_for_api()

    def get_all_tasks(self) -> List[Dict]:
        """Get status of all tasks.

        Returns:
            List of task status dictionaries.
        """
        tasks = []
        with self._lock:
            for stand_engine in self._stand_engines.values():
                for task_id in stand_engine.get_all_task_ids():
                    status = self.get_task_status(task_id)
                    if status:
                        tasks.append(status)
        return tasks

    def get_engine_status(self) -> Dict:
        """Get overall engine status.

        Returns:
            Status dictionary.
        """
        with self._lock:
            total_tasks = sum(
                se.get_task_count() for se in self._stand_engines.values()
            )
            total_cameras = sum(
                se.get_camera_count() for se in self._stand_engines.values()
            )

            return {
                "initialized": self._initialized,
                "started": self._started,
                "stand_count": len(self._stand_engines),
                "camera_count": total_cameras,
                "task_count": total_tasks,
                "stands": {
                    sid: se.get_status()
                    for sid, se in self._stand_engines.items()
                }
            }

    def get_capabilities(self) -> Dict:
        """Get engine capabilities for Algorithm Warehouse.

        Returns:
            Capabilities dictionary.
        """
        from .constants import MAX_TASK_CAPACITY, TOTAL_CAPABILITY, CAPABILITY_PER_TASK

        task_count = sum(
            se.get_task_count() for se in self._stand_engines.values()
        )

        return {
            "taskCurNum": task_count,
            "taskTotalNum": MAX_TASK_CAPACITY,
            "totalCapability": TOTAL_CAPABILITY,
            "curCapability": max(0, TOTAL_CAPABILITY - task_count * CAPABILITY_PER_TASK),
<<<<<<< HEAD
            "resolutionCap": [300, 500]
=======
            "resolutionCap": [300, 5000]
>>>>>>> 07331326 (feat: build video analytics task management system)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MainEngine(initialized={self._initialized}, "
            f"started={self._started}, "
            f"stands={len(self._stand_engines)})"
        )
