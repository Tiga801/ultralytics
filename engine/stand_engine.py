"""Stand engine for per-stand coordination.

This module provides the StandEngine class that coordinates multiple
cameras within a single stand (location).
"""

import threading
from typing import Dict, List, Optional

from utils import Logger
from task import TaskConfig, TaskConfigManager
from .camera_engine import CameraEngine


class StandEngine:
    """Per-stand coordinator managing multiple cameras.

    This class manages all CameraEngine instances for a single stand
    (e.g., an airport gate), providing:
    - Camera lifecycle management
    - Task routing to appropriate cameras
    - Stand-level resource coordination
    """

    def __init__(self, stand_id: str, stand_name: str = ""):
        """Initialize stand engine.

        Args:
            stand_id: Unique stand identifier.
            stand_name: Human-readable stand name.
        """
        self.stand_id = stand_id
        self.stand_name = stand_name or stand_id

        # Camera engines
        self._camera_engines: Dict[str, CameraEngine] = {}
        self._lock = threading.RLock()

        # State tracking
        self._started = False

        # Logging
        self.log = Logger.get_logging_method(f"STAND-{stand_name[:18]}")
        self.log(f"StandEngine initialized: {stand_id}")

    # ==================== Camera Management ====================

    def get_or_create_camera_engine(
        self,
        camera_id: str,
        camera_name: str = ""
    ) -> CameraEngine:
        """Get existing or create new camera engine.

        Args:
            camera_id: Camera identifier.
            camera_name: Human-readable camera name.

        Returns:
            CameraEngine instance.
        """
        with self._lock:
            if camera_id not in self._camera_engines:
                camera_name = camera_name or camera_id
                engine = CameraEngine(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    stand_name=self.stand_name
                )
                self._camera_engines[camera_id] = engine
                self.log(f"Created camera engine: {camera_id}")

                # Start if stand is already started
                if self._started:
                    engine.start()

            return self._camera_engines[camera_id]

    def get_camera_engine(self, camera_id: str) -> Optional[CameraEngine]:
        """Get camera engine by ID.

        Args:
            camera_id: Camera identifier.

        Returns:
            CameraEngine or None.
        """
        with self._lock:
            return self._camera_engines.get(camera_id)

    def remove_camera_engine(self, camera_id: str) -> bool:
        """Remove and stop a camera engine.

        Args:
            camera_id: Camera identifier.

        Returns:
            True if removed.
        """
        with self._lock:
            if camera_id not in self._camera_engines:
                return False

            engine = self._camera_engines[camera_id]
            engine.stop()
            del self._camera_engines[camera_id]
            self.log(f"Removed camera engine: {camera_id}")
            return True

    # ==================== Task Management ====================

    def add_task(self, task_id: str) -> bool:
        """Add a task to the appropriate camera.

        The task configuration determines which camera the task
        belongs to.

        Args:
            task_id: Task identifier.

        Returns:
            True if task added successfully.
        """
        with self._lock:
            # Get task config
            config = TaskConfigManager().get_config(task_id)
            if config is None:
                self.log(f"Config not found for task: {task_id}")
                return False

            # Get or create camera engine
            camera_engine = self.get_or_create_camera_engine(
                camera_id=config.camera_id,
                camera_name=config.camera_name
            )

            # Add task to camera
            return camera_engine.add_task(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if removed successfully.
        """
        with self._lock:
            # Get task config to find camera
            config = TaskConfigManager().get_config(task_id)
            if config is None:
                self.log(f"Config not found for task: {task_id}")
                return False

            # Find camera engine
            camera_engine = self.get_camera_engine(config.camera_id)
            if camera_engine is None:
                self.log(f"Camera engine not found: {config.camera_id}")
                return False

            # Remove task
            result = camera_engine.remove_task(task_id)

            # Clean up empty camera engines
            if not camera_engine.has_tasks():
                self.remove_camera_engine(config.camera_id)

            return result

    def pause_task(self, task_id: str) -> bool:
        """Pause a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if paused successfully.
        """
        camera_engine = self._find_camera_for_task(task_id)
        if camera_engine:
            return camera_engine.pause_task(task_id)
        return False

    def resume_task(self, task_id: str) -> bool:
        """Resume a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if resumed successfully.
        """
        camera_engine = self._find_camera_for_task(task_id)
        if camera_engine:
            return camera_engine.resume_task(task_id)
        return False

    def _find_camera_for_task(self, task_id: str) -> Optional[CameraEngine]:
        """Find the camera engine containing a task.

        Args:
            task_id: Task identifier.

        Returns:
            CameraEngine or None.
        """
        config = TaskConfigManager().get_config(task_id)
        if config:
            return self.get_camera_engine(config.camera_id)
        return None

    # ==================== Lifecycle ====================

    def start(self) -> None:
        """Start the stand engine and all cameras."""
        with self._lock:
            if self._started:
                self.log("Stand engine already started")
                return

            self.log("Starting stand engine...")

            # Start all camera engines
            for camera_engine in self._camera_engines.values():
                camera_engine.start()

            self._started = True
            self.log("Stand engine started")

    def stop(self) -> None:
        """Stop the stand engine and all cameras."""
        with self._lock:
            if not self._started:
                return

            self.log("Stopping stand engine...")

            # Stop all camera engines
            for camera_engine in self._camera_engines.values():
                camera_engine.stop()

            self._camera_engines.clear()
            self._started = False
            self.log("Stand engine stopped")

    # ==================== Query Methods ====================

    def get_camera_count(self) -> int:
        """Get number of cameras."""
        with self._lock:
            return len(self._camera_engines)

    def get_task_count(self) -> int:
        """Get total number of tasks across all cameras."""
        with self._lock:
            return sum(ce.get_task_count() for ce in self._camera_engines.values())

    def get_all_task_ids(self) -> List[str]:
        """Get all task IDs across all cameras."""
        with self._lock:
            task_ids = []
            for ce in self._camera_engines.values():
                task_ids.extend(ce._task_manager.all_task_ids())
            return task_ids

    def is_started(self) -> bool:
        """Check if stand engine is started."""
        return self._started

    def get_status(self) -> Dict:
        """Get stand engine status.

        Returns:
            Status dictionary.
        """
        with self._lock:
            return {
                "stand_id": self.stand_id,
                "stand_name": self.stand_name,
                "started": self._started,
                "camera_count": len(self._camera_engines),
                "task_count": self.get_task_count(),
                "cameras": {
                    cid: ce.get_status()
                    for cid, ce in self._camera_engines.items()
                }
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StandEngine(id={self.stand_id}, "
            f"cameras={len(self._camera_engines)}, "
            f"started={self._started})"
        )
