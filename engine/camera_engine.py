"""Camera engine for per-camera video ingestion and task management.

This module provides the CameraEngine class that manages video streams
and tasks for a single camera.
"""

import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from utils import Logger, get_timestamp
from task import TaskManager, TaskConfig, TaskConfigManager
from .config import get_engine_config
from .constants import DEFAULT_QUEUE_SIZE, DEFAULT_FRAME_EXPIRED_TIME


class CameraEngine:
    """Per-camera video ingestion and task execution engine.

    This class manages:
    - Single video stream loading thread per camera
    - Shared frame queue for all tasks
    - Task lifecycle within the camera
    - Automatic stream start/stop based on task count

    Key principles:
    - ONE stream thread per camera (frames shared via queue)
    - Stream starts when first task is added
    - Stream stops when last task is removed
    """

    QUEUE_MAXSIZE = DEFAULT_QUEUE_SIZE
    FRAME_EXPIRED_TIME = DEFAULT_FRAME_EXPIRED_TIME

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        stand_name: str = "default"
    ):
        """Initialize camera engine.

        Args:
            camera_id: Unique camera identifier.
            camera_name: Human-readable camera name.
            stand_name: Parent stand name.
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.stand_name = stand_name

        # Logging
        self.log = Logger.get_logging_method(f"CAM-{camera_name[:18]}")

        # Frame queue (shared buffer for all tasks)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)

        # Task management
        self._task_manager = TaskManager(camera_id)

        # Stream management
        self._stream_loader = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop_event = threading.Event()
        self._is_stream_running = False

        # Engine thread
        self._engine_thread: Optional[threading.Thread] = None
        self._engine_stop_event = threading.Event()

        # Stream info
        self._rtsp_url: Optional[str] = None
        self.camera_width: int = 0
        self.camera_height: int = 0
        self.fps: int = 25

        # Lock for thread safety
        self._lock = threading.RLock()

        self.log(f"CameraEngine initialized: {camera_id}")

    # ==================== Stream Management ====================

    def _get_rtsp_url(self, task_id: str) -> Optional[str]:
        """Get RTSP URL from task configuration.

        Args:
            task_id: Task identifier.

        Returns:
            RTSP URL or None.
        """
        config = TaskConfigManager().get_config(task_id)
        if config:
            return config.rtsp_url
        return None

    def _update_stream_info(self, task_id: str) -> None:
        """Update stream URL from task config.

        Note: Resolution and FPS are obtained from actual stream properties
        when the stream opens, not from config.

        Args:
            task_id: Task identifier.
        """
        config = TaskConfigManager().get_config(task_id)
        if config:
            self._rtsp_url = config.rtsp_url
            self.log(f"Stream URL configured: {self._rtsp_url}")

    def _start_stream(self) -> bool:
        """Start the video stream loading thread.

        After stream opens, syncs actual stream properties (resolution, FPS)
        to CameraEngine and all task configurations.

        Returns:
            True if started successfully.
        """
        if self._is_stream_running:
            self.log("Stream already running")
            return True

        if not self._rtsp_url:
            self.log("No RTSP URL configured")
            return False

        try:
            from stream import StreamLoader

            self._stream_stop_event.clear()

            # Create stream loader
            self._stream_loader = StreamLoader(
                sources=self._rtsp_url,
                vid_stride=1,
                buffer=False,
                reconnect=True
            )

            # Get actual stream properties and sync to task configs
            props = self._stream_loader.get_stream_properties(0)
            if props["width"] > 0 and props["height"] > 0:
                # Update CameraEngine properties with actual values
                self.camera_width = props["width"]
                self.camera_height = props["height"]
                self.fps = int(props["fps"]) if props["fps"] > 0 else self.fps

                # Sync actual values to all task configs
                self._sync_actual_stream_params(props)

                self.log(f"Actual stream info: {self.camera_width}x{self.camera_height} @ {self.fps}fps")

            # Start stream thread
            self._stream_thread = threading.Thread(
                target=self._stream_loop,
                daemon=True,
                name=f"Stream-{self.camera_name}"
            )
            self._stream_thread.start()

            self._is_stream_running = True
            self.log("Stream thread started")
            return True

        except Exception as e:
            self.log(f"Failed to start stream: {e}")
            return False

    def _sync_actual_stream_params(self, props: Dict) -> None:
        """Sync actual stream parameters to all task configs.

        Compares actual values with configured values and logs warnings
        to streams.log when mismatches are detected.

        Args:
            props: Dictionary with width, height, fps from StreamLoader.
        """
        from stream.utils import get_stream_file_logger
        stream_log = get_stream_file_logger()

        for task_id in self._task_manager.all_task_ids():
            config = TaskConfigManager().get_config(task_id)
            if config:
                config.update_actual_stream_params(
                    width=props["width"],
                    height=props["height"],
                    fps=props["fps"],
                    stream_log=stream_log
                )

    def _stop_stream(self) -> None:
        """Stop the video stream loading thread."""
        if not self._is_stream_running:
            return

        self.log("Stopping stream thread...")
        self._stream_stop_event.set()

        # Close stream loader
        if self._stream_loader:
            try:
                self._stream_loader.close()
            except Exception as e:
                self.log(f"Error closing stream loader: {e}")
            self._stream_loader = None

        # Wait for thread
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)
            if self._stream_thread.is_alive():
                self.log("Stream thread did not stop gracefully")

        self._stream_thread = None
        self._is_stream_running = False
        self.log("Stream thread stopped")

    def _stream_loop(self) -> None:
        """Stream reading loop (runs in separate thread)."""
        self.log("Stream loop started")

        # Frame counter for debugging frame skipping
        frame_index = 0

        try:
            for sources, images, info in self._stream_loader:
                if self._stream_stop_event.is_set():
                    break

                if images and len(images) > 0:
                    frame = images[0]
                    timestamp = get_timestamp()
                    frame_index += 1

                    # Frame counter output (stdout only, new line per frame)
                    print(f"[FRAME] Camera: {self.camera_name} | Frame: {frame_index}")

                    # Put frame in queue (non-blocking)
                    try:
                        self._frame_queue.put_nowait(
                            (self.camera_name, timestamp, frame)
                        )
                    except queue.Full:
                        # Queue full, drop oldest frame
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(
                                (self.camera_name, timestamp, frame)
                            )
                        except queue.Empty:
                            pass

        except Exception as e:
            if not self._stream_stop_event.is_set():
                self.log(f"Stream loop error: {e}")

        self.log("Stream loop ended")

    def _update_stream_status(self) -> None:
        """Update stream status based on task requirements.

        Starts stream if tasks need it, stops if no tasks need it.
        """
        with self._lock:
            needs_stream = self._task_manager.has_tasks_requiring_stream()

            if needs_stream and not self._is_stream_running:
                self._start_stream()
            elif not needs_stream and self._is_stream_running:
                self._stop_stream()

    # ==================== Task Management ====================

    def add_task(self, task_id: str) -> bool:
        """Add a task to this camera.

        Args:
            task_id: Task identifier (config must exist).

        Returns:
            True if task added successfully.
        """
        with self._lock:
            # Update stream info from task config
            self._update_stream_info(task_id)

            # Add task to manager
            task = self._task_manager.add_task(task_id)
            if task is None:
                return False

            # Start the task
            task.start()

            # Update stream status (may start stream)
            self._update_stream_status()

            self.log(f"Task added and started: {task_id}")
            return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from this camera.

        Args:
            task_id: Task identifier.

        Returns:
            True if task removed successfully.
        """
        with self._lock:
            result = self._task_manager.remove_task(task_id)

            # Update stream status (may stop stream)
            self._update_stream_status()

            if result:
                self.log(f"Task removed: {task_id}")
            return result

    def pause_task(self, task_id: str) -> bool:
        """Pause a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if paused successfully.
        """
        with self._lock:
            result = self._task_manager.pause_task(task_id)
            self._update_stream_status()
            return result

    def resume_task(self, task_id: str) -> bool:
        """Resume a task.

        Args:
            task_id: Task identifier.

        Returns:
            True if resumed successfully.
        """
        with self._lock:
            result = self._task_manager.resume_task(task_id)
            self._update_stream_status()
            return result

    # ==================== Engine Lifecycle ====================

    def start(self) -> None:
        """Start the camera engine."""
        if self._engine_thread and self._engine_thread.is_alive():
            self.log("Engine already running")
            return

        self._engine_stop_event.clear()
        self._engine_thread = threading.Thread(
            target=self._engine_loop,
            daemon=True,
            name=f"CamEngine-{self.camera_name}"
        )
        self._engine_thread.start()
        self.log("Camera engine started")

    def stop(self) -> None:
        """Stop the camera engine."""
        self.log("Stopping camera engine...")

        # Stop all tasks
        self._task_manager.stop_all()

        # Stop stream
        self._stop_stream()

        # Stop engine thread
        self._engine_stop_event.set()
        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=5)

        self._engine_thread = None
        self.log("Camera engine stopped")

    def _engine_loop(self) -> None:
        """Main engine loop for frame distribution."""
        self.log("Engine loop started")

        # Distribution counter for debugging
        distributed_count = 0

        while not self._engine_stop_event.is_set():
            try:
                # Get frame from queue with timeout
                try:
                    item = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is None:
                    continue

                camera_name, timestamp, frame = item

                # Check frame expiration
                current_time = get_timestamp()
                if current_time - timestamp > self.FRAME_EXPIRED_TIME:
                    continue  # Skip expired frame

                # Distribute frame to all running tasks
                self._task_manager.distribute_frame(
                    camera_name, timestamp, frame
                )

                distributed_count += 1
                # Distribution counter output (stdout only, new line per distribution)
                print(f"[DIST] Camera: {self.camera_name} | Distributed: {distributed_count}")

            except Exception as e:
                self.log(f"Engine loop error: {e}")

        self.log("Engine loop ended")

    # ==================== Query Methods ====================

    def get_task_count(self) -> int:
        """Get number of tasks."""
        return self._task_manager.count()

    def has_tasks(self) -> bool:
        """Check if camera has any tasks."""
        return not self._task_manager.is_empty()

    def is_stream_running(self) -> bool:
        """Check if stream is running."""
        return self._is_stream_running

    def get_status(self) -> Dict:
        """Get camera engine status.

        Returns:
            Status dictionary.
        """
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "stand_name": self.stand_name,
            "rtsp_url": self._rtsp_url,
            "resolution": f"{self.camera_width}x{self.camera_height}",
            "fps": self.fps,
            "stream_running": self._is_stream_running,
            "task_count": self._task_manager.count(),
            "tasks": self._task_manager.all_task_ids(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CameraEngine(id={self.camera_id}, "
            f"tasks={self._task_manager.count()}, "
            f"stream={self._is_stream_running})"
        )
