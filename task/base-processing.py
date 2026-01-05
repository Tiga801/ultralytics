"""Task base class.

This module provides the TaskBase abstract class that all CV tasks must extend.
Each task runs as a separate OS process for resource isolation.
"""

import multiprocessing as mp
import queue
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from utils import Logger
from .config import TaskConfig, TaskConfigManager
from .results import TaskResult
from .state import TaskState, TaskStateMachine


class TaskBase(ABC):
    """Abstract base class for all CV tasks.

    Each task runs as a SEPARATE OS PROCESS for resource isolation.
    This ensures independent memory space, GPU resource isolation,
    and easy cleanup when tasks are stopped.

    Subclasses must implement:
        - on_process(frame, timestamp) -> TaskResult
        - requires_stream() -> bool

    Optional hooks:
        - on_start(): Called when task starts
        - on_stop(): Called when task stops
        - on_pause(): Called when task is paused
        - on_resume(): Called when task resumes

    Example:
        >>> class MyTask(TaskBase):
        ...     def requires_stream(self) -> bool:
        ...         return True
        ...
        ...     def on_process(self, frame, timestamp) -> TaskResult:
        ...         # Process frame and return results
        ...         return TaskResult(task_id=self.task_id)
    """

    def __init__(self, task_id: str):
        """Initialize task.

        Args:
            task_id: Unique task identifier.
        """
        self.task_id = task_id
        self.task_config: TaskConfig = TaskConfigManager().get_config(task_id)

        if self.task_config is None:
            raise ValueError(f"Task config not found for task_id: {task_id}")

        # State management
        self._state_machine = TaskStateMachine()

        # Process communication queues
        self._input_queue: Optional[mp.Queue] = None   # Frames from CameraEngine
        self._output_queue: Optional[mp.Queue] = None  # Results back to manager
        self._control_queue: Optional[mp.Queue] = None # Control commands

        # Process handle
        self._process: Optional[mp.Process] = None

        # Logger
        self.log = Logger.get_logging_method(f"TASK-{task_id[:18]}")

    @property
    def task_name(self) -> str:
        """Get task name."""
        return self.task_config.task_name

    @property
    def task_type(self) -> str:
        """Get task type."""
        return self.task_config.task_type

    @property
    def state(self) -> TaskState:
        """Get current task state."""
        return self._state_machine.state

    def is_running(self) -> bool:
        """Check if task is running."""
        return self._state_machine.is_running()

    def is_paused(self) -> bool:
        """Check if task is paused."""
        return self._state_machine.is_paused()

    def is_stopped(self) -> bool:
        """Check if task is stopped."""
        return self._state_machine.is_stopped()

    # ==================== Lifecycle Methods ====================

    def start(self) -> bool:
        """Start the task process.

        Returns:
            True if started successfully.
        """
        if not self._state_machine.can_transition(TaskState.RUNNING):
            self.log(f"Cannot start task from state {self.state}")
            return False

        try:
            # Create communication queues
            self._input_queue = mp.Queue(maxsize=50)
            self._output_queue = mp.Queue(maxsize=100)
            self._control_queue = mp.Queue(maxsize=10)

            # Create and start process
            self._process = mp.Process(
                target=self._process_main,
                args=(
                    self.task_id,
                    self.task_config.to_dict(),
                    self._input_queue,
                    self._output_queue,
                    self._control_queue
                ),
                daemon=True,
                name=f"Task-{self.task_id[:18]}"
            )
            self._process.start()

            # Transition state
            self._state_machine.transition(TaskState.RUNNING)
            self.log(f"Task started (PID: {self._process.pid})")

            # Call hook
            self.on_start()
            return True

        except Exception as e:
            self.log(f"Failed to start task: {e}")
            self._state_machine.transition(TaskState.ERROR, str(e))
            return False

    def stop(self) -> bool:
        """Stop the task process.

        Returns:
            True if stopped successfully.
        """
        if self._state_machine.is_stopped():
            return True

        try:
            # Send stop command
            if self._control_queue:
                try:
                    self._control_queue.put_nowait({"command": "stop"})
                except queue.Full:
                    pass

            # Wait for process to terminate
            if self._process and self._process.is_alive():
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self.log("Force terminating task process")
                    self._process.terminate()
                    self._process.join(timeout=2)

            # Cleanup queues
            self._cleanup_queues()

            # Transition state
            self._state_machine.transition(TaskState.STOPPED)
            self.log("Task stopped")

            # Call hook
            self.on_stop()
            return True

        except Exception as e:
            self.log(f"Error stopping task: {e}")
            self._state_machine.force_state(TaskState.STOPPED)
            return False

    def pause(self) -> bool:
        """Pause the task.

        Returns:
            True if paused successfully.
        """
        if not self._state_machine.can_transition(TaskState.PAUSED):
            return False

        try:
            if self._control_queue:
                self._control_queue.put_nowait({"command": "pause"})

            self._state_machine.transition(TaskState.PAUSED)
            self.log("Task paused")
            self.on_pause()
            return True

        except Exception as e:
            self.log(f"Error pausing task: {e}")
            return False

    def resume(self) -> bool:
        """Resume the task.

        Returns:
            True if resumed successfully.
        """
        if not self._state_machine.can_transition(TaskState.RUNNING):
            return False

        try:
            if self._control_queue:
                self._control_queue.put_nowait({"command": "resume"})

            self._state_machine.transition(TaskState.RUNNING)
            self.log("Task resumed")
            self.on_resume()
            return True

        except Exception as e:
            self.log(f"Error resuming task: {e}")
            return False

    def _cleanup_queues(self) -> None:
        """Clean up multiprocessing queues."""
        for q in [self._input_queue, self._output_queue, self._control_queue]:
            if q:
                try:
                    while not q.empty():
                        q.get_nowait()
                    q.close()
                    q.join_thread()
                except Exception:
                    pass

        self._input_queue = None
        self._output_queue = None
        self._control_queue = None

    # ==================== Process Entry Point ====================

    @classmethod
    def _process_main(
        cls,
        task_id: str,
        config_dict: Dict[str, Any],
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        control_queue: mp.Queue,
    ) -> None:
        """Main entry point for the task process.

        This runs in a separate process and handles the main processing loop.

        Args:
            task_id: Task identifier.
            config_dict: Task configuration as dictionary.
            input_queue: Queue for receiving frames.
            output_queue: Queue for sending results.
            control_queue: Queue for control commands.
        """
        # Import here to avoid issues with multiprocessing
        from task.registry import TaskRegistry

        # Reconstruct config first (needed for logger setup)
        config = TaskConfig(**{k: v for k, v in config_dict.items()
                              if k in TaskConfig.__dataclass_fields__})

        # Initialize logger in subprocess - use task-specific log file
        Logger.init()
        log = Logger.get_task_logger(
            config.stand_name,
            config.camera_name,
            task_id
        )
        log("Task process started")

        # Create task instance using registry
        try:
            task_cls = TaskRegistry.get_class(config.task_type)
            if task_cls is None:
                log(f"Unknown task type: {config.task_type}")
                return

            # Create a minimal task instance for processing
            # Note: We can't use the full __init__ in subprocess
            task_instance = object.__new__(task_cls)
            task_instance.task_id = task_id
            task_instance.task_config = config
            task_instance.log = log

            # Initialize task-specific resources
            task_instance._init_in_process()

        except Exception as e:
            log(f"Failed to initialize task: {e}")
            log(traceback.format_exc())
            return

        # Processing loop
        paused = False
        running = True
        frame_count = 0

        while running:
            # Check for control commands
            try:
                while not control_queue.empty():
                    cmd = control_queue.get_nowait()
                    if cmd.get("command") == "stop":
                        running = False
                    elif cmd.get("command") == "pause":
                        paused = True
                    elif cmd.get("command") == "resume":
                        paused = False
            except queue.Empty:
                pass

            if not running:
                break

            if paused:
                # When paused, still drain input queue to prevent blocking
                try:
                    input_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            # Get frame from input queue
            try:
                frame_data = input_queue.get(timeout=1.0)
                if frame_data is None:
                    continue

                camera_name, timestamp, frame = frame_data
                frame_count += 1

                # Process frame
                try:
                    result = task_instance.on_process(frame, timestamp)
                    if result:
                        result.frame_id = frame_count
                        # Send result back (non-blocking)
                        try:
                            output_queue.put_nowait(result.to_dict())
                        except queue.Full:
                            pass  # Drop result if queue is full

                except Exception as e:
                    log(f"Error processing frame: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                log(f"Error in processing loop: {e}")

        # Cleanup
        try:
            task_instance._cleanup_in_process()
        except Exception as e:
            log(f"Error in cleanup: {e}")

        log("Task process exiting")

    def _init_in_process(self) -> None:
        """Initialize resources in the subprocess.

        Override this method to load models and allocate GPU resources.
        This is called once when the process starts.
        """
        pass

    def _cleanup_in_process(self) -> None:
        """Cleanup resources in the subprocess.

        Override this method to release models and GPU resources.
        This is called once when the process exits.
        """
        pass

    # ==================== Hook Methods ====================

    def on_start(self) -> None:
        """Hook called when task starts. Override in subclass."""
        pass

    def on_stop(self) -> None:
        """Hook called when task stops. Override in subclass."""
        pass

    def on_pause(self) -> None:
        """Hook called when task is paused. Override in subclass."""
        pass

    def on_resume(self) -> None:
        """Hook called when task resumes. Override in subclass."""
        pass

    # ==================== Abstract Methods ====================

    @abstractmethod
    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a single frame.

        This method is called for each frame in the subprocess.
        Implement task-specific processing logic here.

        Args:
            frame: BGR image as numpy array.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with detections, events, and optional visualization.
        """
        pass

    @abstractmethod
    def requires_stream(self) -> bool:
        """Check if this task requires video stream input.

        Returns:
            True if task needs video frames.
        """
        pass

    # ==================== Communication Methods ====================

    def put_frame(self, camera_name: str, timestamp: float, frame: np.ndarray) -> bool:
        """Put a frame into the task's input queue.

        Called by CameraEngine to distribute frames.

        Args:
            camera_name: Source camera name.
            timestamp: Frame timestamp.
            frame: BGR image.

        Returns:
            True if frame was queued successfully.
        """
        if self._input_queue is None:
            return False

        try:
            self._input_queue.put_nowait((camera_name, timestamp, frame))
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get a result from the task's output queue.

        Args:
            timeout: Maximum time to wait.

        Returns:
            Result dictionary or None.
        """
        if self._output_queue is None:
            return None

        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ==================== Status Methods ====================

    def get_status_for_api(self) -> Dict[str, Any]:
        """Get task status for API response.

        Returns:
            Status dictionary.
        """
        return {
            "taskId": self.task_id,
            "taskName": self.task_name,
            "taskType": self.task_type,
            "state": str(self.state),
            "isRunning": self.is_running(),
            "isPaused": self.is_paused(),
            "processAlive": self._process.is_alive() if self._process else False,
            "pid": self._process.pid if self._process else None,
        }

    def get_task_status_for_algo_warehouse(self) -> int:
        """Get task status code for Algorithm Warehouse.

        Returns:
            Status code: 1=paused, 4=running, others=4
        """
        if self.is_paused():
            return 1
        return 4

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.task_id}, state={self.state})"
