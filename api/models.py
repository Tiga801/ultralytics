"""API request/response models.

This module provides Pydantic models for API request validation
and response serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class TaskRunRequest:
    """Request model for starting tasks.

    Attributes:
        tasks: List of task configurations (AnalyseCondition format).
    """
    tasks: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRunRequest":
        """Create from dictionary.

        Args:
            data: Request data with 'analyseConditions' or 'tasks' key.

        Returns:
            TaskRunRequest instance.
        """
        tasks = data.get("analyseConditions", data.get("tasks", []))
        if isinstance(tasks, dict):
            tasks = [tasks]
        return cls(tasks=tasks)


@dataclass
class TaskDeleteRequest:
    """Request model for deleting tasks.

    Attributes:
        task_ids: List of task IDs to delete.
    """
    task_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDeleteRequest":
        """Create from dictionary.

        Args:
            data: Request data with 'taskIds' or 'analyseConditions' key.

        Returns:
            TaskDeleteRequest instance.
        """
        task_ids = data.get("taskIds", [])
        if not task_ids:
            # Extract from analyseConditions
            conditions = data.get("analyseConditions", [])
            if isinstance(conditions, dict):
                conditions = [conditions]
            task_ids = [c.get("taskID") for c in conditions if c.get("taskID")]
        return cls(task_ids=task_ids)


@dataclass
class TaskPauseRequest:
    """Request model for pausing a task.

    Attributes:
        task_id: Task ID to pause.
    """
    task_id: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPauseRequest":
        """Create from dictionary."""
        task_id = data.get("taskId", data.get("taskID", ""))
        return cls(task_id=task_id)


@dataclass
class TaskResumeRequest:
    """Request model for resuming a task.

    Attributes:
        task_id: Task ID to resume.
    """
    task_id: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResumeRequest":
        """Create from dictionary."""
        task_id = data.get("taskId", data.get("taskID", ""))
        return cls(task_id=task_id)


@dataclass
class TaskStatusRequest:
    """Request model for querying task status.

    Attributes:
        task_ids: Optional list of task IDs to query.
    """
    task_ids: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStatusRequest":
        """Create from dictionary."""
        task_ids = data.get("taskIds", data.get("taskIDs"))
        return cls(task_ids=task_ids)


@dataclass
class ApiResponse:
    """Standard API response model.

    Attributes:
        code: Response code (0 = success).
        message: Response message.
        data: Response data.
    """
    code: int = 0
    message: str = "success"
    data: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data,
        }

    @classmethod
    def success(cls, data: Any = None, message: str = "success") -> "ApiResponse":
        """Create a success response."""
        return cls(code=0, message=message, data=data)

    @classmethod
    def error(cls, code: int, message: str, data: Any = None) -> "ApiResponse":
        """Create an error response."""
        return cls(code=code, message=message, data=data)


@dataclass
class TaskStatusResponse:
    """Task status response model.

    Attributes:
        task_id: Task identifier.
        task_name: Task name.
        task_type: Task type.
        status: Status code (1=paused, 4=running).
        is_running: Whether task is running.
        is_paused: Whether task is paused.
    """
    task_id: str
    task_name: str = ""
    task_type: str = ""
    status: int = 0
    is_running: bool = False
    is_paused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taskId": self.task_id,
            "taskName": self.task_name,
            "taskType": self.task_type,
            "status": self.status,
            "isRunning": self.is_running,
            "isPaused": self.is_paused,
        }


@dataclass
class CapabilityResponse:
    """Capability response model for Algorithm Warehouse.

    Attributes:
        task_cur_num: Current number of tasks.
        task_total_num: Maximum task capacity.
        total_capability: Total capability score.
        cur_capability: Current available capability.
        resolution_cap: Supported resolution range.
    """
    task_cur_num: int = 0
    task_total_num: int = 10
    total_capability: int = 100
    cur_capability: int = 100
    resolution_cap: List[int] = field(default_factory=lambda: [300, 500])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Algorithm Warehouse format."""
        return {
            "taskCurNum": self.task_cur_num,
            "taskTotalNum": self.task_total_num,
            "totalCapability": self.total_capability,
            "curCapability": self.cur_capability,
            "resolutionCap": self.resolution_cap,
        }


@dataclass
class HealthResponse:
    """Health check response model.

    Attributes:
        status: Health status string.
        initialized: Whether engine is initialized.
        started: Whether engine is started.
        task_count: Number of active tasks.
        camera_count: Number of active cameras.
        stand_count: Number of active stands.
    """
    status: str = "healthy"
    initialized: bool = False
    started: bool = False
    task_count: int = 0
    camera_count: int = 0
    stand_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "initialized": self.initialized,
            "started": self.started,
            "taskCount": self.task_count,
            "cameraCount": self.camera_count,
            "standCount": self.stand_count,
        }
