"""Task framework module.

This module provides the core task framework for the task management system:
- TaskBase: Abstract base class for all CV tasks (runs as separate process)
- TaskConfig: Task configuration dataclass
- TaskConfigManager: Centralized configuration storage
- TaskState: Task lifecycle states
- TaskStateMachine: State transition management
- TaskResult: Processing result container
- TaskRegistry: Factory pattern for task creation
- TaskManager: Per-camera task lifecycle management
"""

from .state import TaskState, TaskStateMachine
from .config import TaskConfig, TaskConfigManager
from .results import TaskResult, BatchTaskResult
from .base import TaskBase
from .registry import TaskRegistry
from .manager import TaskManager

__all__ = [
    # State management
    "TaskState",
    "TaskStateMachine",
    # Configuration
    "TaskConfig",
    "TaskConfigManager",
    # Results
    "TaskResult",
    "BatchTaskResult",
    # Base class
    "TaskBase",
    # Factory
    "TaskRegistry",
    # Manager
    "TaskManager",
]
