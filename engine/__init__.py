"""Engine module - Core engine layer for task management system.

This module provides the core engine components:
- MainEngine: Global task scheduler (singleton)
- StandEngine: Per-stand coordinator
- CameraEngine: Per-camera video ingestion and task execution
- InferenceManager: Model loading and caching
- EngineConfig: Global configuration

Engine Hierarchy:
    MainEngine (1 instance)
        └── StandEngine (1 per stand)
                └── CameraEngine (1 per camera)
                        ├── StreamLoader (1 thread)
                        ├── Frame Queue (shared buffer)
                        └── TaskManager
                                └── Task instances (processes)
"""

from .config import (
    EngineConfig,
    get_engine_config,
    set_engine_config,
    init_engine_config,
)
from .constants import (
    TaskType,
    ErrorCode,
    WarehouseTaskStatus,
    get_error_message,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_FRAME_EXPIRED_TIME,
)
from .exceptions import (
    EngineException,
    TaskException,
    TaskNotFoundException,
    TaskAlreadyExistsException,
    TaskStartException,
    TaskStopException,
    CameraException,
    CameraConnectionException,
    StreamException,
    ModelException,
    ModelNotFoundException,
    ModelLoadException,
    InferenceException,
    EngineNotInitializedException,
    ResourceExhaustedException,
)
from .inference_manager import InferenceManager
from .camera_engine import CameraEngine
from .stand_engine import StandEngine
from .main_engine import MainEngine

__all__ = [
    # Configuration
    "EngineConfig",
    "get_engine_config",
    "set_engine_config",
    "init_engine_config",
    # Constants
    "TaskType",
    "ErrorCode",
    "WarehouseTaskStatus",
    "get_error_message",
    "DEFAULT_QUEUE_SIZE",
    "DEFAULT_FRAME_EXPIRED_TIME",
    # Exceptions
    "EngineException",
    "TaskException",
    "TaskNotFoundException",
    "TaskAlreadyExistsException",
    "TaskStartException",
    "TaskStopException",
    "CameraException",
    "CameraConnectionException",
    "StreamException",
    "ModelException",
    "ModelNotFoundException",
    "ModelLoadException",
    "InferenceException",
    "EngineNotInitializedException",
    "ResourceExhaustedException",
    # Core engines
    "InferenceManager",
    "CameraEngine",
    "StandEngine",
    "MainEngine",
]
