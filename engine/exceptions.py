"""Custom exceptions for the engine system.

This module defines custom exception classes used throughout the engine.
"""

from .constants import ErrorCode, get_error_message


class EngineException(Exception):
    """Base exception for engine errors."""

    def __init__(
        self,
        message: str = "",
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    ):
        self.error_code = error_code
        self.message = message or get_error_message(error_code)
        super().__init__(self.message)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "ErrCode": int(self.error_code),
            "ErrMsg": self.message,
        }


class TaskException(EngineException):
    """Exception for task-related errors."""

    def __init__(
        self,
        task_id: str,
        message: str = "",
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    ):
        self.task_id = task_id
        super().__init__(message, error_code)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "TaskID": self.task_id,
            "ErrCode": int(self.error_code),
            "ErrMsg": self.message,
        }


class TaskNotFoundException(TaskException):
    """Exception raised when task is not found."""

    def __init__(self, task_id: str):
        super().__init__(
            task_id=task_id,
            message=f"Task not found: {task_id}",
            error_code=ErrorCode.TASK_NOT_FOUND
        )


class TaskAlreadyExistsException(TaskException):
    """Exception raised when task already exists."""

    def __init__(self, task_id: str):
        super().__init__(
            task_id=task_id,
            message=f"Task already exists: {task_id}",
            error_code=ErrorCode.TASK_ALREADY_EXISTS
        )


class TaskStartException(TaskException):
    """Exception raised when task fails to start."""

    def __init__(self, task_id: str, reason: str = ""):
        message = f"Failed to start task: {task_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            task_id=task_id,
            message=message,
            error_code=ErrorCode.TASK_START_FAILED
        )


class TaskStopException(TaskException):
    """Exception raised when task fails to stop."""

    def __init__(self, task_id: str, reason: str = ""):
        message = f"Failed to stop task: {task_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            task_id=task_id,
            message=message,
            error_code=ErrorCode.TASK_STOP_FAILED
        )


class CameraException(EngineException):
    """Exception for camera-related errors."""

    def __init__(
        self,
        camera_id: str,
        message: str = "",
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    ):
        self.camera_id = camera_id
        super().__init__(message, error_code)


class CameraConnectionException(CameraException):
    """Exception raised when camera connection fails."""

    def __init__(self, camera_id: str, rtsp_url: str = ""):
        message = f"Failed to connect to camera: {camera_id}"
        if rtsp_url:
            message += f" (URL: {rtsp_url})"
        super().__init__(
            camera_id=camera_id,
            message=message,
            error_code=ErrorCode.CAMERA_CONNECTION_FAILED
        )


class StreamException(CameraException):
    """Exception raised for stream errors."""

    def __init__(self, camera_id: str, reason: str = ""):
        message = f"Stream error for camera: {camera_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            camera_id=camera_id,
            message=message,
            error_code=ErrorCode.STREAM_ERROR
        )


class ModelException(EngineException):
    """Exception for model-related errors."""

    def __init__(
        self,
        model_name: str,
        message: str = "",
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    ):
        self.model_name = model_name
        super().__init__(message, error_code)


class ModelNotFoundException(ModelException):
    """Exception raised when model is not found."""

    def __init__(self, model_name: str, model_path: str = ""):
        message = f"Model not found: {model_name}"
        if model_path:
            message += f" (path: {model_path})"
        super().__init__(
            model_name=model_name,
            message=message,
            error_code=ErrorCode.MODEL_NOT_FOUND
        )


class ModelLoadException(ModelException):
    """Exception raised when model fails to load."""

    def __init__(self, model_name: str, reason: str = ""):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            model_name=model_name,
            message=message,
            error_code=ErrorCode.MODEL_LOAD_FAILED
        )


class InferenceException(ModelException):
    """Exception raised during inference."""

    def __init__(self, model_name: str, reason: str = ""):
        message = f"Inference error with model: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            model_name=model_name,
            message=message,
            error_code=ErrorCode.INFERENCE_ERROR
        )


class EngineNotInitializedException(EngineException):
    """Exception raised when engine is not initialized."""

    def __init__(self):
        super().__init__(
            message="Engine not initialized",
            error_code=ErrorCode.ENGINE_NOT_INITIALIZED
        )


class ResourceExhaustedException(EngineException):
    """Exception raised when resources are exhausted."""

    def __init__(self, resource_type: str = ""):
        message = "Resource exhausted"
        if resource_type:
            message += f": {resource_type}"
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_EXHAUSTED
        )
