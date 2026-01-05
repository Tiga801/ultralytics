"""Engine constants and error codes.

This module defines constants used throughout the engine system.
"""

from enum import Enum, IntEnum


class TaskType(str, Enum):
    """Supported task types."""

    CROSS_LINE = "cross_line"
    REGION_INTRUSION = "region_intrusion"
    FACE_DETECTION = "face_detection"
    CROWD_DENSITY = "crowd_density"
    DETECT = "detect"  # Generic detection

    def __str__(self) -> str:
        return self.value


class ErrorCode(IntEnum):
    """Error codes for API responses."""

    SUCCESS = 0
    UNKNOWN_ERROR = 1
    INVALID_REQUEST = 100
    MISSING_PARAMETER = 101
    INVALID_PARAMETER = 102
    TASK_NOT_FOUND = 200
    TASK_ALREADY_EXISTS = 201
    TASK_START_FAILED = 202
    TASK_STOP_FAILED = 203
    TASK_PAUSE_FAILED = 204
    TASK_RESUME_FAILED = 205
    CAMERA_NOT_FOUND = 300
    CAMERA_CONNECTION_FAILED = 301
    STREAM_ERROR = 302
    MODEL_NOT_FOUND = 400
    MODEL_LOAD_FAILED = 401
    INFERENCE_ERROR = 402
    ENGINE_NOT_INITIALIZED = 500
    ENGINE_START_FAILED = 501
    RESOURCE_EXHAUSTED = 600


# Error messages
ERROR_MESSAGES = {
    ErrorCode.SUCCESS: "Success",
    ErrorCode.UNKNOWN_ERROR: "Unknown error",
    ErrorCode.INVALID_REQUEST: "Invalid request",
    ErrorCode.MISSING_PARAMETER: "Missing required parameter",
    ErrorCode.INVALID_PARAMETER: "Invalid parameter value",
    ErrorCode.TASK_NOT_FOUND: "Task not found",
    ErrorCode.TASK_ALREADY_EXISTS: "Task already exists",
    ErrorCode.TASK_START_FAILED: "Failed to start task",
    ErrorCode.TASK_STOP_FAILED: "Failed to stop task",
    ErrorCode.TASK_PAUSE_FAILED: "Failed to pause task",
    ErrorCode.TASK_RESUME_FAILED: "Failed to resume task",
    ErrorCode.CAMERA_NOT_FOUND: "Camera not found",
    ErrorCode.CAMERA_CONNECTION_FAILED: "Failed to connect to camera",
    ErrorCode.STREAM_ERROR: "Stream error",
    ErrorCode.MODEL_NOT_FOUND: "Model not found",
    ErrorCode.MODEL_LOAD_FAILED: "Failed to load model",
    ErrorCode.INFERENCE_ERROR: "Inference error",
    ErrorCode.ENGINE_NOT_INITIALIZED: "Engine not initialized",
    ErrorCode.ENGINE_START_FAILED: "Failed to start engine",
    ErrorCode.RESOURCE_EXHAUSTED: "Resource exhausted",
}


def get_error_message(code: ErrorCode) -> str:
    """Get error message for error code.

    Args:
        code: Error code.

    Returns:
        Error message string.
    """
    return ERROR_MESSAGES.get(code, "Unknown error")


# Task status codes for Algorithm Warehouse
class WarehouseTaskStatus(IntEnum):
    """Task status codes for Algorithm Warehouse reporting."""

    PAUSED = 1
    RUNNING = 4


# Frame processing constants
DEFAULT_QUEUE_SIZE = 50
DEFAULT_FRAME_EXPIRED_TIME = 5.0  # seconds
DEFAULT_STREAM_RECONNECT_DELAY = 5.0  # seconds

# Model constants
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300

# Capacity constants (for warehouse reporting)
MAX_TASK_CAPACITY = 8
TOTAL_CAPABILITY = 1000
CAPABILITY_PER_TASK = 10
