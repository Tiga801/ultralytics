# -*- coding: utf-8 -*-
"""MinIO Upload Task Module.

Provides data structures for upload tasks and results.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class UploadType(Enum):
    """Upload content type enumeration.

    Defines the type of content being uploaded, which determines
    how the data is processed before upload.
    """

    IMAGE = "image"  # OpenCV numpy.ndarray -> JPEG encoded
    VIDEO = "video"  # Video file path or bytes
    BYTES = "bytes"  # Raw bytes with content type


@dataclass
class UploadTask:
    """Upload task descriptor for the async queue.

    Represents a single upload task with all necessary information
    for the background worker to process it.

    Attributes:
        object_name: Target object path in MinIO bucket.
        upload_type: Type of upload (IMAGE, VIDEO, BYTES).
        data: The data to upload:
            - IMAGE: numpy.ndarray (OpenCV image, BGR format)
            - VIDEO: str (file path) or bytes (encoded video data)
            - BYTES: bytes (raw data)
        content_type: MIME type for the upload.
        jpeg_quality: JPEG quality for IMAGE type (1-100).
        task_id: Unique identifier for this upload task.
        created_at: Timestamp when task was created.
    """

    object_name: str
    upload_type: UploadType
    data: Any  # numpy.ndarray, str (path), or bytes
    content_type: str = "application/octet-stream"
    jpeg_quality: int = 95
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)


@dataclass
class UploadResult:
    """Result of an upload operation.

    Contains the outcome of an upload attempt, including success/failure
    status, URL, error information, and performance metrics.

    Attributes:
        success: Whether the upload succeeded.
        object_name: The target object name.
        url: The accessible URL (on success).
        error: Error message (on failure).
        upload_time: Time taken for upload in seconds.
        file_size: Size of uploaded data in bytes.
        content_type: MIME type of uploaded content.
        task_id: The original task identifier.
    """

    success: bool
    object_name: str
    url: Optional[str] = None
    error: Optional[str] = None
    upload_time: float = 0.0
    file_size: int = 0
    content_type: str = "application/octet-stream"
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of upload result.
        """
        return {
            "success": self.success,
            "object_name": self.object_name,
            "url": self.url,
            "error": self.error,
            "upload_time": self.upload_time,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "task_id": self.task_id,
        }
