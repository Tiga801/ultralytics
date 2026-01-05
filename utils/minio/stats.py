# -*- coding: utf-8 -*-
"""MinIO Statistics Module.

Provides statistics tracking for MinIO upload operations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MinIOStatistics:
    """MinIO client statistics for monitoring.

    Tracks upload metrics including success/failure counts, bytes transferred,
    and timing information. Thread-safe when accessed through MinIOClient methods.

    Attributes:
        uploads_total: Total number of upload attempts.
        uploads_success: Number of successful uploads.
        uploads_failed: Number of failed uploads.
        uploads_queued: Current number of tasks in queue.
        bytes_uploaded: Total bytes successfully uploaded.
        last_upload_time: Timestamp of last successful upload.
        images_uploaded: Number of images uploaded.
        videos_uploaded: Number of videos uploaded.
    """

    uploads_total: int = 0
    uploads_success: int = 0
    uploads_failed: int = 0
    uploads_queued: int = 0
    bytes_uploaded: int = 0
    last_upload_time: Optional[float] = None
    images_uploaded: int = 0
    videos_uploaded: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate upload success rate.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 0.0 if no uploads have been attempted.
        """
        if self.uploads_total == 0:
            return 0.0
        return self.uploads_success / self.uploads_total

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.uploads_total = 0
        self.uploads_success = 0
        self.uploads_failed = 0
        self.uploads_queued = 0
        self.bytes_uploaded = 0
        self.last_upload_time = None
        self.images_uploaded = 0
        self.videos_uploaded = 0

    def copy(self) -> "MinIOStatistics":
        """Create a snapshot copy of current statistics.

        Returns:
            New MinIOStatistics instance with copied values.
        """
        return MinIOStatistics(
            uploads_total=self.uploads_total,
            uploads_success=self.uploads_success,
            uploads_failed=self.uploads_failed,
            uploads_queued=self.uploads_queued,
            bytes_uploaded=self.bytes_uploaded,
            last_upload_time=self.last_upload_time,
            images_uploaded=self.images_uploaded,
            videos_uploaded=self.videos_uploaded,
        )
