# -*- coding: utf-8 -*-
"""MinIO Object Storage Module.

Provides a process-safe MinIO client with asynchronous upload queue for
images and videos. Designed for per-process usage in multi-process systems.

Features:
- Background upload thread for non-blocking operations
- OpenCV image encoding and upload (JPEG)
- H.264 encoded video upload
- Thread-safe statistics tracking
- Dedicated logging to logs/minio.log
- Configurable success/failure callbacks
- Context manager support

Example:
    Basic usage with context manager::

        from utils.minio import MinIOClient, MinIOConfig

        config = MinIOConfig(host="192.168.2.234", port=9000)
        with MinIOClient(config) as client:
            url = client.upload_image(image_array, "test/image.jpg")
            print(f"Uploaded to: {url}")

    Video upload::

        with MinIOClient() as client:
            url = client.upload_video("local/video.mp4", "remote/video.mp4")

    With callbacks::

        def on_success(result):
            print(f"Uploaded: {result.url}")

        def on_failure(result):
            print(f"Failed: {result.error}")

        client = MinIOClient(
            config=MinIOConfig(),
            on_success=on_success,
            on_failure=on_failure,
        )
"""

from .client import MinIOClient
from .config import MinIOConfig
from .stats import MinIOStatistics
from .tasks import UploadResult, UploadTask, UploadType
from .utils import build_object_url, generate_object_name, generate_timestamped_name, get_content_type

__all__ = [
    # Main client
    "MinIOClient",
    # Configuration
    "MinIOConfig",
    # Data structures
    "UploadTask",
    "UploadResult",
    "UploadType",
    "MinIOStatistics",
    # Utility functions
    "build_object_url",
    "get_content_type",
    "generate_object_name",
    "generate_timestamped_name",
]

__version__ = "1.0.0"
