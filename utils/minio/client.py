# -*- coding: utf-8 -*-
"""MinIO Client Module.

Provides a process-safe MinIO client with asynchronous upload queue.
Each task process should create its own MinIOClient instance.
"""

import logging
import os
import queue
import threading
import time
import uuid
import cv2
import numpy as np
from io import BytesIO
from typing import Callable, Optional

from minio import Minio
from minio.error import S3Error
from .config import MinIOConfig
from .stats import MinIOStatistics
from .tasks import UploadResult, UploadTask, UploadType
from .utils import build_object_url, get_content_type

# Type aliases for callbacks
SuccessCallback = Callable[[UploadResult], None]
FailureCallback = Callable[[UploadResult], None]


class MinIOClient:
    """Process-safe MinIO client with asynchronous upload queue.

    Designed for per-process usage in a multi-process architecture.
    Each task process should create and manage its own MinIOClient instance.
    NO SINGLETON PATTERN - instantiate directly.

    Features:
    - Background upload thread for non-blocking operations
    - OpenCV image encoding and upload
    - Video file upload support
    - Thread-safe statistics tracking
    - Dedicated logging to logs/minio.log
    - Configurable success/failure callbacks
    - Context manager support for clean lifecycle

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

        Manual lifecycle management::

            client = MinIOClient(config)
            client.start()
            try:
                url = client.upload_image(image, "path/to/image.jpg")
            finally:
                client.stop()
    """

    def __init__(
        self,
        config: Optional[MinIOConfig] = None,
        log_file: str = "logs/minio.log",
        on_success: Optional[SuccessCallback] = None,
        on_failure: Optional[FailureCallback] = None,
    ):
        """Initialize MinIO client.

        Args:
            config: MinIO configuration. Uses default configuration if None.
            log_file: Path to dedicated MinIO log file.
            on_success: Callback invoked on successful upload.
            on_failure: Callback invoked on failed upload.

        Raises:
            ImportError: If minio library is not installed.
        """
        self._config = config or MinIOConfig()
        self._config.validate()

        # Generate unique client identifier
        self._client_id = f"minio_{uuid.uuid4().hex[:8]}"

        # MinIO client instance
        self._client: Optional[Minio] = None

        # Upload queue for async operations
        self._upload_queue: queue.Queue = queue.Queue(
            maxsize=self._config.queue_max_size
        )

        # Threading controls
        self._upload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics with thread-safe access
        self._stats = MinIOStatistics()
        self._stats_lock = threading.Lock()

        # Callbacks
        self._on_success = on_success
        self._on_failure = on_failure

        # Setup dedicated logging
        self._logger = self._setup_logger(log_file)
        self._log_info(
            f"Client initialized - endpoint: {self._config.endpoint}, "
            f"bucket: {self._config.bucket_name}"
        )

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Create dedicated MinIO logger.

        Creates a logger that writes exclusively to the specified log file
        and does not propagate to root logger (no console output).

        Args:
            log_file: Path to log file.

        Returns:
            Configured logger instance.
        """
        # Create unique logger name to avoid conflicts between instances
        logger = logging.getLogger(f"minio.{self._client_id}")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # File handler
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [MINIO] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console if file logging fails
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
            logger.warning(f"Failed to setup file logging: {e}")

        # Prevent propagation to root logger (no console output)
        logger.propagate = False

        return logger

    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)

    def _log_info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log error message."""
        self._logger.error(message)

    # === Lifecycle Methods ===

    def start(self) -> bool:
        """Initialize MinIO connection and start background upload thread.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            self._log_warning("Client is already running")
            return True

        try:
            # Initialize MinIO client
            if not self._init_client():
                self._log_error("Failed to initialize MinIO client")
                return False

            # Start upload thread
            self._stop_event.clear()
            self._running = True
            self._upload_thread = threading.Thread(
                target=self._upload_loop,
                daemon=True,
                name=f"MinIOUploader-{self._client_id}",
            )
            self._upload_thread.start()

            self._log_info("Client started successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to start client: {e}")
            self._running = False
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Stop upload thread and release resources.

        Attempts to drain remaining queue items before stopping.

        Args:
            timeout: Maximum time in seconds to wait for thread to stop.
        """
        self._log_info("Stopping client...")

        self._running = False
        self._stop_event.set()

        # Send poison pill to wake up blocked queue.get()
        try:
            self._upload_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for upload thread to finish
        if self._upload_thread and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=timeout)
            if self._upload_thread.is_alive():
                self._log_warning("Upload thread did not stop gracefully")

        self._client = None
        self._log_info("Client stopped")

    # === Image Upload Methods ===

    def upload_image(
        self,
        image: "np.ndarray",
        object_name: str,
        quality: Optional[int] = None,
    ) -> Optional[str]:
        """Queue an OpenCV image for upload.

        The image is immediately encoded to JPEG and queued for
        background upload. Returns the expected URL immediately.

        Args:
            image: OpenCV image as numpy.ndarray (BGR format).
            object_name: Target path in bucket (e.g., "events/2025/12/24/img.jpg").
            quality: JPEG quality 1-100. Uses config default if None.

        Returns:
            Expected URL if queued successfully, None if queue full or error.
        """
        if not self._running:
            self._log_warning("Cannot upload: client not running")
            return None

        if image is None:
            self._log_error("Image is None")
            return None

        try:
            # Encode image to JPEG
            jpeg_quality = quality if quality is not None else self._config.jpeg_quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            success, buffer = cv2.imencode(".jpg", image, encode_param)

            if not success:
                self._log_error("Image encoding failed")
                return None

            # Create upload task
            data = buffer.tobytes()
            task = UploadTask(
                object_name=object_name,
                upload_type=UploadType.IMAGE,
                data=data,
                content_type="image/jpeg",
                jpeg_quality=jpeg_quality,
            )

            return self._enqueue_task(task)

        except Exception as e:
            self._log_error(f"Image upload failed: {e}")
            return None

    # === Video Upload Methods ===

    def upload_video(
        self,
        video_path: str,
        object_name: str,
    ) -> Optional[str]:
        """Queue a video file for upload.

        Reads the video file and queues it for background upload.

        Args:
            video_path: Local path to video file.
            object_name: Target path in bucket.

        Returns:
            Expected URL if queued successfully, None if error.
        """
        if not self._running:
            self._log_warning("Cannot upload: client not running")
            return None

        if not os.path.exists(video_path):
            self._log_error(f"Video file not found: {video_path}")
            return None

        try:
            # Read video file
            with open(video_path, "rb") as f:
                data = f.read()

            # Determine content type from extension
            content_type = get_content_type(video_path)

            # Create upload task
            task = UploadTask(
                object_name=object_name,
                upload_type=UploadType.VIDEO,
                data=data,
                content_type=content_type,
            )

            return self._enqueue_task(task)

        except Exception as e:
            self._log_error(f"Video upload failed: {e}")
            return None

    def upload_video_bytes(
        self,
        video_data: bytes,
        object_name: str,
        content_type: str = "video/mp4",
    ) -> Optional[str]:
        """Queue video bytes for upload.

        Useful for uploading pre-encoded video data directly.

        Args:
            video_data: Raw video bytes (already encoded).
            object_name: Target path in bucket.
            content_type: MIME type of video.

        Returns:
            Expected URL if queued successfully, None if error.
        """
        if not self._running:
            self._log_warning("Cannot upload: client not running")
            return None

        try:
            task = UploadTask(
                object_name=object_name,
                upload_type=UploadType.VIDEO,
                data=video_data,
                content_type=content_type,
            )

            return self._enqueue_task(task)

        except Exception as e:
            self._log_error(f"Video bytes upload failed: {e}")
            return None

    # === Generic Upload Methods ===

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Queue raw bytes for upload.

        Args:
            data: Bytes to upload.
            object_name: Target path in bucket.
            content_type: MIME type.

        Returns:
            Expected URL if queued successfully, None if error.
        """
        if not self._running:
            self._log_warning("Cannot upload: client not running")
            return None

        try:
            task = UploadTask(
                object_name=object_name,
                upload_type=UploadType.BYTES,
                data=data,
                content_type=content_type,
            )

            return self._enqueue_task(task)

        except Exception as e:
            self._log_error(f"Bytes upload failed: {e}")
            return None

    # === Status & Statistics ===

    def is_running(self) -> bool:
        """Check if upload thread is running.

        Returns:
            True if running, False otherwise.
        """
        return self._running

    def is_connected(self) -> bool:
        """Check if connected to MinIO server.

        Returns:
            True if client is initialized, False otherwise.
        """
        return self._client is not None

    def get_queue_size(self) -> int:
        """Get current number of pending uploads.

        Returns:
            Number of tasks waiting in the queue.
        """
        return self._upload_queue.qsize()

    def get_statistics(self) -> MinIOStatistics:
        """Get a snapshot copy of current statistics.

        Returns a copy that is safe to use without locks.

        Returns:
            Copy of current MinIOStatistics.
        """
        with self._stats_lock:
            self._stats.uploads_queued = self._upload_queue.qsize()
            return self._stats.copy()

    def get_object_url(self, object_name: str) -> str:
        """Build the public URL for an object.

        Args:
            object_name: Object path in bucket.

        Returns:
            Complete URL for accessing the object.
        """
        return build_object_url(
            host=self._config.resolved_public_host,
            port=self._config.resolved_public_port,
            bucket_name=self._config.bucket_name,
            object_name=object_name,
            secure=self._config.secure,
        )

    # === Context Manager ===

    def __enter__(self) -> "MinIOClient":
        """Enter context manager, start client."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, stop client."""
        self.stop()

    # === Private Methods ===

    def _init_client(self) -> bool:
        """Initialize MinIO client and verify bucket exists.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self._client = Minio(
                endpoint=self._config.endpoint,
                access_key=self._config.access_key,
                secret_key=self._config.secret_key,
                secure=self._config.secure,
            )

            # Check if bucket exists, create if not
            if not self._client.bucket_exists(self._config.bucket_name):
                self._client.make_bucket(self._config.bucket_name)
                self._log_info(f"Created bucket: {self._config.bucket_name}")

            self._log_info("MinIO client initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"MinIO client initialization failed: {e}")
            self._client = None
            return False

    def _enqueue_task(self, task: UploadTask) -> Optional[str]:
        """Add upload task to queue.

        Args:
            task: Upload task to enqueue.

        Returns:
            Expected URL if queued successfully, None if queue full.
        """
        try:
            self._upload_queue.put_nowait(task)

            # Build expected URL
            url = self.get_object_url(task.object_name)
            self._log_debug(f"Queued upload: {task.object_name}")

            return url

        except queue.Full:
            self._log_warning(
                f"Upload queue full ({self._config.queue_max_size}), "
                f"task dropped: {task.object_name}"
            )
            with self._stats_lock:
                self._stats.uploads_total += 1
                self._stats.uploads_failed += 1
            return None

    def _upload_loop(self) -> None:
        """Background thread loop for processing upload queue."""
        self._log_debug("Upload loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop event
                task = self._upload_queue.get(timeout=1.0)

                # Check for poison pill
                if task is None:
                    continue

                if isinstance(task, UploadTask):
                    result = self._process_task(task)
                    self._invoke_callback(result)

            except queue.Empty:
                continue
            except Exception as e:
                self._log_error(f"Upload loop error: {e}")

        self._log_debug("Upload loop stopped")

    def _process_task(self, task: UploadTask) -> UploadResult:
        """Process a single upload task.

        Args:
            task: Upload task to process.

        Returns:
            Upload result with success/failure status.
        """
        start_time = time.time()

        try:
            # Get data as bytes
            if isinstance(task.data, bytes):
                data = task.data
            else:
                # Should not happen with proper task creation
                self._log_error(f"Invalid data type: {type(task.data)}")
                return UploadResult(
                    success=False,
                    object_name=task.object_name,
                    error="Invalid data type",
                    task_id=task.task_id,
                )

            # Upload to MinIO
            data_stream = BytesIO(data)
            self._client.put_object(
                bucket_name=self._config.bucket_name,
                object_name=task.object_name,
                data=data_stream,
                length=len(data),
                content_type=task.content_type,
            )

            upload_time = time.time() - start_time
            url = self.get_object_url(task.object_name)

            # Update statistics
            self._update_stats_success(len(data), task.upload_type)

            # Format file size for logging
            size_kb = len(data) / 1024
            self._log_info(
                f"Upload success: {task.object_name} ({size_kb:.1f}KB in {upload_time:.2f}s)"
            )

            return UploadResult(
                success=True,
                object_name=task.object_name,
                url=url,
                upload_time=upload_time,
                file_size=len(data),
                content_type=task.content_type,
                task_id=task.task_id,
            )

        except S3Error as e:
            upload_time = time.time() - start_time
            error_msg = f"S3Error: {e.code} - {e.message}"
            self._log_error(f"Upload failed: {task.object_name} - {error_msg}")
            self._update_stats_failure()

            return UploadResult(
                success=False,
                object_name=task.object_name,
                error=error_msg,
                upload_time=upload_time,
                task_id=task.task_id,
            )

        except Exception as e:
            upload_time = time.time() - start_time
            error_msg = str(e)
            self._log_error(f"Upload failed: {task.object_name} - {error_msg}")
            self._update_stats_failure()

            return UploadResult(
                success=False,
                object_name=task.object_name,
                error=error_msg,
                upload_time=upload_time,
                task_id=task.task_id,
            )

    def _update_stats_success(self, file_size: int, upload_type: UploadType) -> None:
        """Update statistics for successful upload.

        Args:
            file_size: Size of uploaded data in bytes.
            upload_type: Type of upload (IMAGE, VIDEO, BYTES).
        """
        with self._stats_lock:
            self._stats.uploads_total += 1
            self._stats.uploads_success += 1
            self._stats.bytes_uploaded += file_size
            self._stats.last_upload_time = time.time()

            if upload_type == UploadType.IMAGE:
                self._stats.images_uploaded += 1
            elif upload_type == UploadType.VIDEO:
                self._stats.videos_uploaded += 1

    def _update_stats_failure(self) -> None:
        """Update statistics for failed upload."""
        with self._stats_lock:
            self._stats.uploads_total += 1
            self._stats.uploads_failed += 1

    def _invoke_callback(self, result: UploadResult) -> None:
        """Invoke appropriate callback based on result.

        Protected against callback exceptions.

        Args:
            result: Upload result to pass to callback.
        """
        try:
            if result.success and self._on_success:
                self._on_success(result)
            elif not result.success and self._on_failure:
                self._on_failure(result)
        except Exception as e:
            self._log_warning(f"Callback error: {e}")
