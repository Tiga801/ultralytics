# -*- coding: utf-8 -*-
"""MinIO Utility Functions.

Provides URL building, content type detection, and object name generation.
"""

import uuid
from datetime import datetime
from typing import Optional


# MIME type mapping for common file extensions
CONTENT_TYPE_MAP = {
    # Images
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "ico": "image/x-icon",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    # Videos
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "wmv": "video/x-ms-wmv",
    "flv": "video/x-flv",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "m4v": "video/x-m4v",
    "3gp": "video/3gpp",
    "ts": "video/mp2t",
    # Audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
    # Documents
    "pdf": "application/pdf",
    "json": "application/json",
    "xml": "application/xml",
    "txt": "text/plain",
    "html": "text/html",
    "css": "text/css",
    "js": "application/javascript",
    "csv": "text/csv",
    # Archives
    "zip": "application/zip",
    "tar": "application/x-tar",
    "gz": "application/gzip",
    "rar": "application/x-rar-compressed",
    "7z": "application/x-7z-compressed",
}


def build_object_url(
    host: str,
    port: int,
    bucket_name: str,
    object_name: str,
    secure: bool = False,
) -> str:
    """Build complete object access URL.

    Constructs a URL that can be used to access an object in MinIO.

    Args:
        host: MinIO server host.
        port: MinIO server port.
        bucket_name: Bucket name.
        object_name: Object path within bucket.
        secure: Use HTTPS if True, HTTP if False.

    Returns:
        Complete URL like http://host:port/bucket/object_name
    """
    protocol = "https" if secure else "http"
    return f"{protocol}://{host}:{port}/{bucket_name}/{object_name}"


def get_content_type(filename: str) -> str:
    """Determine MIME type from filename extension.

    Args:
        filename: Filename or path with extension.

    Returns:
        MIME type string. Returns 'application/octet-stream' if unknown.
    """
    if not filename or "." not in filename:
        return "application/octet-stream"

    ext = filename.rsplit(".", 1)[-1].lower()
    return CONTENT_TYPE_MAP.get(ext, "application/octet-stream")


def generate_object_name(
    prefix: str = "",
    extension: str = "jpg",
    include_date: bool = True,
) -> str:
    """Generate unique object name with optional date-based path.

    Creates a unique object name using UUID, optionally organized
    by date directory structure.

    Args:
        prefix: Path prefix (e.g., "events", "detections").
        extension: File extension without dot.
        include_date: Include YYYY/MM/DD in path.

    Returns:
        Object name like "prefix/2025/12/24/uuid.jpg"
    """
    parts = []

    if prefix:
        # Remove leading/trailing slashes
        prefix = prefix.strip("/")
        parts.append(prefix)

    if include_date:
        now = datetime.now()
        parts.append(now.strftime("%Y/%m/%d"))

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:16]
    filename = f"{unique_id}.{extension.lstrip('.')}"
    parts.append(filename)

    return "/".join(parts)


def generate_timestamped_name(
    prefix: str = "",
    extension: str = "jpg",
    task_id: Optional[str] = None,
    track_id: Optional[int] = None,
) -> str:
    """Generate timestamped object name for ordered uploads.

    Creates a filename with timestamp for chronological ordering,
    useful for detection results and tracking data.

    Args:
        prefix: Path prefix.
        extension: File extension without dot.
        task_id: Optional task identifier to include.
        track_id: Optional track identifier to include.

    Returns:
        Object name like "prefix/20251224_103045_123_task001_track005.jpg"
    """
    parts = []

    if prefix:
        prefix = prefix.strip("/")
        parts.append(prefix)

    # Generate timestamp-based filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:18]  # YYYYMMDD_HHMMSS_mmm

    name_parts = [timestamp]

    if task_id:
        name_parts.append(f"task{task_id}")

    if track_id is not None:
        name_parts.append(f"track{track_id:04d}")

    filename = "_".join(name_parts) + f".{extension.lstrip('.')}"
    parts.append(filename)

    return "/".join(parts)
