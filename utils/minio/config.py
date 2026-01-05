# -*- coding: utf-8 -*-
"""MinIO Configuration Module.

Provides type-safe configuration management for MinIO client connections.
Supports creation from dictionaries for integration with JSON configuration files.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MinIOConfig:
    """MinIO client configuration.

    Attributes:
        host: MinIO server hostname (without port).
        port: MinIO server port number.
        access_key: Authentication access key.
        secret_key: Authentication secret key.
        bucket_name: Target bucket for uploads.
        secure: Whether to use HTTPS for connections.
        public_host: Public-facing hostname for URL generation (optional).
        public_port: Public-facing port for URL generation (optional).
        queue_max_size: Maximum number of pending upload tasks in queue.
        upload_timeout: Timeout for individual upload operations in seconds.
        jpeg_quality: Default JPEG encoding quality (1-100).
    """

    # Connection settings
    host: str = "easyair-minio"
    port: int = 9000
    access_key: str = "ZXYrzg2D6madjXxX8u8T"
    secret_key: str = "AxbvSyYHDIarCTCYMueGVp68rCDSgs1w7JrsGgyk"

    # Storage settings
    bucket_name: str = "algorithm"
    secure: bool = False

    # Public URL settings (for generating accessible URLs)
    public_host: Optional[str] = None
    public_port: Optional[int] = None

    # Queue settings
    queue_max_size: int = 1000
    upload_timeout: float = 30.0

    # Image settings
    jpeg_quality: int = 100

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.host:
            raise ValueError("host cannot be empty")

        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")

        if not self.access_key:
            raise ValueError("access_key cannot be empty")

        if not self.secret_key:
            raise ValueError("secret_key cannot be empty")

        if not self.bucket_name:
            raise ValueError("bucket_name cannot be empty")

        if self.queue_max_size <= 0:
            raise ValueError(
                f"queue_max_size must be positive, got {self.queue_max_size}"
            )

        if self.upload_timeout <= 0:
            raise ValueError(
                f"upload_timeout must be positive, got {self.upload_timeout}"
            )

        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(
                f"jpeg_quality must be between 1 and 100, got {self.jpeg_quality}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "host": self.host,
            "port": self.port,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
            "bucket_name": self.bucket_name,
            "secure": self.secure,
            "public_host": self.public_host,
            "public_port": self.public_port,
            "queue_max_size": self.queue_max_size,
            "upload_timeout": self.upload_timeout,
            "jpeg_quality": self.jpeg_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinIOConfig":
        """Create configuration from dictionary.

        Supports legacy field names for backward compatibility:
        - 'endpoint' -> extracts host and port

        Args:
            data: Configuration dictionary.

        Returns:
            MinIOConfig instance.
        """
        # Handle legacy 'endpoint' field (format: "host:port")
        host = data.get("host")
        port = data.get("port")

        if host is None and "endpoint" in data:
            endpoint = data["endpoint"]
            if ":" in endpoint:
                parts = endpoint.rsplit(":", 1)
                host = parts[0]
                try:
                    port = int(parts[1])
                except ValueError:
                    port = 9000
            else:
                host = endpoint

        return cls(
            host=str(host) if host else "easyair-minio",
            port=int(port) if port else 9000,
            access_key=data.get("access_key", "ZXYrzg2D6madjXxX8u8T"),
            secret_key=data.get("secret_key", "AxbvSyYHDIarCTCYMueGVp68rCDSgs1w7JrsGgyk"),
            bucket_name=data.get("bucket_name", "algorithm"),
            secure=bool(data.get("secure", False)),
            public_host=data.get("public_host"),
            public_port=int(data["public_port"]) if data.get("public_port") else None,
            queue_max_size=int(data.get("queue_max_size", 1000)),
            upload_timeout=float(data.get("upload_timeout", 30.0)),
            jpeg_quality=int(data.get("jpeg_quality", 95)),
        )

    @property
    def endpoint(self) -> str:
        """Return endpoint in format 'host:port'.

        Returns:
            Endpoint string for MinIO client initialization.
        """
        return f"{self.host}:{self.port}"

    @property
    def public_endpoint(self) -> str:
        """Return public endpoint for URL generation.

        Falls back to standard endpoint if public settings not configured.

        Returns:
            Public endpoint string.
        """
        host = self.public_host or self.host
        port = self.public_port or self.port
        return f"{host}:{port}"

    @property
    def resolved_public_host(self) -> str:
        """Return resolved public host.

        Returns:
            Public host, falling back to host if not set.
        """
        return self.public_host or self.host

    @property
    def resolved_public_port(self) -> int:
        """Return resolved public port.

        Returns:
            Public port, falling back to port if not set.
        """
        return self.public_port or self.port
