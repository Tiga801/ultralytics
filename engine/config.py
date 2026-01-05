"""Global engine configuration.

This module provides the EngineConfig class for managing global
engine settings and parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EngineConfig:
    """Global configuration for the engine system.

    This class holds all global settings that affect the engine behavior.
    """

    # Service settings
    service_name: str = "easyair-common"
    service_port: int = 8666
    debug: bool = False

    # Directory settings
    weights_dir: str = "weights"
    logs_dir: str = "logs"
    runs_dir: str = "runs"

    # Stream settings
    frame_queue_size: int = 50
    frame_expired_time: float = 5.0  # seconds
    default_fps: int = 25

    # Inference settings
    default_device: str = "cuda"
    default_confidence: float = 0.25
    default_iou: float = 0.45

    # Cleanup settings
    file_expired_days: int = 3
    cleanup_interval_hours: int = 1

    # Algorithm warehouse settings
    warehouse_enabled: bool = True
    warehouse_sync_interval: int = 60  # seconds

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure directories exist
        for dir_attr in ['weights_dir', 'logs_dir', 'runs_dir']:
            dir_path = Path(getattr(self, dir_attr))
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            EngineConfig instance.
        """
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "service_name": self.service_name,
            "service_port": self.service_port,
            "debug": self.debug,
            "weights_dir": self.weights_dir,
            "logs_dir": self.logs_dir,
            "runs_dir": self.runs_dir,
            "frame_queue_size": self.frame_queue_size,
            "frame_expired_time": self.frame_expired_time,
            "default_fps": self.default_fps,
            "default_device": self.default_device,
            "default_confidence": self.default_confidence,
            "default_iou": self.default_iou,
            "file_expired_days": self.file_expired_days,
            "cleanup_interval_hours": self.cleanup_interval_hours,
            "warehouse_enabled": self.warehouse_enabled,
            "warehouse_sync_interval": self.warehouse_sync_interval,
        }

    def get_model_path(self, model_name: str) -> str:
        """Get full path to a model file.

        Args:
            model_name: Model filename.

        Returns:
            Full path to model.
        """
        return str(Path(self.weights_dir) / model_name)


# Global config instance
_engine_config: Optional[EngineConfig] = None


def get_engine_config() -> EngineConfig:
    """Get global engine configuration.

    Returns:
        EngineConfig instance.
    """
    global _engine_config
    if _engine_config is None:
        _engine_config = EngineConfig()
    return _engine_config


def set_engine_config(config: EngineConfig) -> None:
    """Set global engine configuration.

    Args:
        config: EngineConfig instance.
    """
    global _engine_config
    _engine_config = config


def init_engine_config(**kwargs) -> EngineConfig:
    """Initialize global engine configuration.

    Args:
        **kwargs: Configuration parameters.

    Returns:
        EngineConfig instance.
    """
    global _engine_config
    _engine_config = EngineConfig(**kwargs)
    return _engine_config
