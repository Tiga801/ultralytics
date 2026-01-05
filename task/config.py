"""Task configuration management.

This module provides TaskConfig dataclass and TaskConfigManager for managing
task configurations throughout the system.
"""

import json
import threading
from dataclasses import dataclass, field
<<<<<<< HEAD
from typing import Any, Callable, Dict, List, Optional, Tuple
=======
from typing import Any, Callable, Dict, List, Optional
>>>>>>> 07331326 (feat: build video analytics task management system)

from utils import SingletonMeta


@dataclass
class TaskConfig:
    """Configuration for a single task.

    This dataclass holds all configuration parameters needed to run a task,
    including device info, model settings, and task-specific parameters.
    """

    # Required fields
    task_id: str
    task_name: str
    task_type: str  # 'cross_line', 'region_intrusion', etc.

    # Device info
    camera_id: str
    camera_name: str
    stand_name: str = "default"
    rtsp_url: str = ""

    # Model settings
    model_name: str = "yolo11m.pt"
    model_path: Optional[str] = None
    device: str = "cuda"
    confidence: float = 0.25
    iou_threshold: float = 0.45

    # Processing parameters
    rtsp_interval: float = 0.1  # Frame grab interval (seconds)
    frame_skip: int = 1         # Process every Nth frame

    # Task-specific parameters (JSON dict)
    config_params: Dict[str, Any] = field(default_factory=dict)

    # Execution period
    exec_period: Optional[Dict[str, str]] = None

    # Reporting (future use)
    mqtt_topic: Optional[str] = None
    minio_bucket: Optional[str] = None

    # Actual stream properties (set at runtime by CameraEngine)
    # These override configured values when available
    actual_width: Optional[int] = None
    actual_height: Optional[int] = None
    actual_fps: Optional[float] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Set default model path if not specified
        if self.model_path is None:
            self.model_path = f"weights/{self.model_name}"

    @classmethod
    def from_api_request(cls, data: Dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from API request data.

        Args:
            data: API request data (AnalyseCondition format).

        Returns:
            TaskConfig instance.

        Example:
            >>> data = {
            ...     "taskID": "cross_line_001",
            ...     "taskName": "cross_line-1",
            ...     "deviceInfo": {
            ...         "deviceCode": "camera_001",
            ...         "deviceName": "Camera-001",
            ...         "sourceRTSP": "rtsp://..."
            ...     },
            ...     "configParam": {
            ...         "task": "cross_line",
            ...         "model": "yolo11m.pt"
            ...     }
            ... }
            >>> config = TaskConfig.from_api_request(data)
        """
        device_info = data.get("deviceInfo", {})
        config_param = data.get("configParam", {})

        # Parse execution period
        exec_period = None
        exec_period_str = data.get("execPeriodTime")
        if exec_period_str:
            try:
                exec_period = json.loads(exec_period_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract stand name from device code or use default
        device_code = device_info.get("deviceCode", "")
        stand_name = data.get("standName", "default")

        return cls(
            stand_name=stand_name,
            camera_id=device_code,
            camera_name=device_info.get("deviceName", ""),
            task_id=data.get("taskID", ""),
            task_name=data.get("taskName", ""),
            task_type=config_param.get("task", "detect"),
            
            rtsp_url=device_info.get("sourceRTSP", ""),

            model_name=config_param.get("model", "yolo11m.pt"),
            model_path=config_param.get("model_path"),
            device=config_param.get("device", "cuda"),
            
            confidence=config_param.get("confidence", 0.25),
            iou_threshold=config_param.get("iou", 0.45),
            rtsp_interval=config_param.get("rtsp_interval", 0.1),
            frame_skip=config_param.get("frame_skip", 1),
            config_params=config_param,
            exec_period=exec_period,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "stand_name": self.stand_name,
            "rtsp_url": self.rtsp_url,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "confidence": self.confidence,
            "iou_threshold": self.iou_threshold,
            "rtsp_interval": self.rtsp_interval,
            "frame_skip": self.frame_skip,
            "config_params": self.config_params,
            "exec_period": self.exec_period,
            # Actual stream properties (set at runtime)
            "actual_width": self.actual_width,
            "actual_height": self.actual_height,
            "actual_fps": self.actual_fps,
        }

    def get_resolution_width(self) -> int:
        """Get resolution width (actual if available, otherwise -1).

        Returns:
            Resolution width in pixels.
        """
        return self.actual_width if self.actual_width is not None else -1

    def get_resolution_height(self) -> int:
        """Get resolution height (actual if available, otherwise -1).

        Returns:
            Resolution height in pixels.
        """
        return self.actual_height if self.actual_height is not None else -1

    def get_stream_fps(self) -> float:
        """Get stream FPS (actual if available, otherwise -1).

        Returns:
            Frames per second.
        """
        return self.actual_fps if self.actual_fps is not None else -1

    def update_actual_stream_params(
        self,
        width: int,
        height: int,
        fps: float,
        stream_log: Optional[Callable[[str], None]] = None
    ) -> None:
        """Update actual stream parameters from runtime detection.

        Compares with configured values and logs warning if different.
        When mismatch occurs, actual values are used instead of configured.

        Args:
            width: Actual stream width from video capture.
            height: Actual stream height from video capture.
            fps: Actual stream FPS from video capture.
            stream_log: Optional logger for streams.log warnings.
        """
        # Store actual values
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps

    def get_extra(self, key: str, default: Any = None) -> Any:
        """Get an extra configuration parameter.

        Checks config_params dict first, then falls back to direct attributes.

        Args:
            key: Parameter name.
            default: Default value if not found.

        Returns:
            Parameter value or default.
        """
        # Check config_params first (for task-specific params)
        if key in self.config_params:
            return self.config_params[key]

        # Map common aliases
        key_mapping = {
            "iou": "iou_threshold",
        }
        attr_name = key_mapping.get(key, key)

        # Check direct attributes
        if hasattr(self, attr_name):
            return getattr(self, attr_name)

        return default

    @property
    def areas_info(self) -> List[Dict[str, Any]]:
        """Get areas/regions configuration for the task."""
        return self.config_params.get("areasInfo", [])

<<<<<<< HEAD
=======
    def get_roi_areas_info(self, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """Get ROI regions converted to pixel coordinates.

        This method parses roi_1 through roi_5 from config_params and converts
        normalized coordinates (0.0-1.0) to pixel coordinates based on actual
        frame dimensions.

        The result is cached to avoid repeated conversion on subsequent calls.

        Args:
            frame_width: Actual video frame width in pixels.
            frame_height: Actual video frame height in pixels.

        Returns:
            List of region dicts with pixel coordinates, merged with any
            existing areasInfo for backward compatibility.
        """
        # Check if already converted (cached)
        cache_key = "_roi_areas_info_cached"
        cached = self.config_params.get(cache_key)
        if cached is not None:
            return cached

        # Import here to avoid circular dependency
        from solutions.utils.roi import parse_roi_from_api

        # Parse and convert ROIs from roi_1 through roi_5
        roi_areas = parse_roi_from_api(
            self.config_params,
            frame_width,
            frame_height
        )

        # Merge with any existing areasInfo (for backward compatibility)
        existing = self.config_params.get("areasInfo", [])
        if existing:
            roi_areas.extend(existing)

        # Cache the result
        self.config_params[cache_key] = roi_areas

        return roi_areas

>>>>>>> 07331326 (feat: build video analytics task management system)

class TaskConfigManager(metaclass=SingletonMeta):
    """Centralized task configuration storage (Singleton).

    This class manages all task configurations in the system, providing
    thread-safe access for adding, retrieving, and removing configs.
    """

    def __init__(self):
        """Initialize the config manager."""
        self._configs: Dict[str, TaskConfig] = {}
        self._lock = threading.RLock()

    def add_config(self, config: TaskConfig) -> None:
        """Add or update a task configuration.

        Args:
            config: TaskConfig to add.
        """
        with self._lock:
            self._configs[config.task_id] = config

    def get_config(self, task_id: str) -> Optional[TaskConfig]:
        """Get a task configuration by ID.

        Args:
            task_id: Task identifier.

        Returns:
            TaskConfig or None if not found.
        """
        with self._lock:
            return self._configs.get(task_id)

    def remove_config(self, task_id: str) -> bool:
        """Remove a task configuration.

        Args:
            task_id: Task identifier.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if task_id in self._configs:
                del self._configs[task_id]
                return True
            return False

    def get_all_task_ids(self) -> List[str]:
        """Get all task IDs.

        Returns:
            List of task IDs.
        """
        with self._lock:
            return list(self._configs.keys())

    def get_all_configs(self) -> List[TaskConfig]:
        """Get all task configurations.

        Returns:
            List of TaskConfig objects.
        """
        with self._lock:
            return list(self._configs.values())

    def get_configs_by_camera(self, camera_id: str) -> List[TaskConfig]:
        """Get all configurations for a specific camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            List of TaskConfig objects for the camera.
        """
        with self._lock:
            return [c for c in self._configs.values() if c.camera_id == camera_id]

    def get_configs_by_stand(self, stand_name: str) -> List[TaskConfig]:
        """Get all configurations for a specific stand.

        Args:
            stand_name: Stand name.

        Returns:
            List of TaskConfig objects for the stand.
        """
        with self._lock:
            return [c for c in self._configs.values() if c.stand_name == stand_name]

    def count(self) -> int:
        """Get total number of configurations.

        Returns:
            Number of configurations.
        """
        with self._lock:
            return len(self._configs)

    def clear(self) -> None:
        """Clear all configurations."""
        with self._lock:
            self._configs.clear()

    def build_config(self, data: Dict[str, Any]) -> TaskConfig:
        """Build and register a TaskConfig from API data.

        Args:
            data: API request data.

        Returns:
            Created TaskConfig.
        """
        config = TaskConfig.from_api_request(data)
        self.add_config(config)
        return config
