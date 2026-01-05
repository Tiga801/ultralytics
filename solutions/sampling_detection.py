"""Video sampling detection task.

This module provides the SamplingDetectionTask class that performs
intelligent video frame sampling based on configurable strategies
and uploads captured frames to MinIO storage.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from task import TaskBase, TaskRegistry, TaskResult
from utils.minio import MinIOClient, MinIOConfig


@dataclass
class StrategyParamConfig:
    """Configuration for a single sampling strategy.

    Attributes:
        strategy_type: 3=full-frame sampling, 4=target-based sampling
        target_type: List of class indices to sample (e.g., [3, 9])
        param_value: Minimum confidence threshold (default 0.5)
        max_value: Maximum confidence threshold (default 1.0)
        output_content: "1"=full image, "2"=ROI image
        target_area: Scaling factor for ROI expansion (default 2.0)
        minimum_width: Minimum target width in pixels
        maximum_width: Maximum target width in pixels
        minimum_height: Minimum target height in pixels
        maximum_height: Maximum target height in pixels
    """

    strategy_type: int = 3
    target_type: List[int] = field(default_factory=list)
    param_value: float = 0.5
    max_value: float = 1.0
    output_content: str = "1"
    target_area: float = 2.0
    minimum_width: int = 0
    maximum_width: int = 10000
    minimum_height: int = 0
    maximum_height: int = 10000


@dataclass
class SamplingConfig:
    """Parsed sampling configuration.

    Attributes:
        output_format: Image format "JPEG" or "PNG"
        bucket_name: MinIO bucket name
        strategy_code: Strategy identifier for path construction
        strategy_name: Human-readable strategy name
        time_unit: Time unit for sampling (0=seconds)
        time_value: Sampling interval value
        start_time: Daily start time "HH:MM"
        end_time: Daily end time "HH:MM"
        strategies: List of strategy configurations
    """

    output_format: str = "JPEG"
    bucket_name: str = "algorithm"
    strategy_code: str = ""
    strategy_name: str = ""
    time_unit: int = 0
    time_value: int = 1
    start_time: str = ""
    end_time: str = ""
    strategies: List[StrategyParamConfig] = field(default_factory=list)


@TaskRegistry.register("sampling")
class SamplingDetectionTask(TaskBase):
    """Task for intelligent video frame sampling with MinIO upload.

    Features:
        - Configurable sampling interval based on FPS
        - Time-based scheduling (start/end times)
        - Full-frame and target-based sampling strategies
        - ROI extraction with configurable expansion
        - Target filtering by class, confidence, and size
        - Async upload to MinIO with timestamped paths

    Config Parameters (from configParam):
        outputFormat: "JPEG" or "PNG"
        bucketName: MinIO bucket name
        strategyCode: Strategy identifier for path construction
        strategyName: Human-readable name
        timeUnit: 0=seconds
        timeValue: Sampling interval (frame_skip = timeValue * fps)
        startTime: Daily start time "HH:MM"
        endTime: Daily end time "HH:MM"
        strategyParamConfigList: List of strategy configurations
    """

    def requires_stream(self) -> bool:
        """This task requires video stream input."""
        return True

    def _init_in_process(self) -> None:
        """Initialize resources in the thread.

        - Parse configuration from task_config
        - Calculate frame skip based on timeValue and fps
        - Initialize model if target-based strategies exist
        - Initialize MinIO client for uploads
        """
        self.log("Initializing SamplingDetectionTask...")

        # Parse configuration
        self._config = self._parse_config()

        # Calculate frame skip: timeValue * fps
        fps = self.task_config.get_stream_fps()
        if fps <= 0:
            fps = 25  # Default fallback
        self._frame_skip = max(1, int(self._config.time_value * fps))
        self.log(
            f"Frame skip: {self._frame_skip} "
            f"(timeValue={self._config.time_value}, fps={fps})"
        )

        # Initialize counters
        self._frame_count = 0
        self._sample_count = 0

        # Circuit breaker for inference errors
        self._inference_error_count = 0
        self._max_consecutive_errors = 10

        # Load model only if target-based strategies exist
        self._predictor = None
        if self._has_target_strategies():
            from models import Predictor

            model_path = self.task_config.get_extra(
                "model_path", "weights/yolo11m.pt"
            )
            # Also check model field from configParam
            model_name = self.task_config.config_params.get("model", "")
            if model_name:
                model_path = f"weights/{model_name}"

            device = self.task_config.get_extra("device", "cuda")
            conf = self._get_min_confidence()

            self._predictor = Predictor(
                model_path=model_path,
                task="detect",
                device=device,
                conf=conf,
            )
            self.log(f"Loaded model: {model_path}")

        # Initialize MinIO client
        jpeg_quality = 100 if self._config.output_format == "JPEG" else 95
        minio_config = MinIOConfig(
            bucket_name=self._config.bucket_name,
            jpeg_quality=jpeg_quality,
        )
        self._minio_client = MinIOClient(minio_config)
        self._minio_client.start()
        self.log(f"MinIO client started for bucket: {self._config.bucket_name}")

        self.log(
            f"SamplingDetectionTask initialized with "
            f"{len(self._config.strategies)} strategies"
        )

    def _cleanup_in_process(self) -> None:
        """Release resources.

        Waits for the MinIO upload queue to drain before stopping,
        with a timeout to prevent indefinite blocking.
        """
        if hasattr(self, "_minio_client") and self._minio_client:
            # Wait for upload queue to drain with timeout
            drain_timeout = 30  # seconds
            start_time = time.time()
            queue_size = self._minio_client.get_queue_size()

            if queue_size > 0:
                self.log(f"Waiting for {queue_size} uploads to complete...")

            while (
                self._minio_client.get_queue_size() > 0
                and (time.time() - start_time) < drain_timeout
            ):
                time.sleep(0.1)

            remaining = self._minio_client.get_queue_size()
            if remaining > 0:
                self.log(f"Timeout: {remaining} uploads still pending")

            self._minio_client.stop()
            self._minio_client = None

        self._predictor = None
        self.log(
            f"SamplingDetectionTask cleaned up. "
            f"Total samples: {getattr(self, '_sample_count', 0)}"
        )

    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a single frame for sampling.

        Flow:
        1. Check time schedule - skip if outside scheduled hours
        2. Apply frame skip based on sampling interval
        3. Run model inference (for target-based strategies)
        4. Process each configured strategy
        5. Generate and upload images to MinIO
        6. Return TaskResult with upload metadata

        Args:
            frame: BGR image as numpy array.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with sampling metadata.
        """
        result = TaskResult(task_id=self.task_id)
        result.timestamp = timestamp

        # Step 1: Check time schedule
        if not self._is_within_schedule(timestamp):
            return result

        # Step 2: Frame skip check
        self._frame_count += 1
        if not self._should_sample_frame(self._frame_count):
            return result

        # Step 3: Run inference (needed for target-based strategies)
        detections = []
        if self._has_target_strategies():
            detections = self._run_inference(frame)

        # Step 4 & 5: Process each strategy
        uploads = []
        for strategy in self._config.strategies:
            if strategy.strategy_type == 3:
                # Full-frame strategy
                uploads.extend(
                    self._process_full_frame_strategy(frame, timestamp)
                )
            elif strategy.strategy_type == 4:
                # Target-based strategy
                uploads.extend(
                    self._process_target_strategy(
                        frame, timestamp, strategy, detections
                    )
                )

        # Step 6: Build result
        result.metadata["uploads"] = uploads
        result.metadata["frame_count"] = self._frame_count
        result.metadata["sample_count"] = self._sample_count

        return result

    # ==================== Configuration Parsing ====================

    def _parse_config(self) -> SamplingConfig:
        """Parse sampling configuration from task config.

        Returns:
            SamplingConfig with all parsed parameters.
        """
        params = self.task_config.config_params

        config = SamplingConfig(
            output_format=params.get("outputFormat", "JPEG").upper(),
            bucket_name=params.get("bucketName", "algorithm"),
            strategy_code=params.get("strategyCode", self.task_id),
            strategy_name=params.get("strategyName", self.task_name),
            time_unit=int(params.get("timeUnit", 0)),
            time_value=int(params.get("timeValue", 1)),
            start_time=params.get("startTime", ""),
            end_time=params.get("endTime", ""),
        )

        # Parse strategy list
        strategy_list = params.get("strategyParamConfigList", [])
        for strategy_param in strategy_list:
            config.strategies.append(self._parse_strategy_param(strategy_param))

        # Default to full-frame if no strategies configured
        if not config.strategies:
            config.strategies.append(StrategyParamConfig(strategy_type=3))
            self.log("No strategies configured, using default full-frame strategy")

        return config

    def _parse_strategy_param(self, param: Dict) -> StrategyParamConfig:
        """Parse a single strategy parameter configuration.

        Args:
            param: Strategy parameter dictionary.

        Returns:
            StrategyParamConfig instance with validated values.
        """
        try:
            # Handle targetType which may be list of ints or strings
            target_type_raw = param.get("targetType", [])
            target_type = []
            for t in target_type_raw:
                if isinstance(t, int):
                    target_type.append(t)
                elif isinstance(t, str):
                    try:
                        target_type.append(int(t))
                    except ValueError:
                        self.log(f"Invalid target type value: {t}")

            # Handle paramValue and maxValue which may be strings
            param_value = param.get("paramValue", 0.5)
            if isinstance(param_value, str):
                try:
                    param_value = float(param_value)
                except ValueError:
                    self.log(f"Invalid paramValue: {param_value}, using 0.5")
                    param_value = 0.5

            max_value = param.get("maxValue", 1.0)
            if isinstance(max_value, str):
                try:
                    max_value = float(max_value)
                except ValueError:
                    self.log(f"Invalid maxValue: {max_value}, using 1.0")
                    max_value = 1.0

            # Validate confidence range
            param_value = max(0.0, min(1.0, param_value))
            max_value = max(param_value, min(1.0, max_value))

            # Parse target_area with validation
            try:
                target_area = float(param.get("targetArea", 2.0))
                if target_area <= 0:
                    self.log(f"Invalid targetArea: {target_area}, using 2.0")
                    target_area = 2.0
            except (ValueError, TypeError):
                target_area = 2.0

            # Parse size bounds with validation
            try:
                min_width = max(0, int(param.get("minimumWidth", 0)))
            except (ValueError, TypeError):
                min_width = 0

            try:
                # Note: typo "maxmumWidth" in spec preserved for compatibility
                max_width = int(param.get("maxmumWidth", 10000))
                max_width = max(min_width, max_width)
            except (ValueError, TypeError):
                max_width = 10000

            try:
                min_height = max(0, int(param.get("minimumHeight", 0)))
            except (ValueError, TypeError):
                min_height = 0

            try:
                # Note: typo "maxmumHeight" in spec preserved for compatibility
                max_height = int(param.get("maxmumHeight", 10000))
                max_height = max(min_height, max_height)
            except (ValueError, TypeError):
                max_height = 10000

            return StrategyParamConfig(
                strategy_type=int(param.get("strategyType", 3)),
                target_type=target_type,
                param_value=param_value,
                max_value=max_value,
                output_content=str(param.get("outputContent", "1")),
                target_area=target_area,
                minimum_width=min_width,
                maximum_width=max_width,
                minimum_height=min_height,
                maximum_height=max_height,
            )
        except Exception as e:
            self.log(f"Error parsing strategy config: {e}, using defaults")
            return StrategyParamConfig()

    # ==================== Scheduling ====================

    def _parse_time_string(self, time_str: str) -> Optional[Tuple[int, int]]:
        """Parse time string in HH:MM format.

        Args:
            time_str: Time string like "07:30" or "22:00"

        Returns:
            Tuple of (hour, minute) if valid, None otherwise.
        """
        try:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0

            # Validate ranges
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                self.log(f"Invalid time {time_str}: hour/minute out of range")
                return None

            return (hour, minute)
        except (ValueError, IndexError):
            self.log(f"Invalid time format: {time_str}")
            return None

    def _is_within_schedule(self, timestamp: float) -> bool:
        """Check if current time is within scheduled operating hours.

        Args:
            timestamp: Unix timestamp of current frame.

        Returns:
            True if within schedule, False otherwise.

        Note:
            If startTime/endTime not configured, always returns True.
            This enables auto-pause/resume based on daily schedule.
        """
        if not self._config.start_time or not self._config.end_time:
            return True

        try:
            current_dt = datetime.fromtimestamp(timestamp)
            current_minutes = current_dt.hour * 60 + current_dt.minute

            # Parse schedule times with validation
            start = self._parse_time_string(self._config.start_time)
            end = self._parse_time_string(self._config.end_time)

            if start is None or end is None:
                self.log("Invalid schedule times, allowing all times")
                return True

            start_minutes = start[0] * 60 + start[1]
            end_minutes = end[0] * 60 + end[1]

            # Handle overnight schedules (e.g., 22:00 to 06:00)
            if start_minutes <= end_minutes:
                return start_minutes <= current_minutes <= end_minutes
            else:
                return current_minutes >= start_minutes or current_minutes <= end_minutes

        except Exception as e:
            self.log(f"Error checking schedule: {e}")
            return True

    def _should_sample_frame(self, frame_count: int) -> bool:
        """Check if this frame should be sampled based on interval.

        Args:
            frame_count: Current frame number (1-indexed).

        Returns:
            True if this frame should be sampled.
        """
        return frame_count % self._frame_skip == 0

    # ==================== Inference ====================

    def _has_target_strategies(self) -> bool:
        """Check if any target-based strategies are configured."""
        return any(s.strategy_type == 4 for s in self._config.strategies)

    def _get_min_confidence(self) -> float:
        """Get minimum confidence threshold across all strategies."""
        if not self._config.strategies:
            return 0.5
        return min(s.param_value for s in self._config.strategies)

    def _run_inference(self, frame: np.ndarray) -> List[Dict]:
        """Run model inference and return detection list.

        Uses a circuit breaker pattern to disable inference after
        too many consecutive errors (e.g., CUDA OOM).

        Args:
            frame: BGR image array.

        Returns:
            List of detection dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - class_id: int
            - confidence: float
            - detection_index: int
        """
        if self._predictor is None:
            return []

        # Circuit breaker: disable inference after too many errors
        if self._inference_error_count >= self._max_consecutive_errors:
            return []

        try:
            results = self._predictor(frame)
            if not results or len(results) == 0:
                self._inference_error_count = 0  # Reset on success
                return []

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                self._inference_error_count = 0  # Reset on success
                return []

            detections = []
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                detections.append(
                    {
                        "bbox": xyxy[i].tolist(),
                        "class_id": int(cls[i]),
                        "confidence": float(conf[i]),
                        "detection_index": i,
                    }
                )

            self._inference_error_count = 0  # Reset on success
            return detections

        except Exception as e:
            self._inference_error_count += 1
            self.log(
                f"Inference error ({self._inference_error_count}/"
                f"{self._max_consecutive_errors}): {e}"
            )
            if self._inference_error_count >= self._max_consecutive_errors:
                self.log("Too many consecutive inference errors, disabling inference")
            return []

    # ==================== Filtering ====================

    def _filter_detections(
        self, detections: List[Dict], strategy: StrategyParamConfig
    ) -> List[Dict]:
        """Filter detections based on strategy configuration.

        Filters by:
        1. Target type (class index)
        2. Confidence range (paramValue to maxValue)
        3. Size bounds (width/height)

        Args:
            detections: List of detection dicts.
            strategy: Strategy configuration with filter parameters.

        Returns:
            Filtered list of detections.
        """
        filtered = []

        for det in detections:
            # Filter by target type (class index)
            if strategy.target_type and det["class_id"] not in strategy.target_type:
                continue

            # Filter by confidence range
            conf = det["confidence"]
            if conf < strategy.param_value or conf > strategy.max_value:
                continue

            # Filter by size bounds
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < strategy.minimum_width or width > strategy.maximum_width:
                continue
            if height < strategy.minimum_height or height > strategy.maximum_height:
                continue

            filtered.append(det)

        return filtered

    # ==================== Image Processing ====================

    def _extract_roi(
        self, frame: np.ndarray, bbox: List[float], scale: float = 2.0
    ) -> Optional[np.ndarray]:
        """Extract ROI from frame with optional expansion.

        Args:
            frame: Source image (BGR).
            bbox: Bounding box [x1, y1, x2, y2].
            scale: Expansion factor (1.0=exact bbox, 2.0=double size).

        Returns:
            Cropped ROI image, or None if extraction fails.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Validate input bbox
        if x2 <= x1 or y2 <= y1:
            self.log(f"Invalid bbox dimensions: {bbox}")
            return None

        # Clamp bbox to frame dimensions
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        # Re-check after clamping
        if x2 <= x1 or y2 <= y1:
            self.log(f"Bbox outside frame bounds: {bbox}")
            return None

        # Calculate center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        box_w = x2 - x1
        box_h = y2 - y1

        # Apply expansion
        new_w = box_w * scale
        new_h = box_h * scale

        # Calculate new bounds and clamp to image dimensions
        new_x1 = int(max(0, cx - new_w / 2))
        new_y1 = int(max(0, cy - new_h / 2))
        new_x2 = int(min(w, cx + new_w / 2))
        new_y2 = int(min(h, cy + new_h / 2))

        # Final validation
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            self.log(f"ROI extraction failed for bbox {bbox}, scale {scale}")
            return None

        roi = frame[new_y1:new_y2, new_x1:new_x2].copy()

        if roi.size == 0:
            self.log(f"Empty ROI extracted from bbox {bbox}")
            return None

        return roi

    # ==================== Upload ====================

    def _generate_object_name(
        self, timestamp: float, track_id: int = 0, suffix: str = ""
    ) -> str:
        """Generate MinIO object path with timestamp.

        Format: {strategyCode}/{YYYY/MM/DD}/{HHMMSSmmm}_{taskId}_{trackId}.{ext}

        Args:
            timestamp: Unix timestamp.
            track_id: Optional tracking ID (0 for full-frame).
            suffix: Optional suffix before extension.

        Returns:
            Object path string.
        """
        dt = datetime.fromtimestamp(timestamp)
        date_path = dt.strftime("%Y/%m/%d")
        timestamp_str = dt.strftime("%H%M%S") + f"{int(dt.microsecond / 1000):03d}"

        ext = "jpg" if self._config.output_format == "JPEG" else "png"

        # Use short task_id to avoid overly long paths
        short_task_id = self.task_id[:8] if len(self.task_id) > 8 else self.task_id

        filename = f"{timestamp_str}_{short_task_id}_{track_id}"
        if suffix:
            filename += f"_{suffix}"
        filename += f".{ext}"

        return f"{self._config.strategy_code}/{date_path}/{filename}"

    def _upload_image(self, image: np.ndarray, object_name: str) -> Optional[str]:
        """Upload image to MinIO.

        Args:
            image: BGR image array.
            object_name: Target path in bucket.

        Returns:
            URL if queued successfully, None otherwise.
        """
        if not self._minio_client or not self._minio_client.is_running():
            self.log("MinIO client not available")
            return None

        try:
            url = self._minio_client.upload_image(image, object_name)
            if url:
                self._sample_count += 1
                self.log(f"Queued upload: {object_name}")
            return url
        except Exception as e:
            self.log(f"Upload error: {e}")
            return None

    # ==================== Strategy Processing ====================

    def _process_full_frame_strategy(
        self, frame: np.ndarray, timestamp: float
    ) -> List[Dict]:
        """Process full-frame sampling strategy (type 3).

        Captures entire frame regardless of detections.

        Args:
            frame: Source image.
            timestamp: Frame timestamp.

        Returns:
            List of upload info dicts.
        """
        object_name = self._generate_object_name(timestamp, track_id=0, suffix="full")
        url = self._upload_image(frame, object_name)

        if url:
            return [
                {
                    "type": "full_frame",
                    "strategy_type": 3,
                    "object_name": object_name,
                    "url": url,
                    "timestamp": timestamp,
                }
            ]
        return []

    def _process_target_strategy(
        self,
        frame: np.ndarray,
        timestamp: float,
        strategy: StrategyParamConfig,
        detections: List[Dict],
    ) -> List[Dict]:
        """Process target-based sampling strategy (type 4).

        For each matching detection:
        - outputContent "1": Upload full frame
        - outputContent "2": Upload ROI with expansion

        Args:
            frame: Source image.
            timestamp: Frame timestamp.
            strategy: Strategy configuration.
            detections: List of detections from inference.

        Returns:
            List of upload info dicts.
        """
        uploads = []

        # Filter detections based on strategy
        filtered = self._filter_detections(detections, strategy)

        # Track if full frame was already uploaded (for deduplication)
        full_frame_uploaded = False

        for det in filtered:
            track_id = det.get("track_id", 0) or det.get("detection_index", 0)

            if strategy.output_content == "2":
                # ROI mode - extract and upload cropped region
                roi = self._extract_roi(
                    frame, det["bbox"], scale=strategy.target_area
                )
                # Skip if ROI extraction failed
                if roi is None:
                    continue

                object_name = self._generate_object_name(
                    timestamp, track_id=track_id, suffix="roi"
                )
                url = self._upload_image(roi, object_name)
            else:
                # Full image mode - upload once for all detections
                # to avoid duplicate uploads of the same frame
                if full_frame_uploaded:
                    continue

                object_name = self._generate_object_name(
                    timestamp, track_id=0, suffix="target"
                )
                url = self._upload_image(frame, object_name)
                full_frame_uploaded = True

            if url:
                uploads.append(
                    {
                        "type": "target",
                        "strategy_type": 4,
                        "output_content": strategy.output_content,
                        "object_name": object_name,
                        "url": url,
                        "timestamp": timestamp,
                        "detection": det,
                    }
                )

        return uploads
