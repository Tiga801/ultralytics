"""Face detection task.

This module provides the FaceDetectionTask class for detecting
and optionally recognizing faces in video streams.
"""

from typing import Dict, Optional, Set
import numpy as np
import cv2
import base64
import uuid
from datetime import datetime

from task import TaskBase, TaskRegistry, TaskResult
from tracks import BYTETracker, TrackerConfig
from utils.mqtt import MQTTClient, MQTTConfig
from utils.minio import MinIOClient, MinIOConfig
from .utils.region import parse_detection_region
from .utils.message import setup_mqtt_logger
from .utils.visual import face_detection_visualization
from .face_detect import get_ori_face_bbox


@TaskRegistry.register("face_detection")
class FaceDetectionTask(TaskBase):
    """Task for face detection and recognition.

    This task detects faces in video streams and optionally
    performs face recognition against a database of known faces.

    Features:
        - Real-time face detection
        - Optional face recognition
        - Face attribute estimation (optional)
        - Configurable detection regions
    """

    def requires_stream(self) -> bool:
        """This task requires video stream input."""
        return True

    def _init_in_process(self) -> None:
        """Initialize resources in the subprocess."""
        from models import Predictor

        self.log(f"Initializing FaceDetectionTask {self.task_id}...")

        # Load pose detection model
        model_path = self.task_config.get_extra("model_path", "weights/yolov8m-pose.pt")
        device = self.task_config.get_extra("device", "cuda")
        conf = self.task_config.get_extra("confidence", 0.5)
        iou = self.task_config.get_extra("iou", 0.45)
        self._predictor = Predictor(model_path=model_path, device=device, conf=conf, iou=iou)

        # Initialize BYTETracker for persistent track IDs
        tracker_config = TrackerConfig.bytetrack_default()
        frame_rate = self.task_config.get_extra("frame_rate", 30)
        self.tracker = BYTETracker(tracker_config, frame_rate=frame_rate)

        # Initialize InsightFace for face detection
        from insightface.app import FaceAnalysis
        self._face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        # Per-track state: {track_id: {"max_kpt_sum": float, "max_face_score": float}}
        self._track_states: Dict[int, Dict[str, float]] = {}

        # Threshold parameters from config
        self._kpt_threshold = self.task_config.get_extra("kpt_score", 0.9)
        self._face_threshold = self.task_config.get_extra("face_score", 0.9)
        self._kpt_max_score = self.task_config.get_extra("kpt_max_score", 0.95)
        self._face_max_score = self.task_config.get_extra("face_max_score", 0.95)
        self.log(f"{self.task_id} Thresholds: kpt_score={self._kpt_threshold}, face_score={self._face_threshold}")
        self.log(f"{self.task_id} Thresholds: kpt_max_score={self._kpt_max_score}, face_max_score={self._face_max_score}")

        # MQTT broker client
        mqtt_config = MQTTConfig(topic_prefix="")
        self._mqtt_client = MQTTClient(mqtt_config)
        self._mqtt_client.start()

        # MQTT file logger (keep for backward compatibility)
        self._mqtt_logger = setup_mqtt_logger(self.task_id)

        # MINIO client
        minio_config = MinIOConfig()
        self._minio_client = MinIOClient(minio_config)
        self._minio_client.start()

        # Face recognition settings
        self._enable_recognition = self.task_config.get_extra("enable_recognition", False)

        # Detection region (optional)
        self._detection_region = parse_detection_region(self.task_config.get_extra("areas_info", None))

        # Frame skip for performance
        self._frame_skip = self.task_config.get_extra("frame_skip", 1)
        self._frame_count = 0

        self.log(f"FaceDetectionTask {self.task_id} initialized, recognition={self._enable_recognition}")

    def _cleanup_in_process(self) -> None:
        """Release resources."""
        self._predictor = None
        self.tracker = None
        self._face_analyzer = None
        self._track_states.clear()

        # Stop MQTT broker client
        if hasattr(self, '_mqtt_client') and self._mqtt_client:
            self._mqtt_client.stop()
            self._mqtt_client = None
        
        # Stop MINIO client
        if hasattr(self, '_minio_client') and self._minio_client:
            self._minio_client.stop()
            self._minio_client = None

        # Close mqtt logger handlers
        if hasattr(self, '_mqtt_logger') and self._mqtt_logger:
            for handler in self._mqtt_logger.handlers[:]:
                handler.close()
                self._mqtt_logger.removeHandler(handler)

        self.log(f"FaceDetectionTask {self.task_id} cleaned up")

    def _cleanup_lost_tracks(self, current_track_ids: Optional[Set[int]] = None) -> None:
        """Remove state for tracks that are no longer active.

        Args:
            current_track_ids: Set of currently active track IDs. If None, clears all.
        """
        if current_track_ids is None:
            self._track_states.clear()
            return
        lost_ids = [tid for tid in self._track_states if tid not in current_track_ids]
        for tid in lost_ids:
            del self._track_states[tid]

    def on_process(self, frame: np.ndarray, timestamp: float) -> TaskResult:
        """Process a frame for face detection.

        Args:
            frame: BGR image.
            timestamp: Frame timestamp.

        Returns:
            TaskResult with face detections.
        """
        dt = datetime.fromtimestamp(timestamp)
        minio_object_prefix = f"face_detection/{dt.year}/{dt.month}/{dt.day}"
        result = TaskResult(task_id=self.task_id)
        result.timestamp = timestamp

        # Frame skip for performance
        self._frame_count += 1
        if self._frame_count % self._frame_skip != 0:
            return result

        # Run pose detection
        predictions = self._predictor(frame)
        if not predictions or len(predictions) == 0:
            self._cleanup_lost_tracks(set())
            return result

        pred = predictions[0]
        boxes = pred.boxes
        keypoints = pred.keypoints

        if boxes is None or len(boxes) == 0:
            self._cleanup_lost_tracks(set())
            return result

        # Extract data for tracker: [x, y, w, h, conf, cls]
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') and boxes.cls is not None else np.zeros(len(boxes))
        detections_array = np.column_stack([xywh, conf, cls])

        # Extract keypoint sums (first 5: nose, eyes, ears)
        kpt_sums_per_det = []
        if keypoints is not None and len(keypoints) > 0:
            kpt_conf = keypoints.conf.cpu().numpy()  # Shape: (N, 17)
            for i in range(len(kpt_conf)):
                kpt_sum = float(np.sum(kpt_conf[i, :5]))
                kpt_sums_per_det.append(kpt_sum)
        else:
            kpt_sums_per_det = [0.0] * len(detections_array)

        # Update tracker -> returns [x1, y1, x2, y2, track_id, score, cls, idx]
        tracks = self.tracker.update(detections_array, frame)

        current_track_ids: Set[int] = set()

        # Calculate absolute keypoint threshold (5 keypoints × threshold)
        absolute_kpt_threshold = 5 * self._kpt_threshold

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            det_idx = int(track[7]) if len(track) > 7 else -1

            current_track_ids.add(track_id)

            # Get current keypoint sum (sum of first 5 keypoint confidences)
            current_kpt_sum = kpt_sums_per_det[det_idx] if 0 <= det_idx < len(kpt_sums_per_det) else 0.0

            # Get previous state
            prev_state = self._track_states.get(track_id, {"max_kpt_sum": -1.0, "max_face_score": -1.0})
            max_kpt_sum = prev_state["max_kpt_sum"]
            max_face_score = prev_state["max_face_score"]

            # Initialize face score for this iteration
            current_face_score = max_face_score if max_face_score >= 0 else 0.0
            should_output = False  # Flag for qualifying detection
            embedding_b64 = frame_face_bbox = frame_expanded_bbox = clothing_color = ""
            body_url = face_url = expanded_face_url = visual_url = ""
            gender = age = -1
            has_glasses = has_luggage = False
            
            # ========== 5-STEP DECISION LOGIC ==========

            # Step 1: Keypoint Absolute Check
            # Check if kpt_sum >= 5 × kpt_score_threshold
            step1_passed = current_kpt_sum >= absolute_kpt_threshold

            if step1_passed:
                # Step 2: Keypoint Temporal Check
                # Check if kpt_sum > max_kpt_sum
                step2_passed = current_kpt_sum > max_kpt_sum + 0.1

                if step2_passed:
                    max_kpt_sum = current_kpt_sum
                    # Step 3: Face Detection (only if Steps 1 & 2 pass)
                    crop_x1, crop_y1 = max(0, x1), max(0, y1)
                    crop_x2, crop_y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        faces = self._face_analyzer.get(cropped_rgb)

                        if faces:
                            best_face = max(faces, key=lambda f: f.det_score)
                            current_face_score = float(best_face.det_score)

                            # Step 4: Face Score Absolute Check
                            # Check if face_score >= face_score_threshold
                            step4_passed = current_face_score >= self._face_threshold

                            if step4_passed:
                                # Step 5: Face Score Temporal Check
                                # Check if face_score > max_face_score
                                step5_passed = current_face_score > max_face_score + 0.1

                                if step5_passed:
                                    # ALL CONDITIONS PASSED - emit output
                                    should_output = True

                                    # Update track state
                                    self._track_states[track_id] = {
                                        "max_kpt_sum": max_kpt_sum,
                                        "max_face_score": current_face_score
                                    }

                                    # convert to base64 strings
                                    embedding_array = np.array(best_face.embedding, dtype=np.float64)
                                    embedding_b64 = base64.b64encode(embedding_array.tobytes()).decode('utf-8')

                                    # Get face bbox from InsightFace (in cropped body image coordinates)
                                    face_bbox = best_face.bbox.astype(int)

                                    # Transform to frame coordinates using clamped crop offset
                                    # Note: Use clamped crop coordinates as offset since face_bbox is relative to cropped image
                                    frame_face_bbox, frame_expanded_bbox = get_ori_face_bbox(
                                        frame.shape[:2], face_bbox, (crop_x1, crop_y1, crop_x2, crop_y2)
                                    )

                                    # Extract face regions from original frame using frame coordinates
                                    body_img = frame[y1:y2, x1:x2]
                                    face_img = frame[frame_face_bbox[1]:frame_face_bbox[3],
                                                     frame_face_bbox[0]:frame_face_bbox[2]]
                                    expanded_face_img = frame[frame_expanded_bbox[1]:frame_expanded_bbox[3],
                                                              frame_expanded_bbox[0]:frame_expanded_bbox[2]]
                                    visual_img = face_detection_visualization(frame)
                                    
                                    unique_id = uuid.uuid4().hex
                                    body_url = f"{minio_object_prefix}/{self.task_id}-body-{track_id}-{unique_id}.jpg"
                                    face_url = f"{minio_object_prefix}/{self.task_id}-face-{track_id}-{unique_id}.jpg"
                                    expanded_face_url = f"{minio_object_prefix}/{self.task_id}-expanded-{track_id}-{unique_id}.jpg"
                                    visual_url = f"{minio_object_prefix}/{self.task_id}-visual-{track_id}-{unique_id}.jpg"
                                    self._minio_client.upload_image(body_img, body_url)
                                    self._minio_client.upload_image(face_img, face_url)
                                    self._minio_client.upload_image(expanded_face_img, expanded_face_url)
                                    self._minio_client.upload_image(visual_img, visual_url)

                                    # cv2.imwrite(f"/home/easyair/ljwork/projects/ultralytics/runs/frame-{track_id}-{unique_id}.jpg", frame)
                                    # cv2.imwrite(f"/home/easyair/ljwork/projects/ultralytics/runs/body-{track_id}-{unique_id}.jpg", frame[y1:y2, x1:x2])
                                    # cv2.imwrite(f"/home/easyair/ljwork/projects/ultralytics/runs/face-{track_id}-{unique_id}.jpg", face_img)
                                    # cv2.imwrite(f"/home/easyair/ljwork/projects/ultralytics/runs/expand_face-{track_id}-{unique_id}.jpg", expanded_face_img)

            # Only output if ALL conditions passed
            if should_output:
                # Log to MQTT file (backward compatibility)
                self._mqtt_logger.info(
                    f"task_id={self.task_id} track_id={track_id} "
                    f"kpt_average_score={current_kpt_sum / 5:.4f} face_score={current_face_score:.4f}"
                )

                # Publish to MQTT broker
                mqtt_payload = {
                    "taskName": self.task_name,
                    "taskID": self.task_id,
                    "deviceInfo": {
                        "deviceName": self.task_config.camera_name,
                        "deviceCode": self.task_config.camera_id,
                        "sourceRTSP": self.task_config.rtsp_url,
                        "resolution": "1920*1080"
                    },
                    "execPeriodTime": "{\"weeks\": \"1,2,3,4,5,6,7\", \"startTime\": \"00:00:00\", \"endTime\": \"23:59:59\"}",
                    "timestamp": int(timestamp * 1000),
                    "results": [
                        {
                            "task": self.task_type,
                            "body": {
                                "track_id": track_id,
                                "object_class": "person",
                                "body_bbox": [x1, y1, x2, y2],
                                "body_confidence": round(current_kpt_sum / 5, 2),
                                "body_url": body_url
                            },
                            "face": {
                                "embedding": embedding_b64,
                                "face_bbox": [int(frame_face_bbox[0]), int(frame_face_bbox[1]), int(frame_face_bbox[2]), int(frame_face_bbox[3])],
                                "face_confidence": round(current_face_score, 2),
                                "face_url": face_url,
                                "expanded_face_url": expanded_face_url
                            },
                            "frame": {
                                "visual_url": visual_url
                            },
                            "features": {
                                "gender": gender,
                                "age": age,
                                "has_glasses": has_glasses,
                                "clothing_color": clothing_color,
                                "has_luggage": has_luggage
                            }
                        }
                    ]
                }

                self._mqtt_client.send("face_detection", mqtt_payload)

        # Cleanup lost tracks
        self._cleanup_lost_tracks(current_track_ids)

        return result
