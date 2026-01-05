import logging
import uuid
from pathlib import Path
from datetime import datetime


def setup_mqtt_logger(task_id) -> logging.Logger:
        """Set up logger for mqtt.log file.

        Returns:
            Logger configured to write to logs/mqtt.log
        """
        logger = logging.getLogger(f"mqtt.{task_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_dir / f"mqtt-{task_id}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
        return logger


def generate_minio_object_name(ext: str, image_type: str = "body") -> str:
        now = datetime.now()
        unique_id = uuid.uuid4().hex
        
        if image_type == "body":
            prefix = "body"
        elif image_type == "face":
            prefix = "face"
        elif image_type == "expanded":
            prefix = "expand"
        elif image_type == "visual":
            prefix = "visual"
        else:
            prefix = image_type
        
        return f"{now.year}/{now.month:02d}/{now.day:02d}/{prefix}-{unique_id}.{ext}"