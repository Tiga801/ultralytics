"""Resource monitoring service.

This module provides the ResourceService class for monitoring
system resources and engine capabilities.
"""

from typing import Any, Dict

from utils import Logger
from engine import MainEngine


class ResourceService:
    """Service for resource monitoring and capability reporting.

    This service provides system resource information and
    capability metrics for the Algorithm Warehouse integration.
    """

    def __init__(self):
        """Initialize resource service."""
        self._engine = MainEngine()
        self.log = Logger.get_logging_method("RES-SVC")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get engine capabilities for Algorithm Warehouse.

        Returns:
            Capabilities dictionary in Algorithm Warehouse format.
        """
        return self._engine.get_capabilities()

    def get_engine_status(self) -> Dict[str, Any]:
        """Get detailed engine status.

        Returns:
            Engine status dictionary.
        """
        return self._engine.get_engine_status()

    def get_health_status(self) -> Dict[str, Any]:
        """Get health check status.

        Returns:
            Health status dictionary.
        """
        status = self._engine.get_engine_status()

        return {
            "status": "healthy" if status.get("started") else "degraded",
            "initialized": status.get("initialized", False),
            "started": status.get("started", False),
            "taskCount": status.get("task_count", 0),
            "cameraCount": status.get("camera_count", 0),
            "standCount": status.get("stand_count", 0),
        }

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get system resource usage.

        Returns:
            Resource usage dictionary.
        """
        import os

        # Basic resource info
        result = {
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        }

        # Try to get memory info
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            result["memory_rss_mb"] = memory_info.rss / (1024 * 1024)
            result["memory_vms_mb"] = memory_info.vms / (1024 * 1024)
            result["cpu_percent"] = process.cpu_percent()
            result["num_threads"] = process.num_threads()
        except ImportError:
            pass

        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_available"] = True
                result["gpu_count"] = torch.cuda.device_count()
                result["gpu_name"] = torch.cuda.get_device_name(0)

                # Memory usage
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                result["gpu_memory_allocated_mb"] = allocated / (1024 * 1024)
                result["gpu_memory_reserved_mb"] = reserved / (1024 * 1024)
            else:
                result["gpu_available"] = False
        except ImportError:
            result["gpu_available"] = False

        return result

    def can_accept_task(self) -> bool:
        """Check if engine can accept new tasks.

        Returns:
            True if new tasks can be accepted.
        """
        capabilities = self.get_capabilities()
        cur_num = capabilities.get("taskCurNum", 0)
        max_num = capabilities.get("taskTotalNum", 10)
        return cur_num < max_num

    def get_available_capacity(self) -> int:
        """Get remaining task capacity.

        Returns:
            Number of additional tasks that can be added.
        """
        capabilities = self.get_capabilities()
        cur_num = capabilities.get("taskCurNum", 0)
        max_num = capabilities.get("taskTotalNum", 10)
        return max(0, max_num - cur_num)
