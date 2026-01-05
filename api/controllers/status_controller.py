"""Status and health controller.

This module provides the StatusController class that handles
status and health check API requests.
"""

from typing import Any, Dict

from utils import Logger
from ..models import ApiResponse
from ..services import ResourceService


class StatusController:
    """Controller for status and health operations.

    This controller handles API requests for:
    - Health checks
    - Capability queries
    - Resource monitoring
    """

    def __init__(self):
        """Initialize status controller."""
        self._service = ResourceService()
        self.log = Logger.get_logging_method("STATUS-CTL")

    def health_check(self) -> Dict[str, Any]:
        """Handle health check request.

        Returns:
            Health status response.
        """
        try:
            status = self._service.get_health_status()
            return ApiResponse.success(data=status).to_dict()
        except Exception as e:
            self.log(f"Error in health_check: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Health check failed: {str(e)}"
            ).to_dict()

    def get_capabilities(self) -> Dict[str, Any]:
        """Handle capability query request.

        Returns:
            Capabilities response in Algorithm Warehouse format.
        """
        try:
            capabilities = self._service.get_capabilities()
            return ApiResponse.success(data=capabilities).to_dict()
        except Exception as e:
            self.log(f"Error in get_capabilities: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Failed to get capabilities: {str(e)}"
            ).to_dict()

    def get_engine_status(self) -> Dict[str, Any]:
        """Handle engine status request.

        Returns:
            Detailed engine status response.
        """
        try:
            status = self._service.get_engine_status()
            return ApiResponse.success(data=status).to_dict()
        except Exception as e:
            self.log(f"Error in get_engine_status: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Failed to get engine status: {str(e)}"
            ).to_dict()

    def get_resource_usage(self) -> Dict[str, Any]:
        """Handle resource usage request.

        Returns:
            Resource usage response.
        """
        try:
            usage = self._service.get_resource_usage()
            return ApiResponse.success(data=usage).to_dict()
        except Exception as e:
            self.log(f"Error in get_resource_usage: {e}")
            return ApiResponse.error(
                code=500,
                message=f"Failed to get resource usage: {str(e)}"
            ).to_dict()
