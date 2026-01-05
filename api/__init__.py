"""API module - Flask API service layer.

This module provides the REST API for the task management system.

Components:
    - app: Flask application factory
    - routes: API endpoint definitions
    - models: Request/response models
    - controllers: Request handlers
    - services: Business logic services

Usage:
    >>> from api import create_app, run_server
    >>> app = create_app()
    >>> run_server(host="0.0.0.0", port=8666)

Or use the WSGI application:
    >>> from api import get_wsgi_app
    >>> app = get_wsgi_app()

API Endpoints:
    GET  /IAP/Status       - Health check
    GET  /IAP/capabilities - Get engine capabilities
    GET  /IAP/engineStatus - Get detailed engine status
    GET  /IAP/resources    - Get resource usage
    POST /IAP/runTask      - Start task(s)
    POST /IAP/deleteTask   - Delete task(s)
    POST /IAP/pauseTask    - Pause a task
    POST /IAP/resumeTask   - Resume a task
    GET/POST /IAP/taskStatus - Get task status
"""

from .app import (
    create_app,
    init_engine,
    run_server,
    run_gunicorn,
    get_wsgi_app,
)
from .models import (
    ApiResponse,
    TaskRunRequest,
    TaskDeleteRequest,
    TaskPauseRequest,
    TaskResumeRequest,
    TaskStatusRequest,
    TaskStatusResponse,
    CapabilityResponse,
    HealthResponse,
)
from .routes import api_bp

__all__ = [
    # Application
    "create_app",
    "init_engine",
    "run_server",
    "run_gunicorn",
    "get_wsgi_app",
    "api_bp",
    # Models
    "ApiResponse",
    "TaskRunRequest",
    "TaskDeleteRequest",
    "TaskPauseRequest",
    "TaskResumeRequest",
    "TaskStatusRequest",
    "TaskStatusResponse",
    "CapabilityResponse",
    "HealthResponse",
]
