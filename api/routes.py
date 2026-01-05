"""Flask API routes.

This module defines all API endpoints for the task management system.
"""

from flask import Blueprint, jsonify, request

from .controllers import TaskController, StatusController
from utils import validate_task_request, parse_analyse_conditions


# Create blueprint
api_bp = Blueprint("api", __name__, url_prefix="/IAP")

# Initialize controllers
task_controller = TaskController()
status_controller = StatusController()


# ==================== Health & Status Endpoints ====================

@api_bp.route("/Status", methods=["GET"])
def health_check():
    """Health check endpoint.

    Returns:
        JSON response with health status.
    """
    return jsonify(status_controller.health_check())


@api_bp.route("/capabilities", methods=["GET"])
def get_capabilities():
    """Get engine capabilities.

    Returns:
        JSON response with capabilities.
    """
    return jsonify(status_controller.get_capabilities())


@api_bp.route("/engineStatus", methods=["GET"])
def get_engine_status():
    """Get detailed engine status.

    Returns:
        JSON response with engine status.
    """
    return jsonify(status_controller.get_engine_status())


@api_bp.route("/resources", methods=["GET"])
def get_resources():
    """Get resource usage.

    Returns:
        JSON response with resource usage.
    """
    return jsonify(status_controller.get_resource_usage())


# ==================== Task Management Endpoints ====================

@api_bp.route("/runTask", methods=["POST"])
def run_task():
    """Start new task(s).

    Validates and parses the request data before passing to controller.

    Request body:
        {
            "analyseConditions": [
                {
                    "taskID": "task_cross_line_001",
                    "taskName": "Line-Crossing-Detection",
                    "deviceInfo": {
                        "deviceName": "Camera1",
                        "deviceCode": "camera1_cross_line_001",
                        "sourceRTSP": "rtsp://192.168.2.71:8554/mystream3"
                    },
                    "configParam": {
                        "task": "cross_line",
                        "model": "yolo11m.pt"
                    }
                }
            ]
        }

    Returns:
        JSON response with result.
    """
    data = request.get_json() or {}

    # Validate request structure
    is_valid, errors = validate_task_request(data)
    if not is_valid:
        return jsonify({
            "code": 400,
            "message": "Invalid request",
            "data": {"errors": errors}
        }), 400

    # Parse into normalized format
    parsed_data = {
        "analyseConditions": parse_analyse_conditions(data)
    }

    return jsonify(task_controller.run_task(parsed_data))


@api_bp.route("/deleteTask", methods=["POST"])
def delete_task():
    """Delete task(s).

    Request body:
        {
            "taskIds": ["task_001", "task_002"]
        }
        or
        {
            "analyseConditions": [
                {"taskID": "task_001"}
            ]
        }

    Returns:
        JSON response with result.
    """
    data = request.get_json() or {}
    return jsonify(task_controller.delete_task(data))


@api_bp.route("/pauseTask", methods=["POST"])
def pause_task():
    """Pause a task.

    Request body:
        {
            "taskId": "task_001"
        }

    Returns:
        JSON response with result.
    """
    data = request.get_json() or {}
    return jsonify(task_controller.pause_task(data))


@api_bp.route("/resumeTask", methods=["POST"])
def resume_task():
    """Resume a paused task.

    Request body:
        {
            "taskId": "task_001"
        }

    Returns:
        JSON response with result.
    """
    data = request.get_json() or {}
    return jsonify(task_controller.resume_task(data))


@api_bp.route("/taskStatus", methods=["GET", "POST"])
def task_status():
    """Get task status.

    GET: Returns all tasks.
    POST: Returns specific tasks by ID.

    Request body (POST):
        {
            "taskIds": ["task_001", "task_002"]
        }

    Returns:
        JSON response with task status.
    """
    if request.method == "POST":
        data = request.get_json() or {}
    else:
        data = {}
    return jsonify(task_controller.get_task_status(data))


# ==================== Error Handlers ====================

@api_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({
        "code": 400,
        "message": "Bad request",
        "data": None,
    }), 400


@api_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({
        "code": 404,
        "message": "Not found",
        "data": None,
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({
        "code": 500,
        "message": "Internal server error",
        "data": None,
    }), 500
