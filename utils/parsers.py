"""API request parsing utilities for task management system.

This module provides functions to parse and validate dynamic parameters from API requests.
It handles the conversion of various input formats to a normalized internal format.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


def parse_analyse_conditions(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse and normalize analyseConditions from API request.

    Extracts task configurations from the request data, handling multiple input formats:
    - 'analyseConditions' key (primary)
    - 'tasks' key (alternative)
    - Single dict (wraps in list)

    Args:
        request_data: The raw request data dict.

    Returns:
        List of normalized task configuration dicts.

    Example:
        >>> data = {"analyseConditions": [{"taskID": "t1", ...}]}
        >>> parse_analyse_conditions(data)
        [{"taskID": "t1", ...}]
    """
    # Try primary key first
    conditions = request_data.get("analyseConditions")

    # Fall back to alternative key
    if conditions is None:
        conditions = request_data.get("tasks")

    # Handle missing key
    if conditions is None:
        return []

    # Wrap single dict in list
    if isinstance(conditions, dict):
        conditions = [conditions]

    # Ensure we have a list
    if not isinstance(conditions, list):
        return []

    return conditions


def parse_config_param(config_param: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and normalize configParam dynamic parameters.

    Converts string values to appropriate types and validates numeric ranges.

    Supported parameters:
    - Thresholds: confidence, iou, kpt_score, face_score (float 0.0-1.0)
    - Processing: frame_skip, rtsp_interval (int >= 0)
    - Boolean: enable_tracking, save_images

    Args:
        config_param: The configParam dict from API request.

    Returns:
        Normalized config dict with properly typed values.

    Example:
        >>> config = {"confidence": "0.5", "frame_skip": "2"}
        >>> parse_config_param(config)
        {"confidence": 0.5, "frame_skip": 2}
    """
    if not config_param or not isinstance(config_param, dict):
        return {}

    result = dict(config_param)  # Shallow copy

    # Parse float thresholds (0.0-1.0)
    float_keys = ["rtsp_interval",
                  "confidence", "iou", "kpt_score", "face_score"]
    for key in float_keys:
        if key in result:
            result[key] = _parse_float(result[key], 0.0, 1.0, default=0.5)

    # Parse integer settings (>= 0)
    int_keys = ["frame_skip", "max_det"]
    for key in int_keys:
        if key in result:
            result[key] = _parse_int(result[key], min_val=0, default=0)

    # Parse boolean settings
    bool_keys = ["enable_tracking", "save_images", "save_video"]
    for key in bool_keys:
        if key in result:
            result[key] = _parse_bool(result[key])

    # Parse execPeriodTime if present
    if "execPeriodTime" in result:
        parsed_period = parse_exec_period(result["execPeriodTime"])
        if parsed_period:
            result["execPeriodTime"] = parsed_period

    return result


def parse_exec_period(exec_period_str: Any) -> Optional[Dict[str, Any]]:
    """Parse execPeriodTime JSON string into dict.

    Handles double-parsing (JSON string containing JSON) and supports multiple formats:
    - Format 1: {"weekdays": [1,2,3], "start": "08:00", "end": "18:00"}
    - Format 2: {"weeks": "1,2,3", "startTime": "00:00:00", "endTime": "23:59:59"}

    Args:
        exec_period_str: JSON string or already-parsed dict.

    Returns:
        Parsed dict, or None if parsing fails.

    Example:
        >>> parse_exec_period('{"weekdays": [1,2,3], "start": "08:00", "end": "18:00"}')
        {"weekdays": [1, 2, 3], "start": "08:00", "end": "18:00"}
    """
    if exec_period_str is None:
        return None

    # Already a dict
    if isinstance(exec_period_str, dict):
        return _normalize_exec_period(exec_period_str)

    # Try to parse as JSON string
    if not isinstance(exec_period_str, str):
        return None

    try:
        parsed = json.loads(exec_period_str)

        # Handle double-encoded JSON (string within string)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        if isinstance(parsed, dict):
            return _normalize_exec_period(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def _normalize_exec_period(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize execution period to consistent format.

    Converts various input formats to:
    {"weekdays": [1,2,3,...], "start": "HH:MM", "end": "HH:MM"}

    Args:
        data: Parsed execution period dict.

    Returns:
        Normalized execution period dict.
    """
    result = {}

    # Handle weekdays/weeks
    if "weekdays" in data:
        weekdays = data["weekdays"]
        if isinstance(weekdays, list):
            result["weekdays"] = [int(d) for d in weekdays if _is_valid_weekday(d)]
        elif isinstance(weekdays, str):
            result["weekdays"] = [int(d) for d in weekdays.split(",") if _is_valid_weekday(d.strip())]
    elif "weeks" in data:
        weeks = data["weeks"]
        if isinstance(weeks, str):
            result["weekdays"] = [int(d) for d in weeks.split(",") if _is_valid_weekday(d.strip())]
        elif isinstance(weeks, list):
            result["weekdays"] = [int(d) for d in weeks if _is_valid_weekday(d)]

    # Handle start time
    if "start" in data:
        result["start"] = _normalize_time(data["start"])
    elif "startTime" in data:
        result["start"] = _normalize_time(data["startTime"])

    # Handle end time
    if "end" in data:
        result["end"] = _normalize_time(data["end"])
    elif "endTime" in data:
        result["end"] = _normalize_time(data["endTime"])

    return result


def _is_valid_weekday(value: Any) -> bool:
    """Check if value is a valid weekday (1-7)."""
    try:
        day = int(value)
        return 1 <= day <= 7
    except (ValueError, TypeError):
        return False


def _normalize_time(time_str: Any) -> str:
    """Normalize time string to HH:MM format."""
    if not isinstance(time_str, str):
        return "00:00"

    # Strip seconds if present (HH:MM:SS -> HH:MM)
    parts = time_str.strip().split(":")
    if len(parts) >= 2:
        return f"{parts[0]:0>2}:{parts[1]:0>2}"
    return "00:00"


def validate_task_request(request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate complete API request structure for runTask endpoint.

    Checks for required fields in each task configuration:
    - taskID
    - deviceInfo.deviceCode
    - deviceInfo.sourceRTSP
    - configParam.task

    Args:
        request_data: The raw request data dict.

    Returns:
        Tuple of (is_valid, error_messages).
        is_valid is True if all required fields are present.
        error_messages contains descriptions of missing/invalid fields.

    Example:
        >>> data = {"analyseConditions": [{"taskID": "t1"}]}
        >>> is_valid, errors = validate_task_request(data)
        >>> is_valid
        False
        >>> errors
        ['Task 0 (t1): missing deviceInfo.deviceCode', ...]
    """
    errors = []

    conditions = parse_analyse_conditions(request_data)

    if not conditions:
        errors.append("No task configurations found (missing analyseConditions or tasks)")
        return False, errors

    for idx, task_config in enumerate(conditions):
        task_id = task_config.get("taskID", f"unknown_{idx}")
        task_prefix = f"Task {idx} ({task_id})"

        # Check taskID
        if not task_config.get("taskID"):
            errors.append(f"{task_prefix}: missing taskID")

        # Check deviceInfo
        device_info = task_config.get("deviceInfo", {})
        if not isinstance(device_info, dict):
            errors.append(f"{task_prefix}: deviceInfo must be an object")
        else:
            if not device_info.get("deviceCode"):
                errors.append(f"{task_prefix}: missing deviceInfo.deviceCode")
            if not device_info.get("sourceRTSP"):
                errors.append(f"{task_prefix}: missing deviceInfo.sourceRTSP")

        # Check configParam
        config_param = task_config.get("configParam", {})
        if not isinstance(config_param, dict):
            errors.append(f"{task_prefix}: configParam must be an object")
        else:
            if not config_param.get("task"):
                errors.append(f"{task_prefix}: missing configParam.task")

    return len(errors) == 0, errors


def _parse_float(value: Any, min_val: float, max_val: float, default: float) -> float:
    """Parse value to float and clamp to range.

    Args:
        value: Value to parse.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        default: Default value if parsing fails.

    Returns:
        Parsed and clamped float value.
    """
    try:
        result = float(value)
        return max(min_val, min(max_val, result))
    except (ValueError, TypeError):
        return default


def _parse_int(value: Any, min_val: int, default: int) -> int:
    """Parse value to int and clamp to minimum.

    Args:
        value: Value to parse.
        min_val: Minimum allowed value.
        default: Default value if parsing fails.

    Returns:
        Parsed and clamped int value.
    """
    try:
        result = int(value)
        return max(min_val, result)
    except (ValueError, TypeError):
        return default


def _parse_bool(value: Any) -> bool:
    """Parse value to boolean.

    Handles: True, False, "true", "false", 1, 0, "1", "0"

    Args:
        value: Value to parse.

    Returns:
        Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False
