# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **forked ultralytics repository** with custom additions for a **video analytics task management system**. The original ultralytics YOLO framework has been stripped down and augmented with:
- Multi-process task management engine
- REST API for task lifecycle control
- SEI video streaming with H.264 encoding and RTMP output
- Standalone tracking module (ByteTrack, BOTSORT)
- MinIO object storage integration
- MQTT messaging client
- Algorithm Warehouse integration

## Running the Service

```bash
# Development server
<<<<<<< HEAD
python run_service.py --host 0.0.0.0 --port 8666
=======
python run_service.py --host 0.0.0.0 --port 8555
>>>>>>> 07331326 (feat: build video analytics task management system)

# Debug mode
python run_service.py --debug

# Production with Gunicorn
python run_service.py --production --workers 4
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cross_line.py

# Run with verbose output
pytest tests/test_sei.py -v

# Run single test function
pytest tests/test_track.py::test_byte_tracker -v
```

## Architecture

### Engine Hierarchy

```
MainEngine (Singleton)
    └── StandEngine (1 per stand)
            └── CameraEngine (1 per camera)
                    └── TaskManager
                            └── Task instances (separate OS processes)
```

### Key Modules

**`engine/`** - Core task orchestration
- `main_engine.py`: Global singleton coordinator for all tasks
- `stand_engine.py`: Per-stand task grouping
- `camera_engine.py`: Per-camera frame distribution and task management

**`task/`** - Task framework
- `base.py`: `TaskBase` abstract class - each task runs as a **separate OS process** using `multiprocessing.Process`
- `config.py`: `TaskConfig` dataclass and `TaskConfigManager`
- `registry.py`: `TaskRegistry` for dynamic task type registration
- `state.py`: `TaskStateMachine` for state transitions

**`api/`** - Flask REST API
- `app.py`: Application factory, `create_app()` and `run_server()`
- `routes.py`: Blueprint registration
- `controllers/`: Request handlers
- `warehouse/`: Algorithm Warehouse integration client

**`models/`** - Standalone YOLO inference
- `predictor.py`: `Predictor` class for model inference
- `backend.py`: `ModelBackend` for ONNX/TensorRT loading
- `results.py`: `Results`, `Boxes`, `Keypoints` data classes
- `annotator.py`: Drawing utilities

**`tracks/`** - Multi-object tracking
- `byte_tracker.py`: BYTETracker implementation
- `bot_sort.py`: BOTSORT implementation
- `config.py`: `TrackerConfig` dataclass
- `visualize.py`: Track drawing utilities

**`sei/`** - Video streaming pipeline
- `encoder.py`: H.264 encoding via FFmpeg
- `injector.py`: SEI data injection into NAL units
- `streamer.py`: RTMP streaming
- `recorder.py`: Event-triggered MP4 recording
- `pipeline.py`: `SeiStreamingPipeline` orchestration

**`stream/`** - Video input handling
- `stream_loader.py`: RTSP/file stream loading

**`solutions/`** - Task implementations
- `face_detection.py`: Face detection
- `cross_line.py`: Line crossing detection
- `crowd_density.py`: Crowd analysis
- `region_intrusion.py`: Zone intrusion detection
- Each solution registers itself with `TaskRegistry`

**`utils/`** - Shared utilities
- `minio/`: MinIO object storage client with async upload queue
- `mqtt/`: MQTT client for event publishing
- `logger.py`: Centralized logging with per-task log files
- `singleton.py`: `SingletonMeta` metaclass

### Process Model

Tasks run as **separate OS processes** for isolation:
- Main process receives API requests and manages engine hierarchy
- Each task spawns via `multiprocessing.Process`
- Communication via `multiprocessing.Queue`:
  - `input_queue`: Frames from CameraEngine
  - `output_queue`: Results back to manager
  - `control_queue`: pause/resume/stop commands

### Creating a New Task Type

1. Create class extending `TaskBase` in `solutions/`
2. Implement required methods:
   - `requires_stream() -> bool`
   - `on_process(frame, timestamp) -> TaskResult`
   - `_init_in_process()`: Load models here (runs in subprocess)
   - `_cleanup_in_process()`: Release resources
3. Register with decorator: `@TaskRegistry.register("your_task_type")`

### Configuration Flow

```
API Request (AnalyseCondition format)
    → TaskConfigManager.build_config()
    → TaskConfig dataclass
    → Stored in TaskConfigManager (singleton)
    → StandEngine/CameraEngine lookup
    → Task process spawned
```

## Key Patterns

- **Singleton**: `MainEngine`, `TaskConfigManager`, `InferenceManager` use `SingletonMeta`
- **Registry**: `TaskRegistry` for dynamic task type lookup
- **State Machine**: `TaskStateMachine` enforces valid state transitions
- **Factory**: `create_app()` for Flask, `create_predictor()` for inference
- **Context Manager**: MinIO/MQTT clients support `with` statement
