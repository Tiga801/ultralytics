# -*- coding: utf-8 -*-
"""
SEI Video Streaming Module

This module provides a complete H.264 video encoding, SEI injection,
and dual-output (RTMP streaming + event-triggered recording) solution.

Features:
    - H.264 video encoding via FFmpeg
    - SEI (Supplemental Enhancement Information) data injection
    - Real-time RTMP streaming
    - Event-triggered local MP4 recording
    - Ring buffer for pre-event frame capture
    - Configurable event triggers from inference results

Processing Pipeline:
    Raw Video Frames
        │
        ▼
    H.264 Encoding
        │
        ▼
    SEI Injection
        │
        ├──▶ RTMP Streaming (Mode 1: Real-time)
        │
        └──▶ Event Recording (Mode 2: On trigger)

Quick Start:
    >>> from sei import SeiStreamingPipeline, SeiConfig, StreamConfig
    >>>
    >>> # Configure pipeline
    >>> sei_config = SeiConfig(enable=True)
    >>> stream_config = StreamConfig(width=1920, height=1080, fps=10)
    >>>
    >>> # Create and start pipeline
    >>> pipeline = SeiStreamingPipeline(
    ...     sei_config=sei_config,
    ...     stream_config=stream_config
    ... )
    >>> pipeline.start("rtmp://server/live/stream")
    >>>
    >>> # Process frames with inference data
    >>> for frame, inference_data in inference_loop:
    ...     pipeline.push_frame(frame, inference_data, task_id="task001")
    >>>
    >>> # Stop pipeline
    >>> pipeline.stop()

Component Usage:
    # Encoder only
    >>> from sei import H264Encoder, StreamConfig
    >>> encoder = H264Encoder(StreamConfig(width=1920, height=1080))
    >>> encoder.start()
    >>> h264_data = encoder.encode(frame)
    >>> encoder.stop()

    # SEI injection only
    >>> from sei import SeiInjector, SeiPayload
    >>> injector = SeiInjector()
    >>> payload = SeiPayload(frame_timestamp=time.time())
    >>> enhanced_data = injector.inject(h264_data, payload)

    # Event-triggered recording
    >>> from sei import Mp4Recorder, RecorderConfig, SeiEvent
    >>> recorder = Mp4Recorder(RecorderConfig(output_dir="recordings"))
    >>> event = SeiEvent.create("cross_line", "task001")
    >>> recorder.start_recording(event.event_id, event.event_type)

Directory Structure:
    sei/
    ├── __init__.py          # Module exports
    ├── config.py            # Configuration dataclasses
    ├── interfaces.py        # Abstract interfaces
    ├── nalutils.py          # NAL unit utilities
    ├── encoder.py           # H.264 encoder
    ├── injector.py          # SEI injector
    ├── streamer.py          # RTMP streamer
    ├── recorder.py          # MP4 recorder
    ├── buffer.py            # Ring buffer
    ├── output_manager.py    # Dual output coordinator
    ├── events.py            # Event system
    └── pipeline.py          # Complete pipeline
"""

<<<<<<< HEAD
__version__ = "1.0.0"
=======
__version__ = "2.0.0.0"
>>>>>>> 07331326 (feat: build video analytics task management system)
__author__ = "EasyAir"

# Configuration classes
from .config import (
    DEFAULT_SEI_UUID,
    EncoderConfig,
    SeiConfig,
    StreamConfig,
    BufferConfig,
    RecorderConfig,
    OutputConfig,
    SeiPipelineConfig,
    SeiStreamingConfig,  # Legacy alias
)

# Interface definitions
from .interfaces import (
    # Data classes
    SeiPayload,
    EncodedFrame,
    StreamingStatistics,
    RecordingStatistics,
    OutputResult,
    # Abstract interfaces
    H264EncoderInterface,
    SeiInjectorInterface,
    StreamerInterface,
    RecorderInterface,
    BufferInterface,
    # Callback interfaces
    SeiStreamCallback,
    DefaultSeiStreamCallback,
    EventHandlerInterface,
    DefaultEventHandler,
    # Type aliases
    LogFunc,
)

# NAL utilities
from .nalutils import (
    # Functions
    split_nalus,
    get_nal_type,
    is_video_frame_nal,
    is_keyframe_nal,
    make_sei_user_data_unregistered,
    inject_sei_nalu,
    inject_sei_into_h264_data,
    extract_sei_from_nalus,
    parse_sei_payload,
    # Constants
    NAL_START_CODE_3,
    NAL_START_CODE_4,
    NAL_TYPE_NON_IDR,
    NAL_TYPE_IDR,
    NAL_TYPE_SEI,
    NAL_TYPE_SPS,
    NAL_TYPE_PPS,
    SEI_TYPE_USER_DATA_UNREGISTERED,
)

# Core components
from .injector import SeiInjector
from .encoder import H264Encoder, SimpleH264Encoder
from .streamer import RtmpStreamer
from .recorder import Mp4Recorder
from .buffer import FrameRingBuffer, RawFrameBuffer
from .output_manager import OutputManager, OutputStatistics

# Event system
from .events import (
    SeiEvent,
    EventTrigger,
    TriggerCondition,
    EVENT_TYPE_CROSS_LINE,
    EVENT_TYPE_INTRUSION,
    EVENT_TYPE_DETECTION,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_CUSTOM,
    SUPPORTED_EVENT_TYPES,
)

# Pipeline
from .pipeline import (
    SeiStreamingPipeline,
    PushResult,
    PipelineStatistics,
    create_pipeline_from_config,
)


__all__ = [
    # Version
    "__version__",
    "__author__",

    # Configuration
    "DEFAULT_SEI_UUID",
    "EncoderConfig",
    "SeiConfig",
    "StreamConfig",
    "BufferConfig",
    "RecorderConfig",
    "OutputConfig",
    "SeiPipelineConfig",
    "SeiStreamingConfig",

    # Interfaces - Data classes
    "SeiPayload",
    "EncodedFrame",
    "StreamingStatistics",
    "RecordingStatistics",
    "OutputResult",

    # Interfaces - Abstract
    "H264EncoderInterface",
    "SeiInjectorInterface",
    "StreamerInterface",
    "RecorderInterface",
    "BufferInterface",

    # Interfaces - Callbacks
    "SeiStreamCallback",
    "DefaultSeiStreamCallback",
    "EventHandlerInterface",
    "DefaultEventHandler",
    "LogFunc",

    # NAL utilities - Functions
    "split_nalus",
    "get_nal_type",
    "is_video_frame_nal",
    "is_keyframe_nal",
    "make_sei_user_data_unregistered",
    "inject_sei_nalu",
    "inject_sei_into_h264_data",
    "extract_sei_from_nalus",
    "parse_sei_payload",

    # NAL utilities - Constants
    "NAL_START_CODE_3",
    "NAL_START_CODE_4",
    "NAL_TYPE_NON_IDR",
    "NAL_TYPE_IDR",
    "NAL_TYPE_SEI",
    "NAL_TYPE_SPS",
    "NAL_TYPE_PPS",
    "SEI_TYPE_USER_DATA_UNREGISTERED",

    # Core components
    "SeiInjector",
    "H264Encoder",
    "SimpleH264Encoder",
    "RtmpStreamer",
    "Mp4Recorder",
    "FrameRingBuffer",
    "RawFrameBuffer",
    "OutputManager",
    "OutputStatistics",

    # Event system
    "SeiEvent",
    "EventTrigger",
    "TriggerCondition",
    "EVENT_TYPE_CROSS_LINE",
    "EVENT_TYPE_INTRUSION",
    "EVENT_TYPE_DETECTION",
    "EVENT_TYPE_ANOMALY",
    "EVENT_TYPE_CUSTOM",
    "SUPPORTED_EVENT_TYPES",

    # Pipeline
    "SeiStreamingPipeline",
    "PushResult",
    "PipelineStatistics",
    "create_pipeline_from_config",
]
