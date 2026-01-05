# -*- coding: utf-8 -*-
"""
SEI Video Streaming Configuration Module

This module provides configuration dataclasses for the SEI video streaming
pipeline, including encoder, streaming, buffer, and recording configurations.

Configuration Hierarchy:
- EncoderConfig: H.264 encoder parameters
- SeiConfig: SEI injection settings
- StreamConfig: RTMP streaming parameters
- BufferConfig: Ring buffer settings
- RecorderConfig: Event-triggered recording settings
- OutputConfig: Combined output configuration
- SeiPipelineConfig: Complete pipeline configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# Default UUID for SEI identification (16 bytes)
DEFAULT_SEI_UUID = b'EASYAIR_UUID'


@dataclass
class EncoderConfig:
    """
    H.264 Encoder Configuration.

    Controls FFmpeg encoder parameters for video encoding. Supports both
    software (libx264) and hardware (NVENC) encoding.

    Attributes:
        preset: Encoding preset (ultrafast, superfast, veryfast, faster,
                fast, medium, slow, slower, veryslow) for libx264,
                or (p1-p7) for NVENC where p1=fastest, p7=quality
        profile: H.264 profile (baseline, main, high)
        level: H.264 level (e.g., "3.0", "4.0", "4.1")
        crf: Constant Rate Factor (0-51, lower = better quality, default 23)
             Only used for libx264 software encoding
        tune: Tuning option (zerolatency, film, animation, grain, etc.)
              For NVENC, use "ll" (low latency) or "ull" (ultra low latency)
        pixel_format: Output pixel format (default yuv420p)
        gop_size: GOP size in frames (0 = auto based on fps)
        keyint_min: Minimum keyframe interval
        use_hardware: Enable hardware encoding (NVENC) if available
        hardware_encoder: Hardware encoder name (h264_nvenc for NVIDIA)
        fallback_to_software: Fallback to libx264 if hardware unavailable
        bitrate: Target bitrate for hardware encoding (e.g., "4M", "8M")
    """
    preset: str = "ultrafast"
    profile: str = "baseline"
    level: str = "3.0"
    crf: int = 23
    tune: str = "zerolatency"
    pixel_format: str = "yuv420p"
    gop_size: int = 0  # 0 = auto (fps * 2)
    keyint_min: int = 1
    use_hardware: bool = True
    hardware_encoder: str = "h264_nvenc"
    fallback_to_software: bool = True
    bitrate: str = "4M"

    def to_ffmpeg_args(self) -> List[str]:
        """Convert to FFmpeg command line arguments."""
        args = [
            "-preset", self.preset,
            "-profile:v", self.profile,
            "-level", self.level,
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format
        ]
        if self.tune:
            args.extend(["-tune", self.tune])
        return args

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EncoderConfig":
        """Create configuration from dictionary."""
        return cls(
            preset=config_dict.get("preset", "ultrafast"),
            profile=config_dict.get("profile", "baseline"),
            level=config_dict.get("level", "3.0"),
            crf=config_dict.get("crf", 23),
            tune=config_dict.get("tune", "zerolatency"),
            pixel_format=config_dict.get("pixel_format", "yuv420p"),
            gop_size=config_dict.get("gop_size", 0),
            keyint_min=config_dict.get("keyint_min", 1),
            use_hardware=config_dict.get("use_hardware", True),
            hardware_encoder=config_dict.get("hardware_encoder", "h264_nvenc"),
            fallback_to_software=config_dict.get("fallback_to_software", True),
            bitrate=config_dict.get("bitrate", "4M")
        )


@dataclass
class SeiConfig:
    """
    SEI Injection Configuration.

    Controls SEI (Supplemental Enhancement Information) injection behavior.

    Attributes:
        enable: Whether to enable SEI injection
        uuid: 16-byte UUID for SEI identification
        insert_interval: Insert SEI every N frames (1 = every frame)
        include_timestamp: Include frame timestamp in SEI
        include_frame_count: Include frame count in SEI
        custom_uuid_string: Custom UUID as string (converted to bytes)
    """
    enable: bool = True
    uuid: bytes = field(default_factory=lambda: DEFAULT_SEI_UUID)
    insert_interval: int = 1
    include_timestamp: bool = True
    include_frame_count: bool = True
    custom_uuid_string: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert custom UUID string to bytes if provided
        if self.custom_uuid_string:
            uuid_bytes = self.custom_uuid_string.encode('utf-8')[:16]
            self.uuid = uuid_bytes.ljust(16, b'\x00')

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SeiConfig":
        """Create configuration from dictionary."""
        return cls(
            enable=config_dict.get("enable", config_dict.get("sei_enabled", True)),
            insert_interval=config_dict.get("insert_interval", config_dict.get("sei_insert_interval", 1)),
            include_timestamp=config_dict.get("include_timestamp", config_dict.get("sei_include_timestamp", True)),
            include_frame_count=config_dict.get("include_frame_count", config_dict.get("sei_include_frame_count", True)),
            custom_uuid_string=config_dict.get("uuid", config_dict.get("sei_uuid"))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable": self.enable,
            "insert_interval": self.insert_interval,
            "include_timestamp": self.include_timestamp,
            "include_frame_count": self.include_frame_count,
            "uuid": self.uuid.decode('utf-8', errors='replace').rstrip('\x00')
        }


@dataclass
class StreamConfig:
    """
    RTMP Streaming Configuration.

    Controls RTMP streaming parameters and connection settings.

    Attributes:
        rtmp_url: RTMP server URL for streaming
        width: Video width in pixels
        height: Video height in pixels
        fps: Frame rate (frames per second)
        buffer_size: Frame buffer size for async streaming
        encoder_config: H.264 encoder configuration
        ffmpeg_path: Path to FFmpeg executable
        retry_interval: Seconds between reconnection attempts
        max_retries: Maximum reconnection attempts
    """
    rtmp_url: str = ""
    width: int = 1920
    height: int = 1080
    fps: float = 10.0
    buffer_size: int = 100
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    ffmpeg_path: str = "ffmpeg"
    retry_interval: float = 5.0
    max_retries: int = 10

    @property
    def frame_interval(self) -> float:
        """Calculate frame interval in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StreamConfig":
        """Create configuration from dictionary."""
        # Process encoder configuration
        encoder_dict = config_dict.get("encoder", {})
        encoder_config = EncoderConfig.from_dict(encoder_dict)

        return cls(
            rtmp_url=config_dict.get("rtmp_url", ""),
            width=config_dict.get("width", 1920),
            height=config_dict.get("height", 1080),
            fps=config_dict.get("fps", 10.0),
            buffer_size=config_dict.get("buffer_size", 100),
            encoder_config=encoder_config,
            ffmpeg_path=config_dict.get("ffmpeg_path", "ffmpeg"),
            retry_interval=config_dict.get("retry_interval", 5.0),
            max_retries=config_dict.get("max_retries", 10)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rtmp_url": self.rtmp_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "buffer_size": self.buffer_size,
            "ffmpeg_path": self.ffmpeg_path,
            "retry_interval": self.retry_interval,
            "max_retries": self.max_retries,
            "encoder": {
                "preset": self.encoder_config.preset,
                "profile": self.encoder_config.profile,
                "level": self.encoder_config.level,
                "crf": self.encoder_config.crf,
                "tune": self.encoder_config.tune,
                "pixel_format": self.encoder_config.pixel_format
            }
        }


@dataclass
class BufferConfig:
    """
    Ring Buffer Configuration.

    Controls the ring buffer for frame storage, used for event-triggered
    recording to capture frames before and after events.

    Attributes:
        capacity: Total buffer capacity in frames
        pre_event_frames: Number of frames to retain before event trigger
        post_event_frames: Number of frames to capture after event trigger
        overflow_strategy: How to handle buffer overflow ('drop_oldest' or 'block')
    """
    capacity: int = 150
    pre_event_frames: int = 50
    post_event_frames: int = 50
    overflow_strategy: str = "drop_oldest"

    def __post_init__(self):
        """Validate configuration."""
        if self.capacity < self.pre_event_frames + self.post_event_frames:
            self.capacity = self.pre_event_frames + self.post_event_frames + 10

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BufferConfig":
        """Create configuration from dictionary."""
        return cls(
            capacity=config_dict.get("capacity", 150),
            pre_event_frames=config_dict.get("pre_event_frames", 50),
            post_event_frames=config_dict.get("post_event_frames", 50),
            overflow_strategy=config_dict.get("overflow_strategy", "drop_oldest")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capacity": self.capacity,
            "pre_event_frames": self.pre_event_frames,
            "post_event_frames": self.post_event_frames,
            "overflow_strategy": self.overflow_strategy
        }


@dataclass
class RecorderConfig:
    """
    Event-Triggered Recording Configuration.

    Controls MP4 recording parameters for event-triggered video saving.

    Attributes:
        output_dir: Base directory for recorded files
        filename_pattern: Pattern for output filenames
            Available placeholders: {task_id}, {timestamp}, {event_type}, {event_id}
        pre_event_frames: Frames to include before event (from buffer)
        post_event_frames: Frames to record after event trigger
        max_recording_duration: Maximum recording duration in seconds
        codec: Video codec for recording
        container: Container format (mp4, mkv)
        enable_sei_in_recording: Include SEI data in recordings
        ffmpeg_path: Path to FFmpeg executable
        max_concurrent_recordings: Maximum simultaneous recordings
    """
    output_dir: str = "recordings"
    filename_pattern: str = "{task_id}_{timestamp}_{event_type}.mp4"
    pre_event_frames: int = 50
    post_event_frames: int = 50
    max_recording_duration: float = 60.0
    codec: str = "libx264"
    container: str = "mp4"
    enable_sei_in_recording: bool = True
    ffmpeg_path: str = "ffmpeg"
    max_concurrent_recordings: int = 3

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RecorderConfig":
        """Create configuration from dictionary."""
        return cls(
            output_dir=config_dict.get("output_dir", "recordings"),
            filename_pattern=config_dict.get("filename_pattern", "{task_id}_{timestamp}_{event_type}.mp4"),
            pre_event_frames=config_dict.get("pre_event_frames", 50),
            post_event_frames=config_dict.get("post_event_frames", 50),
            max_recording_duration=config_dict.get("max_recording_duration", 60.0),
            codec=config_dict.get("codec", "libx264"),
            container=config_dict.get("container", "mp4"),
            enable_sei_in_recording=config_dict.get("enable_sei_in_recording", True),
            ffmpeg_path=config_dict.get("ffmpeg_path", "ffmpeg"),
            max_concurrent_recordings=config_dict.get("max_concurrent_recordings", 3)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "filename_pattern": self.filename_pattern,
            "pre_event_frames": self.pre_event_frames,
            "post_event_frames": self.post_event_frames,
            "max_recording_duration": self.max_recording_duration,
            "codec": self.codec,
            "container": self.container,
            "enable_sei_in_recording": self.enable_sei_in_recording,
            "ffmpeg_path": self.ffmpeg_path,
            "max_concurrent_recordings": self.max_concurrent_recordings
        }


@dataclass
class OutputConfig:
    """
    Unified Output Configuration.

    Combines RTMP streaming and event recording configurations.

    Attributes:
        rtmp_enabled: Enable RTMP streaming output
        recording_enabled: Enable event-triggered recording
        stream_config: RTMP streaming configuration
        recorder_config: Recording configuration
        buffer_config: Ring buffer configuration
    """
    rtmp_enabled: bool = True
    recording_enabled: bool = True
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    recorder_config: RecorderConfig = field(default_factory=RecorderConfig)
    buffer_config: BufferConfig = field(default_factory=BufferConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OutputConfig":
        """Create configuration from dictionary."""
        return cls(
            rtmp_enabled=config_dict.get("rtmp_enabled", True),
            recording_enabled=config_dict.get("recording_enabled", True),
            stream_config=StreamConfig.from_dict(config_dict.get("stream", {})),
            recorder_config=RecorderConfig.from_dict(config_dict.get("recorder", {})),
            buffer_config=BufferConfig.from_dict(config_dict.get("buffer", {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rtmp_enabled": self.rtmp_enabled,
            "recording_enabled": self.recording_enabled,
            "stream": self.stream_config.to_dict(),
            "recorder": self.recorder_config.to_dict(),
            "buffer": self.buffer_config.to_dict()
        }


@dataclass
class SeiPipelineConfig:
    """
    Complete SEI Pipeline Configuration.

    Top-level configuration combining all pipeline components.

    Attributes:
        sei_config: SEI injection configuration
        encoder_config: H.264 encoder configuration
        output_config: Output (streaming + recording) configuration
        log_file: Log file path (relative to logs directory)
    """
    sei_config: SeiConfig = field(default_factory=SeiConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    log_file: str = "sei.log"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SeiPipelineConfig":
        """Create configuration from dictionary."""
        sei_dict = config_dict.get("sei", {})
        encoder_dict = config_dict.get("encoder", {})
        output_dict = config_dict.get("output", {})

        # Handle flat configuration format
        if not sei_dict and not encoder_dict and not output_dict:
            sei_dict = {k: v for k, v in config_dict.items() if k.startswith("sei_")}
            encoder_dict = config_dict.get("encoder", {})
            output_dict = config_dict

        return cls(
            sei_config=SeiConfig.from_dict(sei_dict or config_dict),
            encoder_config=EncoderConfig.from_dict(encoder_dict or config_dict),
            output_config=OutputConfig.from_dict(output_dict or config_dict),
            log_file=config_dict.get("log_file", "sei.log")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sei": self.sei_config.to_dict(),
            "encoder": {
                "preset": self.encoder_config.preset,
                "profile": self.encoder_config.profile,
                "level": self.encoder_config.level,
                "crf": self.encoder_config.crf,
                "tune": self.encoder_config.tune,
                "pixel_format": self.encoder_config.pixel_format
            },
            "output": self.output_config.to_dict(),
            "log_file": self.log_file
        }


# Legacy compatibility alias
SeiStreamingConfig = SeiPipelineConfig
