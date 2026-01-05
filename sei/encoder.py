# -*- coding: utf-8 -*-
"""
H.264 Encoder Module

This module provides H.264 video encoding using FFmpeg. Supports both
persistent encoder processes for high-throughput streaming and simple
per-frame encoding for lower-frequency scenarios.

Features:
- FFmpeg-based H.264 encoding (libx264)
- Persistent encoder process (reduced latency)
- Dynamic resolution handling
- Keyframe detection
- Configurable encoding parameters

Encoding Pipeline:
    Raw BGR Frame (numpy.ndarray)
        │
        ▼
    FFmpeg stdin (rawvideo)
        │
        ▼
    libx264 encoder
        │
        ▼
    H.264 NAL units

Usage:
    >>> from sei.encoder import H264Encoder
    >>> from sei.config import StreamConfig
    >>> config = StreamConfig(width=1920, height=1080, fps=10)
    >>> encoder = H264Encoder(config)
    >>> encoder.start()
    >>> h264_data = encoder.encode(frame)
    >>> encoder.stop()
"""

import subprocess
import threading
import time
import select
from typing import Optional, Any, Tuple

from .config import StreamConfig, EncoderConfig
from .interfaces import H264EncoderInterface, LogFunc
from .nalutils import split_nalus, get_nal_type, NAL_TYPE_IDR


class H264Encoder(H264EncoderInterface):
    """
    Persistent H.264 Encoder.

    Uses a persistent FFmpeg process for video encoding. The process
    remains running between frames, reducing startup overhead and
    enabling proper GOP handling.

    FFmpeg Command:
        ffmpeg -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - [libx264 options] -f h264 -

    Attributes:
        config: Stream configuration
        log: Logging function

    Example:
        >>> from sei.encoder import H264Encoder
        >>> from sei.config import StreamConfig
        >>> config = StreamConfig(width=1920, height=1080, fps=10)
        >>> encoder = H264Encoder(config)
        >>> encoder.start()
        >>> for frame in frames:
        ...     h264_data = encoder.encode(frame)
        ...     if h264_data:
        ...         process(h264_data)
        >>> encoder.stop()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize H.264 encoder.

        Args:
            config: Stream configuration
            log_func: Logging function
        """
        self._config = config or StreamConfig()
        self._log = log_func or self._default_log

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_count = 0

        self._width = self._config.width
        self._height = self._config.height

        # Read buffer for encoded data
        self._read_buffer = b''
        self._buffer_lock = threading.Lock()

        # Hardware encoder availability (checked on start)
        self._use_nvenc = False
        self._nvenc_checked = False

    @staticmethod
    def _default_log(message: str) -> None:
        """Default logging to stdout with timestamp."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def _check_nvenc_available(self) -> bool:
        """
        Check if NVIDIA NVENC hardware encoder is available.

        Returns:
            True if h264_nvenc is available in FFmpeg
        """
        if self._nvenc_checked:
            return self._use_nvenc

        self._nvenc_checked = True
        encoder_config = self._config.encoder_config

        # Skip check if hardware encoding is disabled
        if not encoder_config.use_hardware:
            self._use_nvenc = False
            return False

        try:
            result = subprocess.run(
                [self._config.ffmpeg_path, '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            hardware_encoder = encoder_config.hardware_encoder
            self._use_nvenc = hardware_encoder in result.stdout

            if self._use_nvenc:
                self._log(f"Hardware encoder '{hardware_encoder}' available")
            else:
                if encoder_config.fallback_to_software:
                    self._log(f"Hardware encoder '{hardware_encoder}' not available, using libx264")
                else:
                    self._log(f"Hardware encoder '{hardware_encoder}' not available")

            return self._use_nvenc

        except subprocess.TimeoutExpired:
            self._log("FFmpeg encoder check timed out")
            self._use_nvenc = False
            return False
        except Exception as e:
            self._log(f"Failed to check hardware encoder: {e}")
            self._use_nvenc = False
            return False

    def start(self) -> bool:
        """
        Start encoder process.

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._running:
                return True

            try:
                # Check hardware encoder availability
                self._check_nvenc_available()

                self._process = self._create_encoder_process()
                self._running = True
                self._frame_count = 0
                self._read_buffer = b''

                encoder_type = "NVENC" if self._use_nvenc else "libx264"
                self._log(f"H.264 encoder started ({encoder_type})")
                return True

            except Exception as e:
                self._log(f"Encoder start failed: {e}")
                return False

    def stop(self) -> None:
        """Stop encoder process."""
        with self._lock:
            self._running = False

            if self._process:
                try:
                    self._process.stdin.close()
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except Exception as e:
                    self._log(f"Encoder stop error: {e}")
                    try:
                        self._process.kill()
                    except:
                        pass
                finally:
                    self._process = None

            self._log("H.264 encoder stopped")

    def encode(self, frame: Any) -> Optional[bytes]:
        """
        Encode single video frame.

        Args:
            frame: OpenCV BGR image (numpy.ndarray)

        Returns:
            H.264 encoded bytes, or None on failure
        """
        if not self._running or not self._process:
            return None

        try:
            # Validate frame dimensions
            if hasattr(frame, 'shape'):
                h, w = frame.shape[:2]
                if w != self._width or h != self._height:
                    self._log(f"Frame size mismatch: {w}x{h} vs {self._width}x{self._height}")

            # Write raw frame to encoder
            frame_bytes = frame.tobytes()
            self._process.stdin.write(frame_bytes)
            self._process.stdin.flush()

            # Read encoded data
            encoded_data = self._read_encoded_data()

            if encoded_data:
                self._frame_count += 1

            return encoded_data

        except BrokenPipeError:
            self._log("Encoder pipe broken, restarting...")
            self._restart_encoder()
            return None

        except Exception as e:
            self._log(f"Encode error: {e}")
            return None

    def encode_with_keyframe_info(self, frame: Any) -> Tuple[Optional[bytes], bool]:
        """
        Encode frame and detect if result contains keyframe.

        Args:
            frame: OpenCV BGR image

        Returns:
            Tuple of (encoded_data, is_keyframe)
        """
        encoded_data = self.encode(frame)
        if encoded_data is None:
            return None, False

        is_keyframe = self._detect_keyframe(encoded_data)
        return encoded_data, is_keyframe

    def is_running(self) -> bool:
        """Check if encoder is running."""
        return self._running and self._process is not None

    def set_resolution(self, width: int, height: int) -> None:
        """
        Set encoding resolution.

        Restarts encoder if resolution changes while running.

        Args:
            width: Video width in pixels
            height: Video height in pixels
        """
        if width != self._width or height != self._height:
            self._width = width
            self._height = height

            if self._running:
                self._log(f"Resolution changed to {width}x{height}, restarting encoder")
                self._restart_encoder()

    @property
    def frame_count(self) -> int:
        """Get total encoded frames."""
        return self._frame_count

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        return (self._width, self._height)

    def _create_encoder_process(self) -> subprocess.Popen:
        """
        Create FFmpeg encoder process.

        Supports both NVENC hardware encoding and libx264 software encoding.
        NVENC is used when available and enabled, otherwise falls back to libx264.

        Returns:
            FFmpeg subprocess
        """
        encoder_config = self._config.encoder_config

        # Calculate GOP size
        gop_size = encoder_config.gop_size
        if gop_size <= 0:
            gop_size = int(self._config.fps * 2)

        # Base command (input)
        cmd = [
            self._config.ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self._width}x{self._height}',
            '-r', str(self._config.fps),
            '-i', '-',
        ]

        if self._use_nvenc:
            # NVIDIA NVENC hardware encoding
            # Map software presets to NVENC presets (p1=fastest, p7=quality)
            nvenc_preset_map = {
                'ultrafast': 'p1',
                'superfast': 'p1',
                'veryfast': 'p2',
                'faster': 'p3',
                'fast': 'p4',
                'medium': 'p5',
                'slow': 'p6',
                'slower': 'p7',
                'veryslow': 'p7',
            }
            nvenc_preset = nvenc_preset_map.get(encoder_config.preset, 'p1')

            # Map software tune to NVENC tune
            nvenc_tune_map = {
                'zerolatency': 'll',      # low latency
                'fastdecode': 'll',
                'film': 'hq',             # high quality
                'animation': 'hq',
            }
            nvenc_tune = nvenc_tune_map.get(encoder_config.tune, 'll')

            cmd.extend([
                '-c:v', encoder_config.hardware_encoder,
                '-preset', nvenc_preset,
                '-tune', nvenc_tune,
                '-rc', 'cbr',              # Constant bitrate for streaming
                '-b:v', encoder_config.bitrate,
                '-profile:v', encoder_config.profile,
                '-g', str(gop_size),
                '-pix_fmt', encoder_config.pixel_format,
                '-f', 'h264',
                '-'
            ])
        else:
            # libx264 software encoding (original behavior)
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', encoder_config.preset,
                '-profile:v', encoder_config.profile,
                '-level', encoder_config.level,
                '-crf', str(encoder_config.crf),
                '-pix_fmt', encoder_config.pixel_format,
                '-f', 'h264',
                '-tune', encoder_config.tune,
                '-g', str(gop_size),
                '-keyint_min', str(encoder_config.keyint_min),
                '-sc_threshold', '0',
                '-'
            ])

        self._log(f"Encoder command: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )

        return process

    def _read_encoded_data(self) -> Optional[bytes]:
        """
        Read all available encoded data from FFmpeg stdout.

        Uses non-blocking read with select to drain all available data,
        preventing buffer accumulation that could block the encoder.
        Timeout is adjusted based on fps to accommodate NVENC latency at low frame rates.

        Returns:
            Encoded bytes, or None if no data available
        """
        if not self._process or not self._process.stdout:
            return None

        try:
            all_data = b''
            # Calculate timeout based on frame rate
            # At low fps, NVENC needs more time to produce output
            frame_interval = 1.0 / self._config.fps if self._config.fps > 0 else 0.5
            per_iteration_timeout = max(0.1, frame_interval * 0.5)  # At least 100ms, or half frame interval

            # Keep reading while data is available
            while True:
                ready, _, _ = select.select([self._process.stdout], [], [], per_iteration_timeout)
                if not ready:
                    break
                chunk = self._process.stdout.read(65536)
                if not chunk:
                    break
                all_data += chunk

            return all_data if all_data else None

        except Exception as e:
            self._log(f"Read error: {e}")
            return None

    def _restart_encoder(self) -> None:
        """Restart encoder process."""
        self.stop()
        time.sleep(0.1)
        self.start()

    def _detect_keyframe(self, h264_data: bytes) -> bool:
        """
        Detect if H.264 data contains a keyframe (IDR).

        Args:
            h264_data: H.264 encoded bytes

        Returns:
            True if data contains IDR NAL unit
        """
        nalus = split_nalus(h264_data)
        for nalu in nalus:
            nal_type = get_nal_type(nalu)
            if nal_type == NAL_TYPE_IDR:
                return True
        return False


class SimpleH264Encoder(H264EncoderInterface):
    """
    Simple H.264 Encoder (per-frame process).

    Creates a new FFmpeg process for each frame. Suitable for
    low-frequency encoding scenarios where latency is not critical.

    Note: Less efficient than H264Encoder for high-throughput use.

    Example:
        >>> encoder = SimpleH264Encoder(config)
        >>> encoder.start()
        >>> h264_data = encoder.encode(frame)
        >>> encoder.stop()
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize simple encoder.

        Args:
            config: Stream configuration
            log_func: Logging function
        """
        self._config = config or StreamConfig()
        self._log = log_func or (lambda x: None)
        self._running = False
        self._width = self._config.width
        self._height = self._config.height

    def start(self) -> bool:
        """Start encoder (sets running flag)."""
        self._running = True
        return True

    def stop(self) -> None:
        """Stop encoder (clears running flag)."""
        self._running = False

    def encode(self, frame: Any) -> Optional[bytes]:
        """
        Encode frame using temporary FFmpeg process.

        Args:
            frame: OpenCV BGR image

        Returns:
            H.264 encoded bytes, or None on failure
        """
        if not self._running:
            return None

        try:
            h, w = frame.shape[:2]

            cmd = [
                self._config.ffmpeg_path,
                '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}',
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-f', 'h264',
                '-'
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            encoded_data, _ = process.communicate(
                input=frame.tobytes(),
                timeout=10
            )

            return encoded_data if encoded_data else None

        except Exception as e:
            self._log(f"Encode error: {e}")
            return None

    def is_running(self) -> bool:
        """Check if encoder is running."""
        return self._running

    def set_resolution(self, width: int, height: int) -> None:
        """Set resolution for encoding."""
        self._width = width
        self._height = height
