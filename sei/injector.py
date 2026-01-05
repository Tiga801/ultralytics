# -*- coding: utf-8 -*-
"""
SEI Injector Module

This module provides the SEI (Supplemental Enhancement Information) injector
for embedding custom data into H.264 video streams.

Features:
- SEI NAL unit creation with user data unregistered type
- H.264 stream SEI injection
- Configurable UUID identification
- Injection counting and statistics

The injector follows H.264/AVC specification for SEI message embedding,
ensuring compatibility with standard video players and decoders.
"""

from typing import Optional

from .config import SeiConfig, DEFAULT_SEI_UUID
from .interfaces import SeiInjectorInterface, SeiPayload, LogFunc
from .nalutils import (
    make_sei_user_data_unregistered,
    inject_sei_into_h264_data,
    split_nalus,
    inject_sei_nalu
)


class SeiInjector(SeiInjectorInterface):
    """
    SEI Injector Implementation.

    Injects SEI data into H.264 video streams. The injector creates
    SEI NAL units with user data unregistered type (type 5) and
    embeds them before video frame NALs.

    Attributes:
        config: SEI configuration
        log: Logging function

    Example:
        >>> from sei import SeiConfig, SeiInjector, SeiPayload
        >>> config = SeiConfig(enable=True)
        >>> injector = SeiInjector(config)
        >>> payload = SeiPayload(frame_timestamp=time.time())
        >>> enhanced_frame = injector.inject(h264_frame, payload)
    """

    def __init__(
        self,
        config: Optional[SeiConfig] = None,
        log_func: Optional[LogFunc] = None
    ):
        """
        Initialize SEI injector.

        Args:
            config: SEI configuration (optional, uses defaults if not provided)
            log_func: Logging function (optional)
        """
        self._config = config or SeiConfig()
        self._log = log_func or (lambda x: None)
        self._uuid = self._config.uuid
        self._injection_count = 0

    def inject(self, h264_data: bytes, payload: SeiPayload) -> bytes:
        """
        Inject SEI data into H.264 stream.

        The SEI NAL unit is inserted before the first video frame NAL
        in the H.264 data. If SEI is disabled in config, returns the
        original data unchanged.

        Args:
            h264_data: Original H.264 encoded data
            payload: SEI payload containing metadata

        Returns:
            H.264 data with SEI injected
        """
        if not self._config.enable:
            return h264_data

        try:
            # Convert payload to bytes
            payload_bytes = payload.to_bytes()

            # Inject SEI into H.264 stream
            enhanced_data = inject_sei_into_h264_data(
                h264_data,
                payload_bytes,
                self._uuid
            )

            self._injection_count += 1
            return enhanced_data

        except Exception as e:
            self._log(f"SEI injection failed: {e}")
            return h264_data

    def create_sei_nal_unit(self, payload: SeiPayload) -> bytes:
        """
        Create a standalone SEI NAL unit.

        Creates an SEI NAL unit without injecting it into an existing
        stream. Useful for manual stream assembly.

        Args:
            payload: SEI payload

        Returns:
            SEI NAL unit bytes
        """
        payload_bytes = payload.to_bytes()
        return make_sei_user_data_unregistered(payload_bytes, self._uuid)

    def set_uuid(self, uuid_bytes: bytes) -> None:
        """
        Set SEI UUID identifier.

        The UUID is embedded in every SEI NAL unit and can be used
        to identify SEI data from this source.

        Args:
            uuid_bytes: 16-byte UUID (padded/truncated automatically)
        """
        if len(uuid_bytes) < 16:
            uuid_bytes = uuid_bytes.ljust(16, b'\x00')
        elif len(uuid_bytes) > 16:
            uuid_bytes = uuid_bytes[:16]

        self._uuid = uuid_bytes
        self._log(f"SEI UUID set: {uuid_bytes[:16]}")

    @property
    def injection_count(self) -> int:
        """Get total number of successful injections."""
        return self._injection_count

    @property
    def config(self) -> SeiConfig:
        """Get current SEI configuration."""
        return self._config

    @property
    def uuid(self) -> bytes:
        """Get current UUID."""
        return self._uuid

    def reset_count(self) -> None:
        """Reset injection counter to zero."""
        self._injection_count = 0

    def is_enabled(self) -> bool:
        """Check if SEI injection is enabled."""
        return self._config.enable
