# -*- coding: utf-8 -*-
"""
H.264 NAL Unit Utilities Module

This module provides utilities for H.264 NAL unit manipulation, including:
- NAL unit splitting and parsing
- SEI NAL unit creation (user data unregistered type)
- SEI injection into H.264 streams
- NAL type identification

NAL Unit Types (H.264/AVC):
- 1-5: Video frames (1=non-IDR slice, 5=IDR slice)
- 6: SEI (Supplemental Enhancement Information)
- 7: SPS (Sequence Parameter Set)
- 8: PPS (Picture Parameter Set)

SEI Structure (user_data_unregistered):
- Start code (3 bytes): 0x00 0x00 0x01
- NAL header (1 byte): 0x06 (SEI type)
- SEI payload type (1 byte): 0x05 (user data unregistered)
- Payload size (variable length encoding)
- UUID (16 bytes)
- User data (variable)
- RBSP trailing bits: 0x80
"""

from typing import List, Optional

# H.264 NAL unit start codes
NAL_START_CODE_3 = b'\x00\x00\x01'      # 3-byte start code
NAL_START_CODE_4 = b'\x00\x00\x00\x01'  # 4-byte start code

# NAL unit type constants
NAL_TYPE_NON_IDR = 1   # Non-IDR slice
NAL_TYPE_IDR = 5       # IDR slice (keyframe)
NAL_TYPE_SEI = 6       # Supplemental Enhancement Information
NAL_TYPE_SPS = 7       # Sequence Parameter Set
NAL_TYPE_PPS = 8       # Picture Parameter Set

# SEI type constants
SEI_TYPE_USER_DATA_UNREGISTERED = 5  # User data unregistered

# Default UUID for SEI identification
DEFAULT_SEI_UUID = b'EASYAIR_UUID'


def split_nalus(data: bytes) -> List[bytes]:
    """
    Split H.264 data stream into individual NAL units.

    Parses the input byte stream and identifies NAL unit boundaries
    based on start codes (0x000001 or 0x00000001).

    Args:
        data: Raw H.264 byte stream

    Returns:
        List of NAL units, each including its start code

    Example:
        >>> data = b'\\x00\\x00\\x00\\x01\\x67...'
        >>> nalus = split_nalus(data)
        >>> len(nalus)
        5
    """
    nalus = []
    i = 0

    while i < len(data):
        # Find NAL start code
        if data[i:i+4] == NAL_START_CODE_4:
            start = i
            i += 4
        elif data[i:i+3] == NAL_START_CODE_3:
            start = i
            i += 3
        else:
            i += 1
            continue

        # Find next NAL start code position
        next4 = data.find(NAL_START_CODE_4, i)
        next3 = data.find(NAL_START_CODE_3, i)

        if next4 == -1 and next3 == -1:
            # No more start codes, remaining data is one NAL unit
            nalus.append(data[start:])
            break

        # Find closest start code position
        nexts = [x for x in [next4, next3] if x != -1]
        next_start = min(nexts) if nexts else len(data)

        # Extract current NAL unit
        nalus.append(data[start:next_start])
        i = next_start

    return nalus


def get_nal_type(nalu: bytes) -> int:
    """
    Get NAL unit type from NAL unit data.

    The NAL type is stored in the lower 5 bits of the NAL header byte,
    which follows the start code.

    Args:
        nalu: NAL unit data including start code

    Returns:
        NAL type (1-31), or 0 if invalid

    Example:
        >>> nalu = b'\\x00\\x00\\x00\\x01\\x67...'
        >>> get_nal_type(nalu)
        7  # SPS
    """
    # Skip start code to find NAL header
    if nalu.startswith(NAL_START_CODE_4):
        header_pos = 4
    elif nalu.startswith(NAL_START_CODE_3):
        header_pos = 3
    else:
        header_pos = 0

    if len(nalu) > header_pos:
        # NAL type is in lower 5 bits of header
        return nalu[header_pos] & 0x1F

    return 0


def is_video_frame_nal(nal_type: int) -> bool:
    """
    Check if NAL type represents a video frame.

    Video frame NAL types are 1-5 (coded slice types).

    Args:
        nal_type: NAL unit type

    Returns:
        True if NAL type is a video frame

    Example:
        >>> is_video_frame_nal(1)  # Non-IDR
        True
        >>> is_video_frame_nal(5)  # IDR
        True
        >>> is_video_frame_nal(6)  # SEI
        False
    """
    return 1 <= nal_type <= 5


def is_keyframe_nal(nal_type: int) -> bool:
    """
    Check if NAL type represents a keyframe (IDR).

    Args:
        nal_type: NAL unit type

    Returns:
        True if NAL type is IDR (keyframe)
    """
    return nal_type == NAL_TYPE_IDR


def make_sei_user_data_unregistered(
    payload_bytes: bytes,
    uuid_bytes: Optional[bytes] = None
) -> bytes:
    """
    Create SEI NAL unit with user data unregistered payload.

    SEI NAL unit structure:
    - NAL start code (3 bytes): 0x00 0x00 0x01
    - NAL header (1 byte): 0x06 (SEI type)
    - SEI payload:
      - Payload type (1 byte): 0x05 (user data unregistered)
      - Payload size (variable length encoding)
      - UUID (16 bytes)
      - User data
      - RBSP trailing bits: 0x80

    Args:
        payload_bytes: User data to embed in SEI
        uuid_bytes: 16-byte UUID for identification (default: EASYAIR_UUID)

    Returns:
        Complete SEI NAL unit bytes

    Example:
        >>> payload = b'{"frame": 1}'
        >>> sei = make_sei_user_data_unregistered(payload)
        >>> sei.startswith(b'\\x00\\x00\\x01\\x06')
        True
    """
    payload_type = SEI_TYPE_USER_DATA_UNREGISTERED
    payload = b''

    # Generate or validate UUID
    if uuid_bytes is None:
        uuid_bytes = DEFAULT_SEI_UUID

    # Always ensure UUID is exactly 16 bytes
    if len(uuid_bytes) < 16:
        # Pad UUID to 16 bytes if too short
        uuid_bytes = uuid_bytes.ljust(16, b'\x00')
    elif len(uuid_bytes) > 16:
        # Truncate UUID to 16 bytes if too long
        uuid_bytes = uuid_bytes[:16]

    # Build SEI payload data: UUID + user data
    payload_data = uuid_bytes + payload_bytes
    payload_size = len(payload_data)

    # Add SEI payload type
    payload += bytes([payload_type])

    # Variable length encoding for payload size
    # H.264 spec: each 0xFF represents 255, final byte is remainder
    sz = payload_size
    while sz >= 0xFF:
        payload += b'\xFF'
        sz -= 0xFF
    payload += bytes([sz])

    # Add payload data
    payload += payload_data

    # Add RBSP trailing bits
    payload += b'\x80'

    # Build complete NAL unit: start code + NAL header (0x06=SEI) + payload
    return NAL_START_CODE_3 + bytes([0x06]) + payload


def inject_sei_nalu(
    nalus: List[bytes],
    sei_payload: bytes,
    uuid_bytes: Optional[bytes] = None
) -> List[bytes]:
    """
    Inject SEI NAL unit into a NAL unit sequence.

    Insertion strategy: Insert SEI before the first video frame NAL (type 1-5).
    This ensures SEI data is associated with the following video frame.

    Args:
        nalus: List of NAL units
        sei_payload: SEI user data (JSON bytes, etc.)
        uuid_bytes: Optional 16-byte UUID

    Returns:
        NAL unit list with SEI inserted

    Example:
        >>> nalus = [sps_nalu, pps_nalu, idr_nalu]
        >>> payload = b'{"frame": 1}'
        >>> new_nalus = inject_sei_nalu(nalus, payload)
        >>> len(new_nalus)
        4  # Added one SEI NAL
    """
    sei_nalu = make_sei_user_data_unregistered(sei_payload, uuid_bytes)

    # Find insertion point (before first video frame NAL)
    for i, nalu in enumerate(nalus):
        nal_type = get_nal_type(nalu)
        if is_video_frame_nal(nal_type):
            # Insert SEI before video frame
            return nalus[:i] + [sei_nalu] + nalus[i:]

    # No suitable position found, append at end
    return nalus + [sei_nalu]


def inject_sei_into_h264_data(
    h264_data: bytes,
    sei_payload: bytes,
    uuid_bytes: Optional[bytes] = None
) -> bytes:
    """
    Inject SEI data into H.264 byte stream.

    Args:
        h264_data: Original H.264 data
        sei_payload: SEI user data
        uuid_bytes: Optional 16-byte UUID

    Returns:
        H.264 data with SEI injected

    Example:
        >>> h264_frame = get_encoded_frame()
        >>> payload = b'{"tracks": [...]}'
        >>> enhanced_frame = inject_sei_into_h264_data(h264_frame, payload)
    """
    # Split into NAL units
    nalus = split_nalus(h264_data)

    if not nalus:
        # No NAL units found, create SEI and prepend to original data
        sei_nalu = make_sei_user_data_unregistered(sei_payload, uuid_bytes)
        return sei_nalu + h264_data

    # Inject SEI
    enhanced_nalus = inject_sei_nalu(nalus, sei_payload, uuid_bytes)

    # Reassemble
    return b''.join(enhanced_nalus)


def extract_sei_from_nalus(nalus: List[bytes]) -> List[bytes]:
    """
    Extract SEI NAL units from NAL unit list.

    Args:
        nalus: List of NAL units

    Returns:
        List of SEI NAL units only
    """
    sei_nalus = []
    for nalu in nalus:
        nal_type = get_nal_type(nalu)
        if nal_type == NAL_TYPE_SEI:
            sei_nalus.append(nalu)
    return sei_nalus


def parse_sei_payload(sei_nalu: bytes) -> Optional[bytes]:
    """
    Parse SEI NAL unit and extract user data.

    Args:
        sei_nalu: SEI NAL unit data

    Returns:
        User data bytes (excluding UUID), or None if parsing fails
    """
    try:
        # Skip start code
        if sei_nalu.startswith(NAL_START_CODE_4):
            pos = 4
        elif sei_nalu.startswith(NAL_START_CODE_3):
            pos = 3
        else:
            pos = 0

        # Skip NAL header
        if pos < len(sei_nalu):
            nal_type = sei_nalu[pos] & 0x1F
            if nal_type != NAL_TYPE_SEI:
                return None
            pos += 1

        # Read SEI payload type
        if pos >= len(sei_nalu):
            return None
        sei_type = sei_nalu[pos]
        pos += 1

        if sei_type != SEI_TYPE_USER_DATA_UNREGISTERED:
            return None

        # Read payload size (variable length encoding)
        payload_size = 0
        while pos < len(sei_nalu):
            byte = sei_nalu[pos]
            pos += 1
            payload_size += byte
            if byte != 0xFF:
                break

        # Skip UUID (16 bytes)
        pos += 16

        # Extract user data
        if pos < len(sei_nalu):
            # Remove possible trailing byte
            user_data = sei_nalu[pos:]
            if user_data.endswith(b'\x80'):
                user_data = user_data[:-1]
            return user_data

        return None

    except Exception:
        return None
