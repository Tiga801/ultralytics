# SEI Module

Video streaming pipeline with H.264 encoding, SEI metadata injection, RTMP streaming, and event-triggered MP4 recording.

## Overview

The SEI (Supplemental Enhancement Information) module provides:
- **H.264 encoding** via FFmpeg (libx264 + NVENC hardware acceleration)
- **SEI injection** for embedding custom metadata into H.264 streams
- **RTMP streaming** for real-time video delivery
- **Event-triggered recording** with pre/post-event frame capture
- **Ring buffering** for efficient frame storage

## Architecture

### Component Hierarchy

```
SeiStreamingPipeline
    ├── H264Encoder (FFmpeg subprocess)
    ├── SeiInjector (NAL manipulation)
    └── OutputManager
            ├── FrameRingBuffer (pre-event storage)
            ├── RtmpStreamer (RTMP output)
            └── Mp4Recorder (event recordings)
```

### Data Flow

```
Raw Frame (numpy BGR)
    │
    ▼
H264Encoder.encode() ──▶ H.264 NAL units
    │
    ▼
SeiInjector.inject() ──▶ H.264 + SEI metadata
    │
    ▼
OutputManager.output_frame()
    ├──▶ FrameRingBuffer (always)
    ├──▶ RtmpStreamer (if streaming)
    └──▶ Mp4Recorder (if event active)
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclasses (`EncoderConfig`, `SeiConfig`, `StreamConfig`, etc.) |
| `interfaces.py` | Abstract interfaces and data classes (`SeiPayload`, `EncodedFrame`, `H264EncoderInterface`) |
| `encoder.py` | H.264 encoding via FFmpeg (`H264Encoder`) |
| `injector.py` | SEI metadata injection (`SeiInjector`) |
| `nalutils.py` | H.264 NAL unit parsing and manipulation |
| `buffer.py` | Ring buffer for frame storage (`FrameRingBuffer`) |
| `streamer.py` | RTMP streaming (`RtmpStreamer`) |
| `recorder.py` | MP4 event recording (`Mp4Recorder`) |
| `events.py` | Event system (`SeiEvent`, `EventTrigger`) |
| `output_manager.py` | Dual output coordinator (`OutputManager`) |
| `pipeline.py` | Main orchestrator (`SeiStreamingPipeline`) |

## Key Classes

### SeiStreamingPipeline

Main entry point. Orchestrates encoding, SEI injection, and output.

```python
from sei import SeiStreamingPipeline, SeiConfig, StreamConfig

pipeline = SeiStreamingPipeline(
    sei_config=SeiConfig(enable=True),
    stream_config=StreamConfig(width=1920, height=1080, fps=10)
)
pipeline.start("rtmp://server/live/stream")
pipeline.push_frame(frame, inference_data={"detections": [...]})
pipeline.stop()
```

**Key Methods:**
- `start(rtmp_url)` - Initialize components and start streaming
- `stop()` - Graceful shutdown
- `push_frame(frame, inference_data, custom_data, task_id)` - Process frame synchronously
- `push_frame_async()` - Queue frame for background processing
- `trigger_event(event_type, task_id, metadata)` - Manually trigger recording
- `set_resolution(width, height)` - Update encoding resolution
- `enable_sei()` / `disable_sei()` - Toggle SEI injection

### H264Encoder

Persistent FFmpeg subprocess for frame encoding.

```python
from sei import H264Encoder, EncoderConfig

encoder = H264Encoder(EncoderConfig(preset="ultrafast", use_hardware=True))
encoder.start()
h264_data = encoder.encode(frame)  # numpy BGR array
encoder.stop()
```

**Features:**
- Hardware NVENC support with software fallback
- Persistent process (no per-frame overhead)
- Keyframe detection via `encode_with_keyframe_info()`

### SeiInjector

Injects metadata into H.264 streams as SEI NAL units.

```python
from sei import SeiInjector, SeiConfig, SeiPayload

injector = SeiInjector(SeiConfig(enable=True, uuid=b'CUSTOM_UUID00000'))
payload = SeiPayload(timestamp=time.time(), inference_data={"count": 5})
h264_with_sei = injector.inject(h264_data, payload)
```

### FrameRingBuffer

Thread-safe circular buffer for pre-event frame capture.

```python
from sei import FrameRingBuffer, BufferConfig

buffer = FrameRingBuffer(BufferConfig(capacity=150, pre_event_frames=50))
buffer.push(encoded_frame)
pre_event = buffer.get_pre_event_frames()  # Last 50 frames
```

**Key Methods:**
- `push(frame)` - Add frame (auto-drops oldest if full)
- `get_pre_event_frames()` - Get frames for pre-event capture
- `get_keyframe_and_following()` - Get frames from most recent keyframe

### EventTrigger

Evaluates inference data for event detection.

```python
from sei import EventTrigger

trigger = EventTrigger({
    "type": "detection",
    "classes": ["person"],
    "min_confidence": 0.7,
    "cooldown": 5.0
})
event = trigger.evaluate({"detections": [{"class": "person", "confidence": 0.85}]})
```

**Trigger Types:** `detection`, `count`, `cross_line`, `intrusion`, `custom`

## Configuration

### EncoderConfig

```python
EncoderConfig(
    preset="ultrafast",      # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    profile="baseline",      # baseline, main, high
    level="3.0",
    crf=23,                  # 0-51, lower = better quality
    tune="zerolatency",      # zerolatency, film, animation, grain, stillimage
    use_hardware=True,       # Try NVENC first
    hardware_encoder="h264_nvenc",
    bitrate="4M",            # Optional fixed bitrate
    fallback_to_software=True
)
```

### SeiConfig

```python
SeiConfig(
    enable=True,
    uuid=b'CUSTOM_UUID00000',  # 16 bytes, auto-padded
    insert_interval=1,          # Every N frames
    include_timestamp=True,
    include_frame_count=True
)
```

### StreamConfig

```python
StreamConfig(
    width=1920,
    height=1080,
    fps=10.0,
    buffer_size=100,
    rtmp_url="rtmp://server/live/stream",
    encoder_config=EncoderConfig(...)
)
```

### RecorderConfig

```python
RecorderConfig(
    output_dir="recordings",
    filename_pattern="{task_id}_{timestamp}_{event_type}.mp4",
    pre_event_frames=50,
    post_event_frames=50,
    max_recording_duration=60.0,
    max_concurrent_recordings=3
)
```

### BufferConfig

```python
BufferConfig(
    capacity=150,
    pre_event_frames=50,
    post_event_frames=50,
    overflow_strategy="drop_oldest"
)
```

## SEI Payload Structure

SEI data is embedded as `user_data_unregistered` NAL units (type 5):

```
[Start Code: 0x000001]
[NAL Header: 0x06 (SEI)]
[Payload Type: 0x05 (user_data_unregistered)]
[Payload Size]
[UUID: 16 bytes]
[JSON Data]
[RBSP Trailing: 0x80]
```

Payload JSON format:
```json
{
    "ts": 1234567890.123,
    "it": 1234567890.124,
    "fc": 42,
    "inf": {"detections": [...]},
    "cus": {"custom_field": "value"}
}
```

## Event Types

```python
from sei import (
    EVENT_TYPE_CROSS_LINE,
    EVENT_TYPE_INTRUSION,
    EVENT_TYPE_DETECTION,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_CUSTOM
)
```

## FFmpeg Requirements

### Installation

```bash
# Linux
apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Hardware Acceleration

NVIDIA NVENC requires:
- NVIDIA GPU with NVENC support
- NVIDIA drivers installed
- FFmpeg compiled with `--enable-nvenc`

Check availability:
```bash
ffmpeg -encoders | grep nvenc
```

## NAL Utilities

Low-level H.264 manipulation:

```python
from sei.nalutils import (
    split_nalus,              # Parse NAL boundaries
    get_nal_type,             # Extract NAL type
    is_keyframe_nal,          # Check for IDR frame
    inject_sei_into_h264_data # Insert SEI NAL
)

nalus = split_nalus(h264_data)
for nalu in nalus:
    print(f"NAL type: {get_nal_type(nalu)}")
```

**NAL Type Constants:**
- `NAL_TYPE_NON_IDR` (1) - Non-IDR slice
- `NAL_TYPE_IDR` (5) - IDR slice (keyframe)
- `NAL_TYPE_SEI` (6) - SEI
- `NAL_TYPE_SPS` (7) - Sequence parameter set
- `NAL_TYPE_PPS` (8) - Picture parameter set

## Extension Points

### Custom Encoder

```python
from sei import H264EncoderInterface

class CustomEncoder(H264EncoderInterface):
    def encode(self, frame): ...
    def start(self): ...
    def stop(self): ...
    def is_running(self): ...
    def set_resolution(self, w, h): ...
```

### Custom Event Handler

```python
from sei import EventHandlerInterface

class MyHandler(EventHandlerInterface):
    def on_event_triggered(self, event): ...
    def on_event_ended(self, event): ...
    def on_recording_started(self, event_id, path): ...
    def on_recording_completed(self, event_id, path): ...
```

## Thread Safety

All components are thread-safe:
- `FrameRingBuffer` - `threading.RLock`
- `Mp4Recorder` - `threading.RLock`
- `RtmpStreamer` - `threading.RLock`
- `SeiStreamingPipeline` - Background processing thread with `queue.Queue`

## Statistics

```python
stats = pipeline.get_statistics()
print(f"FPS: {stats.fps}")
print(f"Frames encoded: {stats.frames_encoded}")
print(f"Events triggered: {stats.events_triggered}")

output_stats = pipeline.get_output_statistics()
print(f"Bytes streamed: {output_stats.streaming.bytes_pushed}")
print(f"Recordings completed: {output_stats.recording.recordings_completed}")
```

## Quick Start

### Basic Streaming

```python
from sei import SeiStreamingPipeline, StreamConfig

pipeline = SeiStreamingPipeline(
    stream_config=StreamConfig(width=1920, height=1080, fps=10)
)
pipeline.start("rtmp://localhost/live/test")

while True:
    frame = get_frame()  # Your frame source
    pipeline.push_frame(frame)

pipeline.stop()
```

### Streaming with Event Recording

```python
from sei import SeiStreamingPipeline, StreamConfig, SeiConfig, RecorderConfig

pipeline = SeiStreamingPipeline(
    sei_config=SeiConfig(enable=True),
    stream_config=StreamConfig(width=1920, height=1080, fps=10),
    recorder_config=RecorderConfig(output_dir="events"),
    event_trigger_config={
        "type": "detection",
        "classes": ["person"],
        "min_confidence": 0.7
    }
)
pipeline.start("rtmp://localhost/live/test")

while True:
    frame, inference = get_frame_with_inference()
    result = pipeline.push_frame(frame, inference_data=inference)
    if result.event_triggered:
        print(f"Recording event: {result.event_id}")

pipeline.stop()
```
