#!/bin/bash

pkill -f "mystream" 2>/dev/null

# =============================
# 使用说明检查
# =============================
if [ $# -ne 1 ]; then
    echo "Usage: $0 <NUM>"
    echo "Example: $0 5"
    exit 1
fi

NUM=$1

VIDEO="/home/easyair/ljwork/data/test_video/跨线检测.mp4"
RTSP_SERVER="rtsp://192.168.2.71:8554"

# =============================
# 推流
# =============================
for ((i=1; i<=NUM; i++)); do
    STREAM_URL="${RTSP_SERVER}/mystream${i}"

    echo "Starting stream: ${STREAM_URL}"

    ffmpeg -re -stream_loop -1 \
        -i "${VIDEO}" \
        -an \
        -c:v copy \
        -f rtsp \
        -rtsp_transport tcp \
        "${STREAM_URL}" \
        >/dev/null 2>&1 &
done

echo "Started ${NUM} RTSP streams."

