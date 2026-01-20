# 手动推流
ffmpeg -re -stream_loop -1 -i /home/easyair/ljwork/data/test_video/跨线检测.mp4 -an -c:v copy -f rtsp rtsp://192.168.2.71:8554/mystream1 >/dev/null 2>&1 &


# 启动任务（单个、多个均可）
curl -X POST http://localhost:8555/IAP/runTask -H "Content-Type: application/json" -d @face_detection/face_detection.json
curl -X POST http://localhost:8555/IAP/runTask -H "Content-Type: application/json" -d @cross_line/cross_line.json

# 删除任务（单个、多个均可）
curl -X POST http://localhost:8555/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_001"]}'
curl -X POST http://localhost:8555/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["cross_line_001"]}'
curl -X POST http://localhost:8555/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_001", "face_detection_002", "face_detection_003", "face_detection_004"]}'
curl -X POST http://localhost:8555/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_005", "face_detection_006", "face_detection_007", "face_detection_008"]}'


# 暂停任务（仅支持单个）
curl -X POST http://localhost:8555/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_001"}'

# 恢复任务（仅支持单个）
curl -X POST http://localhost:8555/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_001"}'


