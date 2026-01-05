<<<<<<< HEAD
ffmpeg -re -stream_loop -1 -i /home/easyair/ljwork/data/test_video/跨线检测.mp4 -c copy -f rtsp rtsp://192.168.2.71:8554/mystream3 >/dev/null 2>&1 &

ffmpeg -re -stream_loop -1 -i /home/easyair/ljwork/data/test_video/跨线检测.mp4 -c copy -f rtsp rtsp://192.168.2.71:8554/mystream4 >/dev/null 2>&1 &

# 启动跨线检测的3个任务：两个机位、两个摄像头、三个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/cross_line.json

# 启动跨线检测的1个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand1-camera1-crossline1.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand1-camera1-crossline2.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand2-camera2-crossline3.json

# 删除跨线检测的3个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["cross_line_001", "cross_line_002", "cross_line_003"]}'

# 删除跨线检测的1个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["cross_line_001"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["cross_line_002"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["cross_line_003"]}'

# 暂停跨线检测的1个任务
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_001"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_002"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_003"}'

# 恢复跨线检测的1个任务
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_001"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_002"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "cross_line_003"}'



# 启动人脸检测的3个任务：两个机位、两个摄像头、三个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @face_detection/face_detection.json

# 启动人脸检测的1个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand1-camera1-facedetection1.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand1-camera1-facedetection2.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @cross_line/stand2-camera2-facedetection3.json

# 删除人脸检测的3个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_001", "face_detection_002", "face_detection_003"]}'

# 删除人脸检测的1个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_001"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_002"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["face_detection_003"]}'

# 暂停人脸检测的1个任务
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_001"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_002"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_003"}'

# 恢复人脸检测的1个任务
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_001"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_002"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "face_detection_003"}'
=======
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

>>>>>>> 07331326 (feat: build video analytics task management system)

