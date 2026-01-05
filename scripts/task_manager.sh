ffmpeg -re -stream_loop -1 -i /home/easyair/ljwork/data/test_video/跨线检测.mp4 -c copy -f rtsp rtsp://192.168.2.71:8554/mystream3 >/dev/null 2>&1 &

ffmpeg -re -stream_loop -1 -i /home/easyair/ljwork/data/test_video/跨线检测.mp4 -c copy -f rtsp rtsp://192.168.2.71:8554/mystream4 >/dev/null 2>&1 &

# 启动采样的3个任务：两个机位、两个摄像头、三个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @sampling/sampling.json

# 启动采样的1个任务
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @sampling/stand1-camera1-sampling1.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @sampling/stand1-camera1-sampling2.json
curl -X POST http://localhost:8666/IAP/runTask -H "Content-Type: application/json" -d @sampling/stand2-camera2-sampling3.json

# 删除采样的3个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["sampling_001", "sampling_002", "sampling_003"]}'

# 删除采样的1个任务
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["sampling_001"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["sampling_002"]}'
curl -X POST http://localhost:8666/IAP/deleteTask -H "Content-Type: application/json" -d '{"taskIds": ["sampling_003"]}'

# 暂停采样的1个任务
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "sampling_001"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "sampling_002"}'
curl -X POST http://localhost:8666/IAP/pauseTask -H "Content-Type: application/json" -d '{"taskId": "sampling_003"}'

# 恢复采样的1个任务
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "sampling_001"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "sampling_002"}'
curl -X POST http://localhost:8666/IAP/resumeTask -H "Content-Type: application/json" -d '{"taskId": "sampling_003"}'
