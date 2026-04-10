# Video AI 服务快速开始指南

## 1. 环境准备

### 1.1 安装 Python 3.9+
```bash
# 检查 Python 版本
python --version

# 如果未安装，请从官网下载: https://www.python.org/downloads/
```

### 1.2 安装依赖
```bash
# 安装基础依赖
pip install fastapi uvicorn opencv-python ultralytics requests pydantic loguru redis python-dotenv

# 或者使用 requirements.txt
pip install -r requirements.txt
```

## 2. 快速启动

### 2.1 克隆/下载项目
```bash
# 如果从Git克隆
git clone <repository-url>
cd video_ai
```

### 2.2 创建配置文件
```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 编辑配置文件（可选）
# 修改 Redis 地址、端口等配置
```

### 2.3 启动服务
```bash
# 开发模式（带热重载）
uvicorn src.com.mx.__main__:app --reload --host 0.0.0.0 --port 8000

# 或者使用模块方式
python -m src.com.mx
```

### 2.4 验证服务
打开浏览器访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## 3. 基本使用

### 3.1 注册摄像头
```bash
curl -X POST "http://localhost:8000/api/video/analysis/register" \
  -H "Content-Type: application/json" \
  -d '{
    "rtspUrl": "rtsp://your-camera-ip:554/stream",
    "types": ["PERSON", "CAR"],
    "deviceId": "camera_001",
    "callbackUrl": "http://your-server.com/callback"
  }'
```

### 3.2 查看设备列表
```bash
curl "http://localhost:8000/api/video/analysis/list"
```

### 3.3 暂停/恢复设备
```bash
# 暂停设备
curl -X PUT "http://localhost:8000/api/video/analysis/status" \
  -H "Content-Type: application/json" \
  -d '{
    "deviceId": "camera_001",
    "operate": "PAUSE"
  }'

# 恢复设备
curl -X PUT "http://localhost:8000/api/video/analysis/status" \
  -H "Content-Type: application/json" \
  -d '{
    "deviceId": "camera_001",
    "operate": "RESUME"
  }'
```

### 3.4 删除设备
```bash
curl -X DELETE "http://localhost:8000/api/video/analysis/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "deviceId": "camera_001"
  }'
```

## 4. 回调接收示例

### 4.1 创建简单的回调服务器（Python Flask）
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    data = request.json
    print(f"收到回调: {data['callbackType']}")
    
    if data['callbackType'] == 'RECOGNIZE':
        print(f"识别到 {data['recognizeType']}, 置信度: {data['confidence']}")
    elif data['callbackType'] == 'STATUS':
        print(f"设备状态: {data['deviceStatus']}")
    
    return jsonify({"code": 200, "msg": "接收成功"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 4.2 启动回调服务器
```bash
python callback_server.py
```

## 5. Docker 部署

### 5.1 使用 Docker Compose
```bash
# 启动所有服务
docker-compose -f scripts/docker-compose.yml up -d

# 查看日志
docker-compose -f scripts/docker-compose.yml logs -f

# 停止服务
docker-compose -f scripts/docker-compose.yml down
```

### 5.2 单独构建 Docker 镜像
```bash
# 构建镜像
docker build -t video-ai .

# 运行容器
docker run -p 8000:8000 --env-file .env video-ai
```

## 6. 测试 RTSP 流

### 6.1 使用测试 RTSP 流
```bash
# 安装 RTSP 测试服务器
pip install rtsp-simple-server

# 或者使用 Docker
docker run -p 8554:8554 aler9/rtsp-simple-server
```

### 6.2 推流测试视频
```bash
# 使用 FFmpeg 推送测试视频
ffmpeg -re -stream_loop -1 -i test.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream
```

## 7. 故障排除

### 7.1 常见问题

1. **RTSP 连接失败**
   - 检查网络连通性
   - 验证 RTSP URL 格式
   - 检查摄像头认证信息

2. **YOLO 模型加载失败**
   - 检查网络连接
   - 手动下载模型: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   - 放置到项目根目录

3. **Redis 连接失败**
   - 检查 Redis 服务是否运行
   - 验证配置中的 Redis 地址和端口
   - 检查防火墙设置

### 7.2 查看日志
```bash
# 查看服务日志
tail -f logs/video_ai.log

# 查看错误日志
tail -f logs/error.log
```

## 8. 下一步

1. **阅读完整文档**: 查看 `docs/api.md` 获取详细 API 说明
2. **配置生产环境**: 修改配置文件，设置合适的参数
3. **添加认证**: 为 API 添加认证机制
4. **监控和告警**: 集成 Prometheus 和 Grafana
5. **扩展功能**: 添加更多识别类型和业务逻辑

## 9. 获取帮助

- 查看项目 README.md
- 查看 API 文档: http://localhost:8000/docs
- 查看源码注释
- 提交 Issue 到项目仓库