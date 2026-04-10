# Video AI 视频分析服务

基于 YOLO11 和 OpenCV 的实时视频流 AI 分析服务，支持多摄像头注册、目标检测和回调通知。

## 功能特性

- ✅ 支持 RTSP 流媒体实时分析
- ✅ 基于 YOLO11 的目标检测（支持 CAR/CAT/FIRE/PERSON 等类型）
- ✅ 多摄像头注册、更新、删除管理
- ✅ 实时识别结果回调
- ✅ 设备状态心跳回调（每5秒）
- ✅ 支持暂停/恢复分析任务
- ✅ RESTful API 接口
- ✅ 异步高性能处理

## 技术栈

- **Python 3.9+**
- **FastAPI** - Web 框架
- **YOLO11 (Ultralytics)** - 目标检测
- **OpenCV** - 图像处理
- **Redis** - 状态缓存和任务队列
- **Loguru** - 日志记录

## 快速开始

### 1. 安装依赖

```bash
# python版本 3.9.x
# 使用 pip   
pip install -r requirements.txt

```

### 2. 环境配置

复制环境变量模板并配置：

编辑 `.env` 文件，配置以下参数：

```env
# 服务配置
APP_HOST=127.0.0.1
APP_PORT=9007
APP_DEBUG=true

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# YOLO 模型配置
YOLO_MODEL_PATH=yolo11n.pt
YOLO_CONFIDENCE=0.5
YOLO_IOU=0.45

# 回调配置
CALLBACK_TIMEOUT=5
STATUS_INTERVAL=5
```

### 3. 启动服务

```bash
# 生产模式
python -m src.com.mx
```

### 4. API 文档

服务启动后访问：
- Swagger UI: http://localhost:9007/docs
- ReDoc: http://localhost:9007/redoc

## API 接口

### 1. 摄像头注册
- **POST** `/api/video/analysis/register`
- 注册摄像头并启动分析任务

### 2. 摄像头更新
- **PUT** `/api/video/analysis/update`
- 更新摄像头配置

### 3. 摄像头删除
- **DELETE** `/api/video/analysis/delete`
- 删除摄像头并停止分析

### 4. 状态控制
- **PUT** `/api/video/analysis/status`
- 暂停/恢复分析任务

## 回调接口

### 1. AI 识别结果回调
当检测到目标时，系统会 POST 请求到注册的 `callbackUrl`，包含：
- 识别类型和置信度
- 目标位置坐标
- 帧图片 Base64

### 2. 设备状态回调
每5秒发送一次设备状态心跳，包含：
- 设备在线状态
- 分析状态
- 当前识别类型

## 项目结构

```
video_ai/
├── pyproject.toml          # 项目配置和依赖
├── README.md              # 项目说明
├── requirements.txt       # 依赖列表
├── requirements-dev.txt   # 开发依赖
├── src/
│   └── com.mx/
│       ├── __init__.py
│       ├── __main__.py    # 应用入口
│       ├── api/           # API 路由
│       ├── core/          # 核心配置
│       ├── models/        # 数据模型
│       ├── services/      # 业务逻辑
│       ├── utils/         # 工具函数
│       └── db/            # 数据库/缓存
├── tests/                 # 测试代码
├── logs/                  # 日志文件
├── data/                  # 数据文件
└── docs/                  # 文档
```

## 许可证

MIT License