"""
应用配置管理
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用信息
    VERSION: str = "2026.0.0"
    APP_NAME: str = "Video AI Service"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 9007
    APP_DEBUG: bool = True
    
    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # YOLO 配置
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU: float = 0.45

    # 模型路径配置
    YOLO_ANIMAL_MODEL_PATH: str = "yolo11m.pt"
    YOLO_FIRE_MODEL_PATH: str = "yolo11-firedetect.pt"
    YOLO_HARDHAT_MODEL_PATH: str = "yolo11-hardhat.pt"

    # 回调配置
    CALLBACK_TIMEOUT: int = 5  # 回调超时时间（秒）
    STATUS_INTERVAL: int = 5   # 状态回调间隔（秒）

    # 视频流配置
    RTSP_TIMEOUT: int = 10  # RTSP 连接超时（秒）
    FRAME_SKIP: int = 5     # 跳帧处理，每N帧处理一次
    MAX_CONSECUTIVE_READ_FAILURES: int = 10   # 连续读帧失败触发重连阈值
    INITIAL_RECONNECT_DELAY: int = 3          # 初始重连等待秒数
    MAX_RECONNECT_DELAY: int = 30             # 最大重连等待秒数
    PERSON_STAY_THRESHOLD: float = 5.0        # PERSON 停留阈值（秒）
    PERSON_ABSENCE_RESET: float = 2.0         # PERSON 视为离开并重置状态的时间（秒）
    
    # GPU 加速配置
    USE_GPU_DECODE: bool = True              # 是否启用 GPU 视频解码加速
    GSTREAMER_LATENCY: int = 100             # GStreamer 管道延迟（毫秒）
    
    # 支持的识别类型
    SUPPORTED_TYPES: List[str] = ["CAT", "DOG", "FIRE", "HARDHAT", "NO_HARDHAT", "PERSON"]

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    class Config:
        env_file = ".env"
        case_sensitive = True

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "SUPPORTED_TYPES":
                if raw_val.startswith("[") and raw_val.endswith("]"):
                    return json.loads(raw_val)
                else:
                    return [item.strip() for item in raw_val.split(",") if item.strip()]
            return json.loads(raw_val) if raw_val.startswith("[") or raw_val.startswith("{") else raw_val

    @validator("LOGS_DIR", "DATA_DIR", "MODELS_DIR", pre=True)
    def create_dirs(cls, v: Path) -> Path:
        """创建必要的目录"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def redis_url(self) -> str:
        """Redis 连接 URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


# 全局配置实例
settings = Settings()

