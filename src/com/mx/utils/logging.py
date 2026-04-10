"""
日志配置
"""

import sys
from loguru import logger

from ..core.config import settings


def setup_logging():
    """配置日志"""
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # 文件输出
    log_file = settings.LOGS_DIR / "video_ai.log"
    logger.add(
        log_file,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        compression="zip",
    )
    
    # 错误日志单独文件
    error_log_file = settings.LOGS_DIR / "error.log"
    logger.add(
        error_log_file,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        compression="zip",
    )