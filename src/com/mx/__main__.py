#!/usr/bin/env python3
"""
Video AI 服务主入口
"""

import uvicorn
from loguru import logger

from .core.config import settings
from .core.app import create_app

app = create_app()


def main():
    """启动 FastAPI 应用"""
    logger.info(f"启动 Video AI 服务，版本: {settings.VERSION}")
    logger.info(f"服务地址: http://{settings.APP_HOST}:{settings.APP_PORT}")
    logger.info(f"API 文档: http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
    
    # 使用导入字符串形式，这样 reload 才能正常工作
    uvicorn.run(
        "src.com.mx.core.app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_level="info" if settings.APP_DEBUG else "warning",
        reload=settings.APP_DEBUG,
    )


if __name__ == "__main__":
    main()