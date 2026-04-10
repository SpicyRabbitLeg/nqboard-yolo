"""
FastAPI 应用工厂
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .config import settings
from ..api.v1.router import api_router
from ..utils.logging import setup_logging


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    # 设置日志
    setup_logging()
    
    # 创建应用
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="基于 YOLO11 和 OpenCV 的实时视频流 AI 分析服务",
        docs_url="/docs" if settings.APP_DEBUG else None,
        redoc_url="/redoc" if settings.APP_DEBUG else None,
    )
    
    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(api_router, prefix="/api")
    
    # 添加健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.VERSION,
        }
    
    # 启动事件
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info(f"{settings.APP_NAME} v{settings.VERSION} 正在启动...")
        logger.info(f"调试模式: {settings.APP_DEBUG}")
        logger.info(f"支持识别类型: {', '.join(settings.SUPPORTED_TYPES)}")
    
    # 关闭事件
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info(f"{settings.APP_NAME} 正在关闭...")
    
    return app


# 创建应用实例
app = create_app()
