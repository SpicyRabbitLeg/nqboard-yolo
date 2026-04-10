"""
API 路由配置
"""

from fastapi import APIRouter

from .endpoints import video_analysis

api_router = APIRouter()

# 注册视频分析路由
api_router.include_router(
    video_analysis.router,
    prefix="/video/analysis",
    tags=["video-analysis"],
)