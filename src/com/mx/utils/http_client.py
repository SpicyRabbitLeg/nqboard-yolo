"""
HTTP 客户端工具
"""

import asyncio
import base64
import json
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp
from loguru import logger
import cv2
from ..core.config import settings


class JSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，支持 datetime 序列化"""

    def default(self, obj):
        if isinstance(obj, datetime):
            # 将 datetime 转换为 YYYY-MM-DD HH:mm:ss 格式字符串
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        # 让基类处理其他类型
        return super().default(obj)


class AsyncHTTPClient:
    """异步 HTTP 客户端"""
    
    def __init__(self, timeout: int = settings.CALLBACK_TIMEOUT):
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """进入上下文管理器"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.session:
            await self.session.close()
    
    async def post_json(self, url: str, data: Dict[str, Any], headers: Optional[Dict] = None) -> bool:
        """
        发送 POST JSON 请求
        
        Args:
            url: 请求URL
            data: JSON数据
            headers: 请求头
            
        Returns:
            bool: 是否成功
        """
        if not self.session:
            raise RuntimeError("HTTP客户端未初始化")
        
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Video-AI/{settings.VERSION}",
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            # 使用自定义 JSON 编码器序列化数据
            json_data = json.dumps(data, cls=JSONEncoder, ensure_ascii=False)
            async with self.session.post(url, data=json_data, headers=default_headers) as response:
                if response.status in (200, 201, 204):
                    logger.debug(f"回调成功: {url}, 状态码: {response.status}")
                    return True
                else:
                    logger.warning(f"回调失败: {url}, 状态码: {response.status}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"回调超时: {url}, 超时时间: {self.timeout}s")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"回调网络错误: {url}, 错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"回调未知错误: {url}, 错误: {str(e)}")
            return False


def encode_frame_to_base64(frame, width=640, height=480, quality=80) -> str:
    """
    🔥 已优化：固定图片大小 + 可调节质量 → 转 Base64
    """
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # 质量参数可调，建议 70-95（越高越清晰但文件越大）
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    if not success:
        raise ValueError("图像编码失败")
    
    return base64.b64encode(buffer).decode('utf-8')


async def send_callback(url: str, data: Dict[str, Any]) -> bool:
    """
    发送回调请求
    
    Args:
        url: 回调URL
        data: 回调数据
        
    Returns:
        bool: 是否成功
    """
    async with AsyncHTTPClient() as client:
        return await client.post_json(url, data)