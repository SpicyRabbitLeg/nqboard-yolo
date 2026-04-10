"""
Redis 客户端
"""

import redis
from loguru import logger

from ..core.config import settings


class RedisClient:
    """Redis 客户端"""
    
    def __init__(self):
        self.client: redis.Redis = None
        self._connect()
    
    def _connect(self):
        """连接到 Redis"""
        try:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            
            # 测试连接
            self.client.ping()
            logger.info(f"Redis 连接成功: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"Redis 连接失败: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """检查是否连接"""
        if self.client is None:
            return False
        
        try:
            return self.client.ping()
        except:
            return False
    
    def get(self, key: str) -> str:
        """获取值"""
        if not self.is_connected():
            return None
        
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get 失败: {str(e)}")
            return None
    
    def set(self, key: str, value: str, ex: int = None) -> bool:
        """设置值"""
        if not self.is_connected():
            return False
        
        try:
            return self.client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Redis set 失败: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除键"""
        if not self.is_connected():
            return False
        
        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis delete 失败: {str(e)}")
            return False
    
    def hset(self, key: str, field: str, value: str) -> bool:
        """设置哈希字段"""
        if not self.is_connected():
            return False
        
        try:
            return self.client.hset(key, field, value)
        except Exception as e:
            logger.error(f"Redis hset 失败: {str(e)}")
            return False
    
    def hget(self, key: str, field: str) -> str:
        """获取哈希字段"""
        if not self.is_connected():
            return None
        
        try:
            return self.client.hget(key, field)
        except Exception as e:
            logger.error(f"Redis hget 失败: {str(e)}")
            return None
    
    def hgetall(self, key: str) -> dict:
        """获取所有哈希字段"""
        if not self.is_connected():
            return {}
        
        try:
            return self.client.hgetall(key)
        except Exception as e:
            logger.error(f"Redis hgetall 失败: {str(e)}")
            return {}


# 全局 Redis 客户端实例
redis_client = RedisClient()