"""
设备管理器
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Set
from loguru import logger

from ..core.config import settings
from ..models.schemas import DeviceInfo
from .video_processor import VideoProcessor


class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.devices: Dict[str, VideoProcessor] = {}
        self.device_info: Dict[str, Dict] = {}
    
    async def register_device(self, device_id: str, rtsp_url: str, target_types: Set[str],
                            callback_url: str, username: Optional[str] = None,
                            password: Optional[str] = None) -> DeviceInfo:
        """
        注册设备
        
        Args:
            device_id: 设备ID
            rtsp_url: RTSP流地址
            target_types: 目标类型集合
            callback_url: 回调地址
            username: RTSP用户名
            password: RTSP密码
            
        Returns:
            DeviceInfo: 设备信息
        """
        # 检查设备是否已存在
        if device_id in self.devices:
            raise ValueError(f"设备 {device_id} 已注册")
        
        # 创建视频处理器
        processor = VideoProcessor(device_id, rtsp_url, target_types, callback_url)
        processor.set_auth(username, password)
        
        # 保存设备
        self.devices[device_id] = processor
        self.device_info[device_id] = {
            "register_time": datetime.now(),
            "rtsp_url": rtsp_url,
            "target_types": target_types,
            "callback_url": callback_url,
            "status": "RUNNING",
        }
        
        # 启动视频处理
        await processor.start()
        
        logger.info(f"设备 {device_id} 注册成功")
        
        return DeviceInfo(
            deviceId=device_id,
            status="RUNNING",
            registerTime=datetime.now(),
        )
    
    async def update_device(self, device_id: str, rtsp_url: str, target_types: Set[str],
                          callback_url: str, username: Optional[str] = None,
                          password: Optional[str] = None) -> DeviceInfo:
        """
        更新设备配置
        
        Args:
            device_id: 设备ID
            rtsp_url: RTSP流地址
            target_types: 目标类型集合
            callback_url: 回调地址
            username: RTSP用户名
            password: RTSP密码
            
        Returns:
            DeviceInfo: 设备信息
        """
        # 检查设备是否存在
        if device_id not in self.devices:
            raise ValueError(f"设备 {device_id} 未注册")
        
        # 获取处理器
        processor = self.devices[device_id]
        
        # 更新配置
        await processor.update_config(rtsp_url, target_types, callback_url, username, password)
        
        # 更新设备信息
        self.device_info[device_id].update({
            "rtsp_url": rtsp_url,
            "target_types": target_types,
            "callback_url": callback_url,
            "status": "RUNNING",
        })
        
        logger.info(f"设备 {device_id} 配置更新成功")
        
        return DeviceInfo(
            deviceId=device_id,
            status="RUNNING",
        )
    
    async def delete_device(self, device_id: str) -> bool:
        """
        删除设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否成功
        """
        # 检查设备是否存在
        if device_id not in self.devices:
            raise ValueError(f"设备 {device_id} 未注册")
        
        # 停止处理器
        processor = self.devices[device_id]
        await processor.stop()
        
        # 移除设备
        del self.devices[device_id]
        del self.device_info[device_id]
        
        logger.info(f"设备 {device_id} 删除成功")
        return True
    
    async def pause_device(self, device_id: str) -> DeviceInfo:
        """
        暂停设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            DeviceInfo: 设备信息
        """
        # 检查设备是否存在
        if device_id not in self.devices:
            raise ValueError(f"设备 {device_id} 未注册")
        
        # 暂停处理器
        processor = self.devices[device_id]
        await processor.pause()
        
        # 更新状态
        self.device_info[device_id]["status"] = "PAUSED"
        
        logger.info(f"设备 {device_id} 已暂停")
        
        return DeviceInfo(
            deviceId=device_id,
            status="PAUSED",
        )
    
    async def resume_device(self, device_id: str) -> DeviceInfo:
        """
        恢复设备
        
        Args:
            device_id: 设备ID
            
        Returns:
            DeviceInfo: 设备信息
        """
        # 检查设备是否存在
        if device_id not in self.devices:
            raise ValueError(f"设备 {device_id} 未注册")
        
        # 恢复处理器
        processor = self.devices[device_id]
        await processor.resume()
        
        # 更新状态
        self.device_info[device_id]["status"] = "RUNNING"
        
        logger.info(f"设备 {device_id} 已恢复")
        
        return DeviceInfo(
            deviceId=device_id,
            status="RUNNING",
        )
    
    def get_device_status(self, device_id: str) -> Optional[Dict]:
        """
        获取设备状态
        
        Args:
            device_id: 设备ID
            
        Returns:
            Optional[Dict]: 设备状态信息
        """
        return self.device_info.get(device_id)
    
    def get_all_devices(self) -> Dict[str, Dict]:
        """
        获取所有设备
        
        Returns:
            Dict[str, Dict]: 所有设备信息
        """
        return self.device_info.copy()
    
    async def shutdown(self):
        """关闭所有设备"""
        logger.info("正在关闭所有设备...")
        
        tasks = []
        for device_id, processor in self.devices.items():
            tasks.append(processor.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.devices.clear()
        self.device_info.clear()
        
        logger.info("所有设备已关闭")


# 全局设备管理器实例
device_manager = DeviceManager()