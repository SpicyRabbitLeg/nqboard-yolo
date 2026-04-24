"""
Pydantic 数据模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from ..core.config import settings


class ResponseBase(BaseModel):
    """响应基类"""
    code: int = Field(200, description="响应码：200成功 500失败 400参数错误")
    msg: str = Field("操作成功", description="响应描述")
    data: Optional[Any] = Field(None, description="业务数据")


class DeviceRegister(BaseModel):
    """设备注册请求"""
    rtsp_url: str = Field(..., alias="rtspUrl", description="RTSP流地址")
    rtsp_username: Optional[str] = Field(None, alias="rtspUsername", description="RTSP用户名")
    rtsp_password: Optional[str] = Field(None, alias="rtspPassword", description="RTSP密码")
    types: List[str] = Field(..., description="AI识别类型列表")
    device_id: str = Field(..., alias="deviceId", description="唯一设备ID")
    callback_url: str = Field(..., alias="callbackUrl", description="回调地址")
    
    @validator("types")
    def validate_types(cls, v):
        """验证识别类型"""
        for t in v:
            if t.upper() not in settings.SUPPORTED_TYPES:
                raise ValueError(f"不支持的识别类型: {t}")
        return [t.upper() for t in v]
    
    @validator("rtsp_url")
    def validate_rtsp_url(cls, v):
        """验证 RTSP URL"""
        if not v.startswith(("rtsp://", "rtsps://")):
            raise ValueError("RTSP URL 必须以 rtsp:// 或 rtsps:// 开头")
        return v
    
    class Config:
        allow_population_by_field_name = True


class DeviceUpdate(DeviceRegister):
    """设备更新请求"""
    pass


class DeviceDelete(BaseModel):
    """设备删除请求"""
    device_id: str = Field(..., alias="deviceId", description="设备唯一ID")
    
    class Config:
        allow_population_by_field_name = True


class DeviceStatusControl(BaseModel):
    """设备状态控制请求"""
    device_id: str = Field(..., alias="deviceId", description="设备唯一ID")
    operate: str = Field(..., description="操作类型：PAUSE-暂停 RESUME-恢复")
    
    @validator("operate")
    def validate_operate(cls, v):
        """验证操作类型"""
        if v.upper() not in ["PAUSE", "RESUME"]:
            raise ValueError("操作类型必须是 PAUSE 或 RESUME")
        return v.upper()
    
    class Config:
        allow_population_by_field_name = True


class DeviceInfo(BaseModel):
    """设备信息响应"""
    device_id: str = Field(..., alias="deviceId", description="设备ID")
    status: str = Field(..., description="状态：RUNNING-运行中 PAUSED-已暂停")
    register_time: Optional[datetime] = Field(None, alias="registerTime", description="注册时间")
    
    class Config:
        allow_population_by_field_name = True


class RecognizeCallback(BaseModel):
    """AI识别结果回调"""
    callback_type: str = Field("RECOGNIZE", alias="callbackType", description="回调类型")
    device_id: str = Field(..., alias="deviceId", description="设备ID")
    rtsp_url: str = Field(..., alias="rtspUrl", description="RTSP流地址")
    recognize_type: str = Field(..., alias="recognizeType", description="识别到的具体类型")
    confidence: float = Field(..., description="识别置信度 0-1")
    frame_time: datetime = Field(..., alias="frameTime", description="帧时间")
    frame_base64: str = Field(..., alias="frameBase64", description="当前帧图片Base64编码")
    target_location: Dict[str, int] = Field(..., alias="targetLocation", description="目标坐标")
    
    class Config:
        allow_population_by_field_name = True


class StatusCallback(BaseModel):
    """设备状态回调"""
    callback_type: str = Field("STATUS", alias="callbackType", description="回调类型")
    device_id: str = Field(..., alias="deviceId", description="设备ID")
    rtsp_url: str = Field(..., alias="rtspUrl", description="RTSP流地址")
    device_status: str = Field(..., alias="deviceStatus", description="设备状态")
    analysis_status: str = Field(..., alias="analysisStatus", description="分析状态")
    current_types: List[str] = Field(..., alias="currentTypes", description="当前生效识别类型")
    timestamp: datetime = Field(..., description="时间戳")
    
    class Config:
        allow_population_by_field_name = True