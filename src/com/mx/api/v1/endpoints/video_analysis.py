"""
视频分析 API 端点
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from ....models.schemas import (
    DeviceRegister,
    DeviceUpdate,
    DeviceDelete,
    DeviceStatusControl,
    DeviceInfo,
    ResponseBase,
)
from ....services.device_manager import device_manager

router = APIRouter()


@router.post("/register", response_model=ResponseBase)
async def register_device(request: DeviceRegister):
    """
    摄像头 AI 服务注册接口
    
    第三方注册摄像头，配置 RTSP 流、识别类型、回调地址，系统启动实时流分析
    """
    try:
        # 注册设备
        device_info = await device_manager.register_device(
            device_id=request.device_id,
            rtsp_url=request.rtsp_url,
            target_types=set(request.types),
            callback_url=request.callback_url,
            username=request.rtsp_username,
            password=request.rtsp_password,
        )
        
        return ResponseBase(
            code=200,
            msg="服务注册成功，已启动视频分析",
            data=device_info.dict(by_alias=True),
        )
        
    except ValueError as e:
        logger.warning(f"设备注册参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"设备注册失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务注册失败，请稍后重试",
        )


@router.put("/update", response_model=ResponseBase)
async def update_device(request: DeviceUpdate):
    """
    摄像头 AI 服务更新接口
    
    更新已注册设备的 RTSP 信息、识别类型、回调地址、账号密码
    """
    try:
        # 更新设备
        device_info = await device_manager.update_device(
            device_id=request.device_id,
            rtsp_url=request.rtsp_url,
            target_types=set(request.types),
            callback_url=request.callback_url,
            username=request.rtsp_username,
            password=request.rtsp_password,
        )
        
        return ResponseBase(
            code=200,
            msg="服务更新成功，已重新加载配置",
            data=device_info.dict(by_alias=True),
        )
        
    except ValueError as e:
        logger.warning(f"设备更新参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"设备更新失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务更新失败，请稍后重试",
        )


@router.delete("/delete", response_model=ResponseBase)
async def delete_device(request: DeviceDelete):
    """
    摄像头 AI 服务删除接口
    
    注销设备，停止流拉取、AI 分析、所有回调
    """
    try:
        # 删除设备
        success = await device_manager.delete_device(request.device_id)
        
        if success:
            return ResponseBase(
                code=200,
                msg="服务删除成功，已停止所有任务",
                data=None,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务删除失败",
            )
        
    except ValueError as e:
        logger.warning(f"设备删除参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"设备删除失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务删除失败，请稍后重试",
        )


@router.put("/status", response_model=ResponseBase)
async def control_device_status(request: DeviceStatusControl):
    """
    服务状态控制（暂停 / 恢复）
    
    暂停 / 恢复指定设备的 AI 分析与回调
    """
    try:
        if request.operate == "PAUSE":
            # 暂停设备
            device_info = await device_manager.pause_device(request.device_id)
            msg = "设备已暂停分析"
        else:
            # 恢复设备
            device_info = await device_manager.resume_device(request.device_id)
            msg = "设备已恢复分析"
        
        return ResponseBase(
            code=200,
            msg=msg,
            data=device_info.dict(by_alias=True),
        )
        
    except ValueError as e:
        logger.warning(f"设备状态控制参数错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"设备状态控制失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="状态控制失败，请稍后重试",
        )


@router.get("/list", response_model=ResponseBase)
async def list_devices():
    """
    获取所有设备列表
    """
    try:
        devices = device_manager.get_all_devices()
        return ResponseBase(
            code=200,
            msg="获取设备列表成功",
            data=devices,
        )
    except Exception as e:
        logger.error(f"获取设备列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取设备列表失败",
        )


@router.get("/status/{device_id}", response_model=ResponseBase)
async def get_device_status(device_id: str):
    """
    获取设备状态
    """
    try:
        status_info = device_manager.get_device_status(device_id)
        if status_info:
            return ResponseBase(
                code=200,
                msg="获取设备状态成功",
                data=status_info,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"设备 {device_id} 不存在",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取设备状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取设备状态失败",
        )