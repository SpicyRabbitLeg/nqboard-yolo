"""
视频流处理服务（带帧读取失败兜底方案）
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Set
import cv2
import numpy as np
from loguru import logger

from ..core.config import settings
from ..models.schemas import RecognizeCallback, StatusCallback
from ..utils.http_client import encode_frame_to_base64, send_callback
from .yolo_detector import detector


class VideoProcessor:
    """视频处理器（带兜底重连机制）"""

    def __init__(self, device_id: str, rtsp_url: str, target_types: Set[str], callback_url: str):
        """
        初始化视频处理器
        """
        self.device_id = device_id
        self.rtsp_url = rtsp_url
        self.target_types = target_types
        self.callback_url = callback_url

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.last_status_time = datetime.now()

        # RTSP认证
        self.rtsp_username: Optional[str] = None
        self.rtsp_password: Optional[str] = None

        # ===================== 兜底方案新增参数 =====================
        self.consecutive_read_failures = 0  # 连续读帧失败计数
        self.max_consecutive_failures = settings.MAX_CONSECUTIVE_READ_FAILURES  # 最大连续失败次数（超过触发重连）
        self.reconnect_delay = settings.INITIAL_RECONNECT_DELAY                  # 初始重连等待秒数
        self.max_reconnect_delay = settings.MAX_RECONNECT_DELAY                   # 最大重连等待秒数
        self.is_reconnecting = False         # 重连状态锁，避免并发重连

        # 人员非法入侵检测状态（基于停留时长）
        self.person_present_since: Optional[datetime] = None  # 第一次检测到 PERSON 的时间
        self.person_last_seen: Optional[datetime] = None      # 最近一次检测到 PERSON 的时间
        self.person_intrusion_reported: bool = False          # 当前这一段连续停留是否已经上报
        self.person_stay_threshold: float = settings.PERSON_STAY_THRESHOLD  # 停留阈值（秒）
        self.person_absence_reset: float = settings.PERSON_ABSENCE_RESET    # 视为离开并重置状态的时间（秒）

    def set_auth(self, username: Optional[str], password: Optional[str]):
        """设置RTSP认证信息"""
        self.rtsp_username = username
        self.rtsp_password = password

    async def connect(self) -> bool:
        """连接到RTSP流"""
        try:
            rtsp_url = self.rtsp_url
            if self.rtsp_username and self.rtsp_password:
                if rtsp_url.startswith("rtsp://"):
                    protocol = "rtsp://"
                    address = rtsp_url[7:]
                elif rtsp_url.startswith("rtsps://"):
                    protocol = "rtsps://"
                    address = rtsp_url[8:]
                else:
                    raise ValueError(f"不支持的RTSP协议: {rtsp_url}")
                rtsp_url = f"{protocol}{self.rtsp_username}:{self.rtsp_password}@{address}"

            logger.info(f"设备 {self.device_id} 正在连接RTSP: {rtsp_url}")

            # 释放旧连接
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.RTSP_TIMEOUT * 1000)

            if not self.cap.isOpened():
                logger.error(f"设备 {self.device_id} RTSP连接失败")
                self.cap = None  # 连接失败置空
                return False

            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"设备 {self.device_id} 连接成功但无法读取首帧")
                self.cap.release()
                self.cap = None
                return False

            # 连接成功，重置失败计数
            self.consecutive_read_failures = 0
            self.reconnect_delay = settings.INITIAL_RECONNECT_DELAY
            logger.info(f"设备 {self.device_id} RTSP连接成功")
            return True

        except Exception as e:
            logger.error(f"设备 {self.device_id} 连接异常: {str(e)}")
            self.cap = None
            return False

    async def start(self):
        """开始处理视频流"""
        if self.running:
            logger.warning(f"设备 {self.device_id} 已经在运行")
            return

        self.running = True  # 提前标记运行状态
        if not await self.connect():
            logger.error(f"设备 {self.device_id} 启动失败，将在后台尝试重连")
            # 启动失败不退出，启动重连任务
            asyncio.create_task(self._reconnect_loop())
            return

        asyncio.create_task(self._process_loop())
        logger.info(f"设备 {self.device_id} 视频处理已启动")

    async def stop(self):
        """停止处理视频流"""
        self.running = False
        self.is_reconnecting = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"设备 {self.device_id} 视频处理已停止")

    async def pause(self):
        """暂停处理"""
        self.paused = True
        logger.info(f"设备 {self.device_id} 视频处理已暂停")

    async def resume(self):
        """恢复处理"""
        self.paused = False
        logger.info(f"设备 {self.device_id} 视频处理已恢复")

    async def update_config(self, rtsp_url: str, target_types: Set[str], callback_url: str,
                          username: Optional[str] = None, password: Optional[str] = None):
        """更新配置"""
        await self.stop()
        self.rtsp_url = rtsp_url
        self.target_types = target_types
        self.callback_url = callback_url
        self.rtsp_username = username
        self.rtsp_password = password
        await self.start()

    async def _reconnect_loop(self):
        """自动重连循环（帧读取失败后触发）"""
        # 重入锁：防止并发重连
        if self.is_reconnecting:
            return
        self.is_reconnecting = True

        logger.warning(f"设备 {self.device_id} 启动重连机制，当前等待: {self.reconnect_delay}s")

        # 修复核心：先判断cap不为None，再判断isOpened()
        while self.running:
            await asyncio.sleep(self.reconnect_delay)

            # 尝试重连
            if await self.connect():
                logger.success(f"设备 {self.device_id} 重连成功！恢复正常处理")
                asyncio.create_task(self._process_loop())
                break

            # 重连失败，指数退避（3s→6s→12s→30s封顶）
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            logger.warning(f"设备 {self.device_id} 重连失败，下次重试: {self.reconnect_delay}s")

            # 发送离线状态回调
            await self._send_status_callback()

        self.is_reconnecting = False

    async def _process_loop(self):
        """视频处理循环（带帧失败兜底）"""
        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(1)
                    continue

                # 连接已关闭 → 触发重连
                if self.cap is None or not self.cap.isOpened():
                    logger.error(f"设备 {self.device_id} 连接已断开，启动重连")
                    asyncio.create_task(self._reconnect_loop())
                    await asyncio.sleep(1)
                    return

                # 读取帧
                ret, frame = self.cap.read()

                if not ret:
                    self.consecutive_read_failures += 1
                    logger.warning(f"设备 {self.device_id} 读帧失败 [{self.consecutive_read_failures}/{self.max_consecutive_failures}]")

                    # 连续失败达到阈值 → 强制重连
                    if self.consecutive_read_failures >= self.max_consecutive_failures:
                        logger.error(f"设备 {self.device_id} 连续读帧失败超限，强制重启流")
                        # 清理资源
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                        # 触发重连
                        asyncio.create_task(self._reconnect_loop())
                        return

                    await asyncio.sleep(0.5)
                    continue

                # 读帧成功 → 重置失败计数
                self.consecutive_read_failures = 0
                self.frame_count += 1

                # 跳帧处理
                if self.frame_count % settings.FRAME_SKIP != 0:
                    continue

                # 目标检测
                detections = detector.detect(frame, list(self.target_types))

                current_time = datetime.now()

                # 人员非法入侵检测（基于停留时长 > person_stay_threshold）
                if "PERSON" in self.target_types:
                    person_detections = [d for d in detections if d.get("type") == "PERSON"]
                    has_person = len(person_detections) > 0

                    if has_person:
                        if self.person_present_since is None:
                            self.person_present_since = current_time
                        self.person_last_seen = current_time

                        stay_seconds = (current_time - self.person_present_since).total_seconds()

                        # 达到停留阈值且当前这段连续停留尚未上报 → 上报一次
                        if stay_seconds >= self.person_stay_threshold and not self.person_intrusion_reported:
                            # 选取置信度最高的 PERSON 目标作为上报对象
                            best_person = max(person_detections, key=lambda x: x.get("confidence", 0.0))
                            await self._send_recognize_callback(frame, best_person)
                            self.person_intrusion_reported = True
                    else:
                        # 一段时间内未检测到 PERSON，则认为人员已经离开，重置状态
                        if self.person_last_seen is not None:
                            absence_seconds = (current_time - self.person_last_seen).total_seconds()
                            if absence_seconds >= self.person_absence_reset:
                                self.person_present_since = None
                                self.person_last_seen = None
                                self.person_intrusion_reported = False

                # 其他类型的实时回调
                for detection in detections:
                    if detection.get("type") == "PERSON" and "PERSON" in self.target_types:
                        continue
                    await self._send_recognize_callback(frame, detection)

                # 状态回调
                time_diff = (current_time - self.last_status_time).total_seconds()
                if time_diff >= settings.STATUS_INTERVAL:
                    await self._send_status_callback()
                    self.last_status_time = current_time

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"设备 {self.device_id} 处理异常: {str(e)}")
                await asyncio.sleep(1)

    async def _send_recognize_callback(self, frame: np.ndarray, detection: Dict[str, Any]):
        try:
            detection_list = [detection]
            frame_with_bbox = detector.draw_detections(frame, detection_list)
            frame_base64 = encode_frame_to_base64(frame_with_bbox)

            callback_data = RecognizeCallback(
                deviceId=self.device_id,
                rtspUrl=self.rtsp_url,
                recognizeType=detection["type"],
                confidence=detection["confidence"],
                frameTime=datetime.now(),
                frameBase64=frame_base64,
                targetLocation=detection["location"],
            )

            success = await send_callback(self.callback_url, callback_data.dict(by_alias=True))
            if success:
                logger.debug(f"设备 {self.device_id} 识别回调发送成功")
        except Exception as e:
            logger.error(f"设备 {self.device_id} 识别回调异常: {str(e)}")

    async def _send_status_callback(self):
        try:
            # 修复：判断cap是否为None，避免调用isOpened()报错
            is_stream_open = self.cap is not None and self.cap.isOpened()
            status = "ONLINE" if (self.running and not self.paused and is_stream_open) else "OFFLINE"
            analysis = "RUNNING" if not self.paused else "PAUSED"

            callback_data = StatusCallback(
                deviceId=self.device_id,
                rtspUrl=self.rtsp_url,
                deviceStatus=status,
                analysisStatus=analysis,
                currentTypes=list(self.target_types),
                timestamp=datetime.now(),
            )

            success = await send_callback(self.callback_url, callback_data.dict(by_alias=True))
        except Exception as e:
            logger.error(f"设备 {self.device_id} 状态回调异常: {str(e)}")