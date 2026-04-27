"""
视频流处理服务（带帧读取失败兜底方案，支持 GPU 加速）
"""

import asyncio
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Set, List

import cv2
import numpy as np
from loguru import logger

from ..core.config import settings
from ..models.schemas import RecognizeCallback, StatusCallback
from ..utils.http_client import encode_frame_to_base64, send_callback
from .yolo_detector import detector


class VideoProcessor:
    """视频处理器（带兜底重连机制，支持 GPU 加速）"""

    _gpu_available: Optional[bool] = None
    _gstreamer_available: Optional[bool] = None

    def __init__(
        self,
        device_id: str,
        rtsp_url: str,
        target_types: Set[str],
        callback_url: str,
    ):
        self.device_id = device_id
        self.rtsp_url = rtsp_url
        self.target_types = target_types
        self.callback_url = callback_url

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.last_status_time = datetime.now()

        self.rtsp_username: Optional[str] = None
        self.rtsp_password: Optional[str] = None

        self.use_gpu: bool = settings.USE_GPU_DECODE
        self._capture_type: str = "unknown"

        # 兜底重连机制参数
        self.consecutive_read_failures = 0
        self.max_consecutive_failures = settings.MAX_CONSECUTIVE_READ_FAILURES
        self.reconnect_delay = settings.INITIAL_RECONNECT_DELAY
        self.max_reconnect_delay = settings.MAX_RECONNECT_DELAY
        self.is_reconnecting = False

        # 人员入侵检测状态
        self._person_present_since: Optional[datetime] = None
        self._person_last_seen: Optional[datetime] = None
        self._person_intrusion_reported = False

    @property
    def capture_info(self) -> str:
        """获取当前捕获方式信息"""
        return f"{self._capture_type} (GPU: {'启用' if self.use_gpu else '禁用'})"

    @property
    def _person_detecting(self) -> bool:
        """是否检测人员入侵"""
        return "PERSON" in self.target_types

    # ==================== 静态检测方法 ====================

    @staticmethod
    def check_gpu_available() -> bool:
        """检测 NVIDIA GPU 是否可用"""
        if VideoProcessor._gpu_available is not None:
            return VideoProcessor._gpu_available

        VideoProcessor._gpu_available = False

        try:
            has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False

            if not has_cuda:
                try:
                    import torch
                    VideoProcessor._gpu_available = torch.cuda.is_available()
                except ImportError:
                    pass

            if VideoProcessor._gpu_available:
                logger.info("检测到 NVIDIA GPU，GPU 加速可用")
            else:
                logger.info("未检测到 NVIDIA GPU，将使用 CPU 解码")

        except Exception as e:
            logger.warning(f"GPU 检测失败: {e}")
            VideoProcessor._gpu_available = False

        return VideoProcessor._gpu_available

    @staticmethod
    def check_gstreamer_available() -> bool:
        """检测 GStreamer 是否可用"""
        if VideoProcessor._gstreamer_available is not None:
            return VideoProcessor._gstreamer_available

        VideoProcessor._gstreamer_available = cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER)

        if VideoProcessor._gstreamer_available:
            logger.info("GStreamer 后端可用")
        else:
            logger.warning("GStreamer 后端不可用，将使用标准 OpenCV 解码")

        return VideoProcessor._gstreamer_available

    @staticmethod
    def _create_gstreamer_pipeline(rtsp_url: str, latency: int = 100) -> str:
        """创建 GStreamer 管道（根据操作系统选择合适的硬件解码器）"""
        system = platform.system()

        if system == "Linux":
            pipeline = (
                f"rtspsrc location={rtsp_url} latency={latency} ! "
                f"rtph264depay ! h264parse ! video/x-h264,stream-format=byte-stream ! "
                f"nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink"
            )
        elif system == "Windows":
            pipeline = (
                f"rtspsrc location={rtsp_url} latency={latency} ! "
                f"rtph264depay ! h264parse ! "
                f"msdkh264dec ! videoconvert ! video/x-raw,format=BGR ! appsink"
            )
        else:
            pipeline = (
                f"rtspsrc location={rtsp_url} latency={latency} ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! "
                f"videoconvert ! video/x-raw,format=BGR ! appsink"
            )

        return pipeline

    # ==================== 连接管理 ====================

    def _build_rtsp_url(self, raw_url: str) -> str:
        """构建包含认证信息的 RTSP URL"""
        if not self.rtsp_username or not self.rtsp_password:
            return raw_url

        if raw_url.startswith("rtsp://"):
            protocol = "rtsp://"
            address = raw_url[7:]
        elif raw_url.startswith("rtsps://"):
            protocol = "rtsps://"
            address = raw_url[8:]
        else:
            raise ValueError(f"不支持的RTSP协议: {raw_url}")

        return f"{protocol}{self.rtsp_username}:{self.rtsp_password}@{address}"

    def _create_capture(self, rtsp_url: str) -> cv2.VideoCapture:
        """创建视频捕获对象，优先使用 GPU 加速"""
        gpu_available = self.check_gpu_available()
        gstreamer_available = self.check_gstreamer_available()

        if not self.use_gpu or not gpu_available or not gstreamer_available:
            self._capture_type = "CPU (OpenCV)"
            if self.use_gpu and not gpu_available:
                logger.warning(f"设备 {self.device_id} GPU 不可用，回退到 CPU 解码")
            elif self.use_gpu and not gstreamer_available:
                logger.warning(f"设备 {self.device_id} GStreamer 不可用，回退到 CPU 解码")
            return cv2.VideoCapture(rtsp_url)

        try:
            pipeline = self._create_gstreamer_pipeline(rtsp_url)
            self._capture_type = "GPU (GStreamer)"
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                raise RuntimeError("GStreamer 管道创建失败")

            logger.info(f"设备 {self.device_id} 使用 GPU 加速解码 (GStreamer)")
            return cap

        except Exception as e:
            logger.warning(f"设备 {self.device_id} GPU 加速失败: {e}，回退到 CPU 解码")
            self._capture_type = "CPU (OpenCV)"
            return cv2.VideoCapture(rtsp_url)

    def set_auth(self, username: Optional[str], password: Optional[str]):
        """设置 RTSP 认证信息"""
        self.rtsp_username = username
        self.rtsp_password = password

    async def connect(self) -> bool:
        """连接到 RTSP 流"""
        try:
            rtsp_url = self._build_rtsp_url(self.rtsp_url)
            logger.info(f"设备 {self.device_id} 正在连接RTSP: {rtsp_url}")

            if self.cap:
                self.cap.release()

            self.cap = self._create_capture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.RTSP_TIMEOUT * 1000)

            if not self.cap.isOpened():
                logger.error(f"设备 {self.device_id} RTSP连接失败")
                self.cap = None
                return False

            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"设备 {self.device_id} 连接成功但无法读取首帧")
                self.cap.release()
                self.cap = None
                return False

            self.consecutive_read_failures = 0
            self.reconnect_delay = settings.INITIAL_RECONNECT_DELAY
            logger.info(f"设备 {self.device_id} RTSP连接成功")
            return True

        except Exception as e:
            logger.error(f"设备 {self.device_id} 连接异常: {str(e)}")
            self.cap = None
            return False

    def _reset_person_state(self):
        """重置人员入侵检测状态"""
        self._person_present_since = None
        self._person_last_seen = None
        self._person_intrusion_reported = False

    def _is_stream_connected(self) -> bool:
        """检查视频流是否处于连接状态"""
        return self.cap is not None and self.cap.isOpened()

    # ==================== 生命周期管理 ====================

    async def start(self):
        """开始处理视频流"""
        if self.running:
            logger.warning(f"设备 {self.device_id} 已经在运行")
            return

        self.running = True
        self._reset_person_state()

        if not await self.connect():
            logger.error(f"设备 {self.device_id} 启动失败，将在后台尝试重连")
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

    async def update_config(
        self,
        rtsp_url: str,
        target_types: Set[str],
        callback_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """更新配置"""
        await self.stop()
        self.rtsp_url = rtsp_url
        self.target_types = target_types
        self.callback_url = callback_url
        self.rtsp_username = username
        self.rtsp_password = password
        await self.start()

    # ==================== 重连机制 ====================

    async def _reconnect_loop(self):
        """自动重连循环（帧读取失败后触发）"""
        if self.is_reconnecting:
            return
        self.is_reconnecting = True

        logger.warning(f"设备 {self.device_id} 启动重连机制，当前等待: {self.reconnect_delay}s")

        while self.running:
            await asyncio.sleep(self.reconnect_delay)

            if await self.connect():
                logger.success(f"设备 {self.device_id} 重连成功！恢复正常处理")
                self._reset_person_state()
                asyncio.create_task(self._process_loop())
                break

            self.reconnect_delay = min(
                self.reconnect_delay * 2, self.max_reconnect_delay
            )
            logger.warning(f"设备 {self.device_id} 重连失败，下次重试: {self.reconnect_delay}s")
            await self._send_status_callback()

        self.is_reconnecting = False

    # ==================== 视频处理循环 ====================

    async def _process_loop(self):
        """视频处理循环（带帧失败兜底）"""
        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(1)
                    continue

                if not self._is_stream_connected():
                    logger.error(f"设备 {self.device_id} 连接已断开，启动重连")
                    asyncio.create_task(self._reconnect_loop())
                    return

                ret, frame = self.cap.read()

                if not ret:
                    await self._handle_frame_read_failure()
                    continue

                await self._handle_frame_success(frame)

            except Exception as e:
                logger.error(f"设备 {self.device_id} 处理异常: {str(e)}")
                await asyncio.sleep(1)

    async def _handle_frame_read_failure(self):
        """处理帧读取失败"""
        self.consecutive_read_failures += 1
        logger.warning(
            f"设备 {self.device_id} 读帧失败 [{self.consecutive_read_failures}/{self.max_consecutive_failures}]"
        )

        if self.consecutive_read_failures >= self.max_consecutive_failures:
            logger.error(f"设备 {self.device_id} 连续读帧失败超限，强制重启流")
            if self.cap:
                self.cap.release()
            self.cap = None
            asyncio.create_task(self._reconnect_loop())

        await asyncio.sleep(0.5)

    async def _handle_frame_success(self, frame: np.ndarray):
        """处理帧读取成功"""
        self.consecutive_read_failures = 0
        self.frame_count += 1

        if self.frame_count % settings.FRAME_SKIP != 0:
            return

        detections = detector.detect(frame, list(self.target_types))
        current_time = datetime.now()

        if self._person_detecting:
            await self._process_person_detection(frame, detections, current_time)

        await self._process_other_detections(frame, detections, current_time)
        await self._maybe_send_status_callback(current_time)

        await asyncio.sleep(0.01)

    # ==================== 人员入侵检测 ====================

    async def _process_person_detection(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        current_time: datetime,
    ):
        """处理人员入侵检测逻辑"""
        person_detections = [d for d in detections if d.get("type") == "PERSON"]
        has_person = bool(person_detections)

        if has_person:
            if self._person_present_since is None:
                self._person_present_since = current_time
            self._person_last_seen = current_time

            stay_seconds = (current_time - self._person_present_since).total_seconds()

            if (
                stay_seconds >= settings.PERSON_STAY_THRESHOLD
                and not self._person_intrusion_reported
            ):
                best_person = max(person_detections, key=lambda x: x.get("confidence", 0.0))
                await self._send_recognize_callback(frame, best_person)
                self._person_intrusion_reported = True
        else:
            await self._check_person_absence(current_time)

    async def _check_person_absence(self, current_time: datetime):
        """检查人员是否离开并重置状态"""
        if self._person_last_seen is None:
            return

        absence_seconds = (current_time - self._person_last_seen).total_seconds()
        if absence_seconds >= settings.PERSON_ABSENCE_RESET:
            self._reset_person_state()

    async def _process_other_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        current_time: datetime,
    ):
        """处理非人员类型的检测结果"""
        del current_time  # 未使用，但保留参数签名一致性

        for detection in detections:
            if detection.get("type") == "PERSON":
                continue
            await self._send_recognize_callback(frame, detection)

    async def _maybe_send_status_callback(self, current_time: datetime):
        """根据时间间隔发送状态回调"""
        time_diff = (current_time - self.last_status_time).total_seconds()
        if time_diff >= settings.STATUS_INTERVAL:
            await self._send_status_callback()
            self.last_status_time = current_time

    # ==================== 回调发送 ====================

    async def _send_recognize_callback(
        self, frame: np.ndarray, detection: Dict[str, Any]
    ):
        """发送识别结果回调"""
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

            success = await send_callback(
                self.callback_url, callback_data.dict(by_alias=True)
            )
            if success:
                logger.debug(f"设备 {self.device_id} 识别回调发送成功")
        except Exception as e:
            logger.error(f"设备 {self.device_id} 识别回调异常: {str(e)}")

    async def _send_status_callback(self):
        """发送状态回调"""
        try:
            is_stream_open = self._is_stream_connected()
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

            await send_callback(self.callback_url, callback_data.dict(by_alias=True))
        except Exception as e:
            logger.error(f"设备 {self.device_id} 状态回调异常: {str(e)}")
