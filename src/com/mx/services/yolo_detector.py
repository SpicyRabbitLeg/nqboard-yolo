"""
YOLO 目标检测服务
支持多模型切换：动物检测、火灾检测、安全帽检测
"""

import asyncio
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Set
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger

from ..core.config import settings


class YOLODetector:
    """YOLO 目标检测器，支持多模型切换"""
    
    def __init__(self):
        self.models: Dict[str, Optional[YOLO]] = {
            "animal": None,    # 动物检测模型
            "fire": None,      # 火灾检测模型
            "hardhat": None,   # 安全帽检测模型
        }
        self.class_names: Dict[str, Dict[int, str]] = {
            "animal": {},
            "fire": {},
            "hardhat": {},
        }
        self.model_paths: Dict[str, str] = {
            "animal": settings.YOLO_ANIMAL_MODEL_PATH,    # 动物检测模型
            "fire": settings.YOLO_FIRE_MODEL_PATH,        # 火灾检测模型
            "hardhat": settings.YOLO_HARDHAT_MODEL_PATH,  # 安全帽检测模型
        }
        self._load_models()
        
        # 并发控制
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor = None
    
    def _init_async_components(self):
        """初始化异步组件（在事件循环中调用）"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)
        if self._executor is None:
            import concurrent.futures
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=settings.INFERENCE_THREAD_POOL_SIZE,
                thread_name_prefix="yolo_inference_"
            )
    
    async def async_detect(self, frame: np.ndarray, target_types: List[str]) -> List[Dict[str, Any]]:
        """
        异步检测图像中的目标
        
        Args:
            frame: 图像帧
            target_types: 目标类型列表
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if self._semaphore is None or self._executor is None:
            self._init_async_components()
        
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            detect_fn = partial(self.detect, frame, target_types)
            return await loop.run_in_executor(self._executor, detect_fn)
    
    async def async_draw_detections(self, frame: np.ndarray, 
                                     detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        异步在图像上绘制检测结果
        
        Args:
            frame: 原始图像
            detections: 检测结果
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if self._executor is None:
            self._init_async_components()
        
        loop = asyncio.get_event_loop()
        draw_fn = partial(self.draw_detections, frame, detections)
        return await loop.run_in_executor(self._executor, draw_fn)
    
    def _load_model(self, model_type: str):
        """加载指定类型的 YOLO 模型"""
        try:
            model_path = self.model_paths.get(model_type)
            if not model_path:
                logger.error(f"未找到 {model_type} 类型的模型路径配置")
                return
            
            logger.info(f"加载 {model_type} YOLO 模型: {model_path}")
            model = YOLO(model_path)
            self.models[model_type] = model
            
            # 获取类别名称
            if hasattr(model, 'names'):
                self.class_names[model_type] = model.names
                logger.info(f"{model_type} 模型类别: {model.names}")
            else:
                # 根据模型类型设置默认类别映射
                if model_type == "animal":
                    self.class_names[model_type] = {
                        0: "CAT",
                        1: "DOG",
                    }
                elif model_type == "fire":
                    self.class_names[model_type] = {
                        0: "FIRE",
                    }
                elif model_type == "hardhat":
                    self.class_names[model_type] = {
                        0: "HARDHAT",
                        1: "NO_HARDHAT",  # 未戴安全帽
                    }
                logger.warning(f"{model_type} 模型使用默认类别映射")
            
            logger.info(f"{model_type} YOLO 模型加载成功")
            
        except Exception as e:
            logger.error(f"{model_type} YOLO 模型加载失败: {str(e)}")
            self.models[model_type] = None
    
    def _load_models(self):
        """加载所有 YOLO 模型"""
        for model_type in self.models.keys():
            self._load_model(model_type)
    
    def _get_model_type_for_class(self, class_name: str) -> Optional[str]:
        """根据类别名称获取对应的模型类型"""
        class_name_upper = class_name.upper()
        
        if class_name_upper in ["CAT", "DOG"]:
            # 使用通用 yolo11m.pt 模型，包含 PERSON 等通用类别
            return "animal"
        elif class_name_upper == "PERSON":
            # 人员入侵检测同样复用 animal 模型（yolo11m.pt）
            return "animal"
        elif class_name_upper == "FIRE":
            return "fire"
        elif class_name_upper in ["HARDHAT", "NO_HARDHAT"]:
            return "hardhat"
        
        return None
    
    def detect(self, frame: np.ndarray, target_types: List[str]) -> List[Dict[str, Any]]:
        """
        检测图像中的目标
        
        Args:
            frame: 图像帧
            target_types: 目标类型列表
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if not target_types:
            return []
        
        # 按模型类型分组目标类型
        model_targets: Dict[str, Set[str]] = {
            "animal": set(),
            "fire": set(),
            "hardhat": set(),
        }
        
        for target_type in target_types:
            model_type = self._get_model_type_for_class(target_type)
            if model_type:
                model_targets[model_type].add(target_type.upper())
        
        all_detections = []
        
        # 对每个有目标类型的模型进行检测
        for model_type, targets in model_targets.items():
            if not targets:
                continue
                
            model = self.models.get(model_type)
            if model is None:
                logger.warning(f"{model_type} 模型未加载，跳过检测")
                continue
            
            try:
                # 运行推理
                results = model(
                    frame,
                    conf=settings.YOLO_CONFIDENCE,
                    iou=settings.YOLO_IOU,
                    verbose=False,
                )
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.cpu().numpy()
                        
                        for box in boxes:
                            # 获取类别ID和名称
                            class_id = int(box.cls[0])
                            class_name = self.class_names[model_type].get(class_id, f"class_{class_id}")
                            
                            # 只处理目标类型
                            if class_name.upper() not in targets:
                                continue
                            
                            # 获取置信度
                            confidence = float(box.conf[0])
                            
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0]
                            width = x2 - x1
                            height = y2 - y1
                            
                            detection = {
                                "type": class_name.upper(),
                                "confidence": confidence,
                                "location": {
                                    "x": int(x1),
                                    "y": int(y1),
                                    "width": int(width),
                                    "height": int(height),
                                },
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "model_type": model_type,
                            }
                            
                            all_detections.append(detection)
                            
            except Exception as e:
                logger.error(f"{model_type} 模型检测失败: {str(e)}")
                continue
        
        return all_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 原始图像
            detections: 检测结果
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        result_frame = frame.copy()
        
        # 不同类型使用不同颜色
        color_map = {
            "CAT": (255, 165, 0),    # 橙色 - 猫
            "DOG": (255, 69, 0),     # 红色橙色 - 狗
            "FIRE": (0, 0, 255),     # 红色 - 火灾
            "HARDHAT": (0, 255, 0),  # 绿色 - 安全帽
            "NO_HARDHAT": (255, 0, 0), # 蓝色 - 未戴安全帽
            "PERSON": (255, 255, 0),  # 黄色 - 人员
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            label = f"{detection['type']} {detection['confidence']:.2f}"
            
            # 获取颜色，默认为绿色
            color = color_map.get(detection["type"], (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(
                result_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2,
            )
            
            # 绘制标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_frame,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                color,
                -1,
            )
            
            # 绘制标签文本
            cv2.putText(
                result_frame,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # 黑色
                2,
            )
        
        return result_frame


# 全局检测器实例
detector = YOLODetector()