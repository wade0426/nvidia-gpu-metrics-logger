#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - GPU 監控核心

使用 NVIDIA ML Python 庫監控 GPU 狀態並收集指標資料
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import pynvml
except ImportError:
    raise ImportError("請安裝 nvidia-ml-py3: pip install nvidia-ml-py3")

try:
    # 嘗試相對導入（作為模組時）
    from .utils import safe_float, safe_int, format_timestamp
except ImportError:
    # 絕對導入（作為腳本直接執行時）
    from utils import safe_float, safe_int, format_timestamp


@dataclass
class GPUMetrics:
    """GPU 指標資料結構"""
    timestamp: str
    gpu_id: int
    gpu_name: str
    utilization_gpu: float
    utilization_memory: float
    memory_total: int
    memory_used: int
    memory_free: int
    temperature: int
    power_draw: float
    power_limit: float
    fan_speed: int


class GPUMonitor:
    """
    NVIDIA GPU 監控器
    
    負責初始化 NVIDIA ML 庫、偵測 GPU 並收集指標資料
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 GPU 監控器
        
        Args:
            config: 設定字典
        """
        self.config = config
        self.logger = logging.getLogger("gpu_monitor.GPUMonitor")
        self.gpu_count = 0
        self.gpu_handles = []
        self.is_initialized = False
        
        # 初始化 NVIDIA ML 庫
        self._initialize_nvml()
    
    def _initialize_nvml(self) -> bool:
        """
        初始化 NVIDIA ML 庫
        
        Returns:
            True 如果初始化成功，False 如果失敗
        """
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.logger.info(f"NVIDIA ML 庫初始化成功，偵測到 {self.gpu_count} 個 GPU")
            
            # 取得所有 GPU 控制代碼
            self.gpu_handles = []
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    self.logger.info(f"GPU {i}: {gpu_name}")
                except pynvml.NVMLError as e:
                    self.logger.error(f"無法取得 GPU {i} 控制代碼: {e}")
                    self.gpu_handles.append(None)
            
            self.is_initialized = True
            return True
            
        except pynvml.NVMLError as e:
            self.logger.error(f"NVIDIA ML 庫初始化失敗: {e}")
            self.is_initialized = False
            return False
        except Exception as e:
            self.logger.error(f"初始化過程中發生未預期錯誤: {e}")
            self.is_initialized = False
            return False
    
    def get_gpu_count(self) -> int:
        """
        取得 GPU 數量
        
        Returns:
            GPU 數量
        """
        return self.gpu_count if self.is_initialized else 0
    
    def get_gpu_name(self, gpu_id: int) -> str:
        """
        取得 GPU 名稱
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            GPU 名稱，如果取得失敗則返回 "Unknown"
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return "Unknown"
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return "Unknown"
        
        try:
            return pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 名稱: {e}")
            return "Unknown"
    
    def get_gpu_utilization(self, gpu_id: int) -> Dict[str, float]:
        """
        取得 GPU 使用率
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            包含 gpu 和 memory 使用率的字典
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return {"gpu": 0.0, "memory": 0.0}
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return {"gpu": 0.0, "memory": 0.0}
        
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {
                "gpu": safe_float(utilization.gpu),
                "memory": safe_float(utilization.memory)
            }
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 使用率: {e}")
            return {"gpu": 0.0, "memory": 0.0}
    
    def get_gpu_memory_info(self, gpu_id: int) -> Dict[str, int]:
        """
        取得 GPU 記憶體資訊
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            包含 total, used, free 記憶體的字典（單位: MB）
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return {"total": 0, "used": 0, "free": 0}
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return {"total": 0, "used": 0, "free": 0}
        
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = safe_int(memory_info.total // (1024 * 1024))
            used_mb = safe_int(memory_info.used // (1024 * 1024))
            free_mb = safe_int(memory_info.free // (1024 * 1024))
            
            return {
                "total": total_mb,
                "used": used_mb,
                "free": free_mb
            }
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 記憶體資訊: {e}")
            return {"total": 0, "used": 0, "free": 0}
    
    def get_gpu_temperature(self, gpu_id: int) -> int:
        """
        取得 GPU 溫度
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            GPU 溫度（攝氏度）
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return 0
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return 0
        
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return safe_int(temperature)
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 溫度: {e}")
            return 0
    
    def get_gpu_power_info(self, gpu_id: int) -> Dict[str, float]:
        """
        取得 GPU 功耗資訊
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            包含 draw 和 limit 功耗的字典（單位: W）
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return {"draw": 0.0, "limit": 0.0}
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return {"draw": 0.0, "limit": 0.0}
        
        power_draw = 0.0
        power_limit = 0.0
        
        try:
            # 嘗試取得當前功耗
            if hasattr(pynvml, 'nvmlDeviceGetPowerUsage'):
                power_draw = safe_float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            else:
                power_draw = 0.0
                self.logger.debug(f"GPU {gpu_id} 不支援功耗查詢")
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 當前功耗: {e}")
            power_draw = 0.0
        
        try:
            # 嘗試取得功耗限制（新版本 API）
            if hasattr(pynvml, 'nvmlDeviceGetPowerManagementLimitDefault'):
                power_limit = safe_float(pynvml.nvmlDeviceGetPowerManagementLimitDefault(handle) / 1000.0)
            elif hasattr(pynvml, 'nvmlDeviceGetPowerManagementLimit'):
                # 舊版本 API
                power_limit = safe_float(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
            else:
                # 如果都不支援，使用預設值
                power_limit = 0.0
                self.logger.debug(f"GPU {gpu_id} 不支援功耗限制查詢")
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 功耗限制: {e}")
            power_limit = 0.0
        
        return {"draw": power_draw, "limit": power_limit}
    
    def get_gpu_fan_speed(self, gpu_id: int) -> int:
        """
        取得 GPU 風扇轉速
        
        Args:
            gpu_id: GPU 編號
            
        Returns:
            風扇轉速百分比
        """
        if not self.is_initialized or gpu_id >= len(self.gpu_handles):
            return 0
        
        handle = self.gpu_handles[gpu_id]
        if handle is None:
            return 0
        
        try:
            # 嘗試取得風扇轉速
            if hasattr(pynvml, 'nvmlDeviceGetFanSpeed'):
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                return safe_int(fan_speed)
            else:
                self.logger.debug(f"GPU {gpu_id} 不支援風扇轉速查詢")
                return 0
        except pynvml.NVMLError as e:
            self.logger.warning(f"無法取得 GPU {gpu_id} 風扇轉速: {e}")
            return 0
    
    def collect_single_gpu_metrics(self, gpu_id: int, timestamp: Optional[str] = None) -> Optional[GPUMetrics]:
        """
        收集單一 GPU 的所有指標
        
        Args:
            gpu_id: GPU 編號
            timestamp: 時間戳記，如果為 None 則使用當前時間
            
        Returns:
            GPUMetrics 物件，如果收集失敗則返回 None
        """
        if not self.is_initialized or gpu_id >= self.gpu_count:
            return None
        
        if timestamp is None:
            timestamp = format_timestamp()
        
        try:
            # 收集所有指標
            gpu_name = self.get_gpu_name(gpu_id)
            utilization = self.get_gpu_utilization(gpu_id)
            memory_info = self.get_gpu_memory_info(gpu_id)
            temperature = self.get_gpu_temperature(gpu_id)
            power_info = self.get_gpu_power_info(gpu_id)
            fan_speed = self.get_gpu_fan_speed(gpu_id)
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_id=gpu_id,
                gpu_name=gpu_name,
                utilization_gpu=utilization["gpu"],
                utilization_memory=utilization["memory"],
                memory_total=memory_info["total"],
                memory_used=memory_info["used"],
                memory_free=memory_info["free"],
                temperature=temperature,
                power_draw=power_info["draw"],
                power_limit=power_info["limit"],
                fan_speed=fan_speed
            )
            
        except Exception as e:
            self.logger.error(f"收集 GPU {gpu_id} 指標時發生錯誤: {e}")
            return None
    
    def collect_all_gpu_metrics(self, timestamp: Optional[str] = None) -> List[GPUMetrics]:
        """
        收集所有 GPU 的指標
        
        Args:
            timestamp: 時間戳記，如果為 None 則使用當前時間
            
        Returns:
            GPUMetrics 物件列表
        """
        if not self.is_initialized:
            return []
        
        if timestamp is None:
            timestamp = format_timestamp()
        
        metrics_list = []
        for gpu_id in range(self.gpu_count):
            metrics = self.collect_single_gpu_metrics(gpu_id, timestamp)
            if metrics is not None:
                metrics_list.append(metrics)
        
        return metrics_list
    
    def start_monitoring(self, interval_seconds: int = 5, callback=None) -> None:
        """
        開始監控（無限迴圈）
        
        Args:
            interval_seconds: 監控間隔秒數
            callback: 回調函數，接收 GPUMetrics 列表
        """
        if not self.is_initialized:
            self.logger.error("GPU 監控器未初始化，無法開始監控")
            return
        
        self.logger.info(f"開始監控 {self.gpu_count} 個 GPU，間隔 {interval_seconds} 秒")
        
        try:
            while True:
                start_time = time.time()
                
                # 收集指標
                metrics_list = self.collect_all_gpu_metrics()
                
                # 如果有回調函數，調用它
                if callback and metrics_list:
                    try:
                        callback(metrics_list)
                    except Exception as e:
                        self.logger.error(f"回調函數執行失敗: {e}")
                
                # 計算睡眠時間
                elapsed_time = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"收集指標耗時 {elapsed_time:.2f} 秒，超過設定間隔 {interval_seconds} 秒")
                    
        except KeyboardInterrupt:
            self.logger.info("接收到中斷訊號，停止監控")
        except Exception as e:
            self.logger.error(f"監控過程中發生錯誤: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """
        清理資源
        """
        try:
            if self.is_initialized:
                pynvml.nvmlShutdown()
                self.logger.info("NVIDIA ML 庫已關閉")
        except pynvml.NVMLError as e:
            self.logger.warning(f"關閉 NVIDIA ML 庫時發生錯誤: {e}")
        
        self.is_initialized = False
        self.gpu_handles = []
        self.gpu_count = 0
    
    def __del__(self):
        """
        析構函數，確保資源被正確清理
        """
        self.cleanup()
