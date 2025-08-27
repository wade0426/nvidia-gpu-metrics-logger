#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - 資料記錄器

負責將 GPU 指標資料寫入 CSV 檔案，包含檔案輪替、批次寫入等功能
"""

import os
import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
import pandas as pd

try:
    # 嘗試相對導入（作為模組時）
    from .utils import (
        format_timestamp, format_file_timestamp, ensure_directory_exists, 
        get_file_size_mb, safe_float, safe_int
    )
    from .gpu_monitor import GPUMetrics
except ImportError:
    # 絕對導入（作為腳本直接執行時）
    from utils import (
        format_timestamp, format_file_timestamp, ensure_directory_exists, 
        get_file_size_mb, safe_float, safe_int
    )
    from gpu_monitor import GPUMetrics


class DataLogger:
    """
    資料記錄器
    
    負責將 GPU 指標資料寫入 CSV 檔案，支援檔案輪替和批次寫入
    """
    
    # CSV 檔案欄位定義
    CSV_FIELDS = [
        'timestamp', 'gpu_id', 'gpu_name',
        'utilization_gpu', 'utilization_memory',
        'memory_total', 'memory_used', 'memory_free',
        'temperature', 'power_draw', 'power_limit', 'fan_speed'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化資料記錄器
        
        Args:
            config: 設定字典
        """
        self.config = config
        self.logger = logging.getLogger("gpu_monitor.DataLogger")
        
        # 設定參數
        self.output_directory = config.get('output_directory', './logs')
        self.filename_prefix = config.get('csv_filename_prefix', 'gpu_metrics')
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.buffer_size = config.get('buffer_size', 100)
        self.batch_write_size = config.get('batch_write_size', 50)
        self.enable_threading = config.get('enable_threading', True)
        
        # 內部狀態
        self.current_file_path = None
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.write_queue = Queue() if self.enable_threading else None
        self.writer_thread = None
        self.is_running = False
        
        # 確保輸出目錄存在
        if not ensure_directory_exists(self.output_directory):
            raise RuntimeError(f"無法建立輸出目錄: {self.output_directory}")
        
        # 初始化 CSV 檔案
        self._initialize_csv_file()
        
        # 如果啟用多執行緒，啟動寫入執行緒
        if self.enable_threading:
            self._start_writer_thread()
    
    def _initialize_csv_file(self) -> None:
        """
        初始化 CSV 檔案
        """
        timestamp = format_file_timestamp()
        # filename = f"{self.filename_prefix}_{timestamp}.csv"
        filename = f"{self.filename_prefix}.csv"
        self.current_file_path = os.path.join(self.output_directory, filename)
        
        try:
            # 創建檔案並寫入標題
            with open(self.current_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
            
            self.logger.info(f"初始化 CSV 檔案: {self.current_file_path}")
            
        except Exception as e:
            self.logger.error(f"初始化 CSV 檔案失敗: {e}")
            raise
    
    def _check_file_rotation(self) -> bool:
        """
        檢查是否需要檔案輪替
        
        Returns:
            True 如果需要輪替，False 如果不需要
        """
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            return True
        
        current_size = get_file_size_mb(self.current_file_path)
        return current_size >= self.max_file_size_mb
    
    def _rotate_file(self) -> None:
        """
        執行檔案輪替
        """
        old_file = self.current_file_path
        self._initialize_csv_file()
        self.logger.info(f"檔案輪替: {old_file} -> {self.current_file_path}")
    
    def _write_to_csv_direct(self, metrics_list: List[GPUMetrics]) -> bool:
        """
        直接寫入 CSV 檔案（同步方式）
        
        Args:
            metrics_list: GPU 指標列表
            
        Returns:
            True 如果寫入成功，False 如果失敗
        """
        if not metrics_list:
            return True
        
        try:
            # 檢查是否需要檔案輪替
            if self._check_file_rotation():
                self._rotate_file()
            
            # 將 GPUMetrics 物件轉換為字典
            rows = []
            for metrics in metrics_list:
                row = {
                    'timestamp': metrics.timestamp,
                    'gpu_id': metrics.gpu_id,
                    'gpu_name': metrics.gpu_name,
                    'utilization_gpu': safe_float(metrics.utilization_gpu),
                    'utilization_memory': safe_float(metrics.utilization_memory),
                    'memory_total': safe_int(metrics.memory_total),
                    'memory_used': safe_int(metrics.memory_used),
                    'memory_free': safe_int(metrics.memory_free),
                    'temperature': safe_int(metrics.temperature),
                    'power_draw': safe_float(metrics.power_draw),
                    'power_limit': safe_float(metrics.power_limit),
                    'fan_speed': safe_int(metrics.fan_speed)
                }
                rows.append(row)
            
            # 寫入檔案
            with open(self.current_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.CSV_FIELDS)
                writer.writerows(rows)
            
            self.logger.debug(f"成功寫入 {len(rows)} 筆資料到 {self.current_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"寫入 CSV 檔案失敗: {e}")
            return False
    
    def _writer_thread_worker(self) -> None:
        """
        寫入執行緒工作函數
        """
        self.logger.info("資料寫入執行緒已啟動")
        
        while self.is_running:
            try:
                # 從佇列取得資料（超時 1 秒）
                metrics_list = self.write_queue.get(timeout=1.0)
                
                if metrics_list is None:  # 停止訊號
                    break
                
                # 寫入資料
                self._write_to_csv_direct(metrics_list)
                self.write_queue.task_done()
                
            except Empty:
                continue  # 超時，繼續迴圈
            except Exception as e:
                self.logger.error(f"寫入執行緒發生錯誤: {e}")
        
        self.logger.info("資料寫入執行緒已停止")
    
    def _start_writer_thread(self) -> None:
        """
        啟動寫入執行緒
        """
        if self.writer_thread and self.writer_thread.is_alive():
            return
        
        self.is_running = True
        self.writer_thread = threading.Thread(
            target=self._writer_thread_worker,
            name="DataLoggerWriter",
            daemon=True
        )
        self.writer_thread.start()
    
    def _stop_writer_thread(self) -> None:
        """
        停止寫入執行緒
        """
        if not self.writer_thread or not self.writer_thread.is_alive():
            return
        
        self.is_running = False
        
        # 發送停止訊號
        if self.write_queue:
            self.write_queue.put(None)
        
        # 等待執行緒結束
        self.writer_thread.join(timeout=5.0)
        
        if self.writer_thread.is_alive():
            self.logger.warning("寫入執行緒無法正常停止")
    
    def log_metrics(self, metrics_list: List[GPUMetrics]) -> bool:
        """
        記錄 GPU 指標資料
        
        Args:
            metrics_list: GPU 指標列表
            
        Returns:
            True 如果記錄成功，False 如果失敗
        """
        if not metrics_list:
            return True
        
        try:
            if self.enable_threading:
                # 異步寫入
                self.write_queue.put(metrics_list)
                return True
            else:
                # 同步寫入
                return self._write_to_csv_direct(metrics_list)
                
        except Exception as e:
            self.logger.error(f"記錄指標資料失敗: {e}")
            return False
    
    def log_single_metrics(self, metrics: GPUMetrics) -> bool:
        """
        記錄單一 GPU 指標資料
        
        Args:
            metrics: GPU 指標物件
            
        Returns:
            True 如果記錄成功，False 如果失敗
        """
        return self.log_metrics([metrics])
    
    def add_to_buffer(self, metrics_list: List[GPUMetrics]) -> None:
        """
        將資料加入緩存
        
        Args:
            metrics_list: GPU 指標列表
        """
        with self.buffer_lock:
            self.data_buffer.extend(metrics_list)
            
            # 檢查是否達到批次寫入大小
            if len(self.data_buffer) >= self.batch_write_size:
                self.flush_buffer()
    
    def flush_buffer(self) -> bool:
        """
        清空緩存並寫入檔案
        
        Returns:
            True 如果寫入成功，False 如果失敗
        """
        with self.buffer_lock:
            if not self.data_buffer:
                return True
            
            # 取出所有緩存資料
            metrics_to_write = self.data_buffer.copy()
            self.data_buffer.clear()
        
        # 寫入資料
        success = self.log_metrics(metrics_to_write)
        
        if success:
            self.logger.debug(f"成功清空緩存，寫入 {len(metrics_to_write)} 筆資料")
        else:
            self.logger.error(f"清空緩存失敗，遺失 {len(metrics_to_write)} 筆資料")
        
        return success
    
    def get_buffer_size(self) -> int:
        """
        取得當前緩存大小
        
        Returns:
            緩存中的資料筆數
        """
        with self.buffer_lock:
            return len(self.data_buffer)
    
    def export_to_dataframe(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        匯出資料為 Pandas DataFrame
        
        Args:
            file_path: CSV 檔案路徑，如果為 None 則使用當前檔案
            
        Returns:
            DataFrame 物件，如果讀取失敗則返回 None
        """
        try:
            target_file = file_path or self.current_file_path
            
            if not os.path.exists(target_file):
                self.logger.warning(f"檔案不存在: {target_file}")
                return None
            
            df = pd.read_csv(target_file)
            self.logger.info(f"成功讀取 {len(df)} 筆資料從 {target_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"匯出 DataFrame 失敗: {e}")
            return None
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        取得當前檔案資訊
        
        Returns:
            包含檔案資訊的字典
        """
        info = {
            'file_path': self.current_file_path,
            'file_exists': False,
            'file_size_mb': 0.0,
            'record_count': 0,
            'buffer_size': self.get_buffer_size()
        }
        
        if self.current_file_path and os.path.exists(self.current_file_path):
            info['file_exists'] = True
            info['file_size_mb'] = get_file_size_mb(self.current_file_path)
            
            # 嘗試計算記錄數
            try:
                with open(self.current_file_path, 'r', encoding='utf-8') as f:
                    # 扣除標題行
                    info['record_count'] = max(0, sum(1 for _ in f) - 1)
            except Exception as e:
                self.logger.warning(f"無法計算記錄數: {e}")
        
        return info
    
    def cleanup(self) -> None:
        """
        清理資源
        """
        try:
            # 清空緩存
            if self.data_buffer:
                self.logger.info("正在清空剩餘緩存資料...")
                self.flush_buffer()
            
            # 停止寫入執行緒
            if self.enable_threading:
                self._stop_writer_thread()
            
            # 等待佇列中的資料處理完成
            if self.write_queue:
                self.write_queue.join()
            
            self.logger.info("資料記錄器已關閉")
            
        except Exception as e:
            self.logger.error(f"清理資源時發生錯誤: {e}")
    
    def __del__(self):
        """
        析構函數，確保資源被正確清理
        """
        self.cleanup()
    
    def __enter__(self):
        """
        上下文管理器進入
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出
        """
        self.cleanup()
