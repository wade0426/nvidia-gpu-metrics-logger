#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - 工具函數模組

提供時間格式化、檔案管理、設定驗證等工具函數
"""

import os
import logging
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None, 
                 console_output: bool = True) -> logging.Logger:
    """
    設定日誌記錄器
    
    Args:
        log_level: 日誌等級
        log_file: 日誌檔案路徑
        console_output: 是否輸出到控制台
        
    Returns:
        配置好的日誌記錄器
    """
    logger = logging.getLogger("gpu_monitor")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除現有處理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台處理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 檔案處理器
    if log_file:
        ensure_directory_exists(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    格式化時間戳記
    
    Args:
        dt: datetime 物件，如果為 None 則使用當前時間
        
    Returns:
        格式化後的時間字串 (YYYY-MM-DD HH:MM:SS)
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_file_timestamp(dt: Optional[datetime] = None) -> str:
    """
    格式化檔案名稱用的時間戳記
    
    Args:
        dt: datetime 物件，如果為 None 則使用當前時間
        
    Returns:
        適合檔案名稱的時間字串 (YYYYMMDD_HHMMSS)
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def ensure_directory_exists(directory_path: Union[str, Path]) -> bool:
    """
    確保目錄存在，如不存在則建立
    
    Args:
        directory_path: 目錄路徑
        
    Returns:
        True 如果目錄存在或成功建立，False 如果建立失敗
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        logging.error(f"無法建立目錄 {directory_path}: {e}")
        return False


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    取得檔案大小（MB）
    
    Args:
        file_path: 檔案路徑
        
    Returns:
        檔案大小（MB），如果檔案不存在則返回 0
    """
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, FileNotFoundError):
        return 0.0


def validate_config(config: configparser.ConfigParser) -> Dict[str, Any]:
    """
    驗證設定檔內容並返回處理後的設定
    
    Args:
        config: ConfigParser 物件
        
    Returns:
        驗證後的設定字典
        
    Raises:
        ValueError: 設定值無效時
    """
    validated_config = {}
    
    # 監控設定驗證
    monitoring_section = config['MONITORING']
    
    # 檢查間隔時間
    interval = monitoring_section.getint('interval_seconds', 5)
    if interval < 1:
        raise ValueError("監控間隔時間必須大於等於 1 秒")
    validated_config['interval_seconds'] = interval
    
    # 檢查輸出目錄
    output_dir = monitoring_section.get('output_directory', './logs')
    if not ensure_directory_exists(output_dir):
        raise ValueError(f"無法建立輸出目錄: {output_dir}")
    validated_config['output_directory'] = output_dir
    
    # 檢查檔案名稱前綴
    filename_prefix = monitoring_section.get('csv_filename_prefix', 'gpu_metrics')
    if not filename_prefix.strip():
        raise ValueError("檔案名稱前綴不能為空")
    validated_config['csv_filename_prefix'] = filename_prefix.strip()
    
    # 檢查最大檔案大小
    max_size = monitoring_section.getint('max_file_size_mb', 100)
    if max_size < 1:
        raise ValueError("最大檔案大小必須大於等於 1 MB")
    validated_config['max_file_size_mb'] = max_size
    
    # GPU 指標設定驗證
    if config.has_section('GPU_METRICS'):
        gpu_metrics = config['GPU_METRICS']
        validated_config['include_utilization'] = gpu_metrics.getboolean('include_utilization', True)
        validated_config['include_memory'] = gpu_metrics.getboolean('include_memory', True)
        validated_config['include_temperature'] = gpu_metrics.getboolean('include_temperature', True)
        validated_config['include_power'] = gpu_metrics.getboolean('include_power', True)
        validated_config['include_fan_speed'] = gpu_metrics.getboolean('include_fan_speed', True)
    else:
        # 預設值
        validated_config.update({
            'include_utilization': True,
            'include_memory': True,
            'include_temperature': True,
            'include_power': True,
            'include_fan_speed': True
        })
    
    # 日誌設定驗證
    if config.has_section('LOGGING'):
        logging_section = config['LOGGING']
        validated_config['log_level'] = logging_section.get('log_level', 'INFO').upper()
        validated_config['log_file'] = logging_section.get('log_file', './logs/monitor.log')
        validated_config['console_output'] = logging_section.getboolean('console_output', True)
    else:
        validated_config.update({
            'log_level': 'INFO',
            'log_file': './logs/monitor.log',
            'console_output': True
        })
    
    # 效能設定驗證
    if config.has_section('PERFORMANCE'):
        performance_section = config['PERFORMANCE']
        
        buffer_size = performance_section.getint('buffer_size', 100)
        if buffer_size < 1:
            raise ValueError("緩存大小必須大於等於 1")
        validated_config['buffer_size'] = buffer_size
        
        validated_config['enable_threading'] = performance_section.getboolean('enable_threading', True)
        
        batch_size = performance_section.getint('batch_write_size', 50)
        if batch_size < 1:
            raise ValueError("批次寫入大小必須大於等於 1")
        validated_config['batch_write_size'] = batch_size
    else:
        validated_config.update({
            'buffer_size': 100,
            'enable_threading': True,
            'batch_write_size': 50
        })
    
    return validated_config


def load_config(config_path: str = "config/config.ini") -> Dict[str, Any]:
    """
    載入並驗證設定檔
    
    Args:
        config_path: 設定檔路徑
        
    Returns:
        驗證後的設定字典
        
    Raises:
        FileNotFoundError: 設定檔不存在
        ValueError: 設定值無效
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定檔不存在: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    return validate_config(config)


def format_bytes(bytes_value: int) -> str:
    """
    格式化位元組數為人類可讀格式
    
    Args:
        bytes_value: 位元組數
        
    Returns:
        格式化後的字串 (例: "1.5 GB", "512 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全轉換為浮點數
    
    Args:
        value: 要轉換的值
        default: 轉換失敗時的預設值
        
    Returns:
        轉換後的浮點數
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    安全轉換為整數
    
    Args:
        value: 要轉換的值
        default: 轉換失敗時的預設值
        
    Returns:
        轉換後的整數
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
