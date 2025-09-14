#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 監控器單元測試
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu_monitor import GPUMonitor, GPUMetrics
from utils import load_config


class TestGPUMonitor(unittest.TestCase):
    """GPU 監控器測試類別"""
    
    def setUp(self):
        """測試前置作業"""
        self.test_config = {
            'interval_seconds': 5,
            'output_directory': './logs',
            'csv_filename_prefix': 'test_gpu_metrics',
            'max_file_size_mb': 100,
            'include_utilization': True,
            'include_memory': True,
            'include_temperature': True,
            'include_power': True,
            'include_fan_speed': True,
            'log_level': 'INFO',
            'console_output': True,
        }
    
    @patch('gpu_monitor.pynvml')
    def test_gpu_monitor_init_success(self, mock_pynvml):
        """測試 GPU 監控器成功初始化"""
        # 模擬 NVIDIA ML 庫
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        mock_pynvml.nvmlDeviceGetName.return_value = b"GeForce RTX 3080"
        
        # 創建監控器
        monitor = GPUMonitor(self.test_config)
        
        # 驗證初始化
        self.assertTrue(monitor.is_initialized)
        self.assertEqual(monitor.get_gpu_count(), 2)
        
        # 清理
        monitor.cleanup()
    
    @patch('gpu_monitor.pynvml')
    def test_gpu_monitor_init_failure(self, mock_pynvml):
        """測試 GPU 監控器初始化失敗"""
        # 模擬初始化失敗
        mock_pynvml.nvmlInit.side_effect = Exception("NVML 初始化失敗")
        
        # 創建監控器
        monitor = GPUMonitor(self.test_config)
        
        # 驗證初始化失敗
        self.assertFalse(monitor.is_initialized)
        self.assertEqual(monitor.get_gpu_count(), 0)
    
    @patch('gpu_monitor.pynvml')
    def test_collect_gpu_metrics(self, mock_pynvml):
        """測試收集 GPU 指標"""
        # 模擬 NVIDIA ML 庫
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"GeForce RTX 3080"
        
        # 模擬指標資料
        mock_utilization = Mock()
        mock_utilization.gpu = 75
        mock_utilization.memory = 60
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization
        
        mock_memory = Mock()
        mock_memory.total = 12 * 1024 * 1024 * 1024  # 12GB
        mock_memory.used = 8 * 1024 * 1024 * 1024    # 8GB
        mock_memory.free = 4 * 1024 * 1024 * 1024    # 4GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
        
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 250000  # 250W in mW
        mock_pynvml.nvmlDeviceGetPowerManagementLimitDefault.return_value = 320000  # 320W in mW
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 45
        
        # 創建監控器並收集指標
        monitor = GPUMonitor(self.test_config)
        metrics = monitor.collect_single_gpu_metrics(0)
        
        # 驗證指標
        self.assertIsInstance(metrics, GPUMetrics)
        self.assertEqual(metrics.gpu_id, 0)
        self.assertEqual(metrics.gpu_name, "GeForce RTX 3080")
        self.assertEqual(metrics.utilization_gpu, 75)
        self.assertEqual(metrics.utilization_memory, 60)
        self.assertEqual(metrics.memory_total, 12288)  # 12GB in MB
        self.assertEqual(metrics.memory_used, 8192)    # 8GB in MB
        self.assertEqual(metrics.temperature, 65)
        self.assertEqual(metrics.power_draw, 250.0)
        self.assertEqual(metrics.fan_speed, 45)
        
        # 清理
        monitor.cleanup()


class TestGPUMetrics(unittest.TestCase):
    """GPU 指標資料結構測試"""
    
    def test_gpu_metrics_creation(self):
        """測試 GPU 指標物件創建"""
        metrics = GPUMetrics(
            timestamp="2024-01-01 12:00:00",
            gpu_id=0,
            gpu_name="GeForce RTX 3080",
            utilization_gpu=75.0,
            utilization_memory=60.0,
            memory_total=12288,
            memory_used=8192,
            memory_free=4096,
            temperature=65,
            power_draw=250.0,
            power_limit=320.0,
            fan_speed=45
        )
        
        # 驗證所有欄位
        self.assertEqual(metrics.timestamp, "2024-01-01 12:00:00")
        self.assertEqual(metrics.gpu_id, 0)
        self.assertEqual(metrics.gpu_name, "GeForce RTX 3080")
        self.assertEqual(metrics.utilization_gpu, 75.0)
        self.assertEqual(metrics.utilization_memory, 60.0)
        self.assertEqual(metrics.memory_total, 12288)
        self.assertEqual(metrics.memory_used, 8192)
        self.assertEqual(metrics.memory_free, 4096)
        self.assertEqual(metrics.temperature, 65)
        self.assertEqual(metrics.power_draw, 250.0)
        self.assertEqual(metrics.power_limit, 320.0)
        self.assertEqual(metrics.fan_speed, 45)


if __name__ == '__main__':
    unittest.main()
