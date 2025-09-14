#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
資料記錄器單元測試
"""

import unittest
import tempfile
import os
import sys
import csv
from unittest.mock import Mock, patch

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_logger import DataLogger
from gpu_monitor import GPUMetrics


class TestDataLogger(unittest.TestCase):
    """資料記錄器測試類別"""
    
    def setUp(self):
        """測試前置作業"""
        # 創建臨時目錄
        self.temp_dir = tempfile.mkdtemp()
        
        self.test_config = {
            'output_directory': self.temp_dir,
            'csv_filename_prefix': 'test_gpu_metrics',
            'max_file_size_mb': 1,  # 小檔案便於測試
            'buffer_size': 10,
            'batch_write_size': 5,
            'enable_threading': False,  # 關閉多執行緒以便測試
        }
        
        # 測試用的 GPU 指標
        self.test_metrics = GPUMetrics(
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
    
    def tearDown(self):
        """測試後清理"""
        # 清理臨時檔案
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_logger_init(self):
        """測試資料記錄器初始化"""
        logger = DataLogger(self.test_config)
        
        # 驗證初始化
        self.assertIsNotNone(logger.current_file_path)
        self.assertTrue(os.path.exists(logger.current_file_path))
        
        # 驗證 CSV 檔案有標題
        with open(logger.current_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, DataLogger.CSV_FIELDS)
        
        # 清理
        logger.cleanup()
    
    def test_log_single_metrics(self):
        """測試記錄單一指標"""
        logger = DataLogger(self.test_config)
        
        # 記錄指標
        success = logger.log_single_metrics(self.test_metrics)
        self.assertTrue(success)
        
        # 驗證檔案內容
        with open(logger.current_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # 標題 + 1 筆資料
        
        # 清理
        logger.cleanup()
    
    def test_log_multiple_metrics(self):
        """測試記錄多筆指標"""
        logger = DataLogger(self.test_config)
        
        # 創建多筆測試資料
        metrics_list = []
        for i in range(3):
            metrics = GPUMetrics(
                timestamp=f"2024-01-01 12:0{i}:00",
                gpu_id=i,
                gpu_name=f"GeForce RTX 308{i}",
                utilization_gpu=75.0 + i,
                utilization_memory=60.0 + i,
                memory_total=12288,
                memory_used=8192,
                memory_free=4096,
                temperature=65 + i,
                power_draw=250.0 + i,
                power_limit=320.0,
                fan_speed=45 + i
            )
            metrics_list.append(metrics)
        
        # 記錄指標
        success = logger.log_metrics(metrics_list)
        self.assertTrue(success)
        
        # 驗證檔案內容
        with open(logger.current_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)  # 標題 + 3 筆資料
        
        # 清理
        logger.cleanup()
    
    def test_buffer_operations(self):
        """測試緩存操作"""
        logger = DataLogger(self.test_config)
        
        # 測試緩存大小
        self.assertEqual(logger.get_buffer_size(), 0)
        
        # 添加資料到緩存
        logger.add_to_buffer([self.test_metrics])
        self.assertEqual(logger.get_buffer_size(), 1)
        
        # 清空緩存
        logger.flush_buffer()
        self.assertEqual(logger.get_buffer_size(), 0)
        
        # 驗證檔案有資料
        with open(logger.current_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)  # 標題 + 1 筆資料
        
        # 清理
        logger.cleanup()
    
    def test_file_info(self):
        """測試檔案資訊功能"""
        logger = DataLogger(self.test_config)
        
        # 記錄一些資料
        logger.log_single_metrics(self.test_metrics)
        
        # 取得檔案資訊
        file_info = logger.get_file_info()
        
        # 驗證檔案資訊
        self.assertTrue(file_info['file_exists'])
        self.assertGreater(file_info['file_size_mb'], 0)
        self.assertEqual(file_info['record_count'], 1)
        self.assertEqual(file_info['buffer_size'], 0)
        
        # 清理
        logger.cleanup()
    
    @patch('data_logger.pd.read_csv')
    def test_export_to_dataframe(self, mock_read_csv):
        """測試匯出為 DataFrame"""
        logger = DataLogger(self.test_config)
        
        # 模擬 pandas DataFrame
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        
        # 測試匯出
        df = logger.export_to_dataframe()
        
        # 驗證
        self.assertEqual(df, mock_df)
        mock_read_csv.assert_called_once_with(logger.current_file_path)
        
        # 清理
        logger.cleanup()


if __name__ == '__main__':
    unittest.main()
