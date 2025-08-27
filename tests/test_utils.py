#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函數單元測試
"""

import unittest
import tempfile
import os
import sys
import configparser
from datetime import datetime

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    format_timestamp, format_file_timestamp, ensure_directory_exists,
    get_file_size_mb, validate_config, load_config, format_bytes,
    safe_float, safe_int
)


class TestUtils(unittest.TestCase):
    """工具函數測試類別"""
    
    def setUp(self):
        """測試前置作業"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """測試後清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_format_timestamp(self):
        """測試時間戳記格式化"""
        # 測試預設時間
        timestamp = format_timestamp()
        self.assertRegex(timestamp, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        
        # 測試指定時間
        test_dt = datetime(2024, 1, 1, 12, 30, 45)
        timestamp = format_timestamp(test_dt)
        self.assertEqual(timestamp, "2024-01-01 12:30:45")
    
    def test_format_file_timestamp(self):
        """測試檔案時間戳記格式化"""
        # 測試預設時間
        timestamp = format_file_timestamp()
        self.assertRegex(timestamp, r'\d{8}_\d{6}')
        
        # 測試指定時間
        test_dt = datetime(2024, 1, 1, 12, 30, 45)
        timestamp = format_file_timestamp(test_dt)
        self.assertEqual(timestamp, "20240101_123045")
    
    def test_ensure_directory_exists(self):
        """測試目錄建立功能"""
        test_dir = os.path.join(self.temp_dir, "test_subdir")
        
        # 目錄不存在時建立
        self.assertFalse(os.path.exists(test_dir))
        result = ensure_directory_exists(test_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(test_dir))
        
        # 目錄已存在時不報錯
        result = ensure_directory_exists(test_dir)
        self.assertTrue(result)
    
    def test_get_file_size_mb(self):
        """測試檔案大小取得功能"""
        # 不存在的檔案
        non_existent_file = os.path.join(self.temp_dir, "non_existent.txt")
        size = get_file_size_mb(non_existent_file)
        self.assertEqual(size, 0.0)
        
        # 建立測試檔案
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "A" * 1024  # 1KB 內容
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        size = get_file_size_mb(test_file)
        self.assertAlmostEqual(size, 1.0 / 1024, places=4)  # 約 0.001 MB
    
    def test_validate_config(self):
        """測試設定檔驗證功能"""
        # 建立測試設定
        config = configparser.ConfigParser()
        config.add_section('MONITORING')
        config.set('MONITORING', 'interval_seconds', '10')
        config.set('MONITORING', 'output_directory', self.temp_dir)
        config.set('MONITORING', 'csv_filename_prefix', 'test_gpu')
        config.set('MONITORING', 'max_file_size_mb', '50')
        
        # 驗證設定
        validated = validate_config(config)
        
        # 檢查驗證結果
        self.assertEqual(validated['interval_seconds'], 10)
        self.assertEqual(validated['output_directory'], self.temp_dir)
        self.assertEqual(validated['csv_filename_prefix'], 'test_gpu')
        self.assertEqual(validated['max_file_size_mb'], 50)
        
        # 檢查預設值
        self.assertTrue(validated['include_utilization'])
        self.assertTrue(validated['include_memory'])
        self.assertEqual(validated['log_level'], 'INFO')
    
    def test_validate_config_invalid_values(self):
        """測試設定檔無效值驗證"""
        config = configparser.ConfigParser()
        config.add_section('MONITORING')
        config.set('MONITORING', 'interval_seconds', '0')  # 無效值
        config.set('MONITORING', 'output_directory', self.temp_dir)
        config.set('MONITORING', 'csv_filename_prefix', 'test')
        config.set('MONITORING', 'max_file_size_mb', '1')
        
        # 應該拋出 ValueError
        with self.assertRaises(ValueError):
            validate_config(config)
    
    def test_load_config(self):
        """測試設定檔載入功能"""
        # 建立測試設定檔
        config_file = os.path.join(self.temp_dir, "test_config.ini")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("""[MONITORING]
interval_seconds = 5
output_directory = ./logs
csv_filename_prefix = gpu_metrics
max_file_size_mb = 100

[GPU_METRICS]
include_utilization = true
include_memory = true
include_temperature = true
include_power = true
include_fan_speed = true
""")
        
        # 載入設定
        config = load_config(config_file)
        
        # 驗證載入結果
        self.assertEqual(config['interval_seconds'], 5)
        self.assertEqual(config['csv_filename_prefix'], 'gpu_metrics')
        self.assertTrue(config['include_utilization'])
    
    def test_load_config_file_not_found(self):
        """測試載入不存在的設定檔"""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.ini")
        
        with self.assertRaises(FileNotFoundError):
            load_config(non_existent_file)
    
    def test_format_bytes(self):
        """測試位元組格式化功能"""
        # 測試各種大小
        self.assertEqual(format_bytes(512), "512.0 B")
        self.assertEqual(format_bytes(1536), "1.5 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.0 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.0 GB")
    
    def test_safe_float(self):
        """測試安全浮點數轉換"""
        # 正常轉換
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float(42), 42.0)
        
        # 異常值使用預設值
        self.assertEqual(safe_float("invalid"), 0.0)
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float("invalid", 99.9), 99.9)
    
    def test_safe_int(self):
        """測試安全整數轉換"""
        # 正常轉換
        self.assertEqual(safe_int("42"), 42)
        self.assertEqual(safe_int(3.14), 3)
        
        # 異常值使用預設值
        self.assertEqual(safe_int("invalid"), 0)
        self.assertEqual(safe_int(None), 0)
        self.assertEqual(safe_int("invalid", 99), 99)


if __name__ == '__main__':
    unittest.main()
