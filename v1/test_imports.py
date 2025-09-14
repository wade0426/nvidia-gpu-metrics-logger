#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
導入測試腳本

測試所有模組的導入是否正常工作
"""

import os
import sys

# 添加 src 目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_utils_import():
    """測試 utils 模組導入"""
    try:
        from utils import (
            format_timestamp, format_file_timestamp, ensure_directory_exists,
            get_file_size_mb, validate_config, load_config, format_bytes,
            safe_float, safe_int, setup_logging
        )
        print("✓ utils 模組導入成功")
        return True
    except Exception as e:
        print(f"❌ utils 模組導入失敗: {e}")
        return False

def test_gpu_monitor_import():
    """測試 gpu_monitor 模組導入（不初始化 NVML）"""
    try:
        # 暫時禁用 NVML 導入
        import sys
        sys.modules['pynvml'] = type(sys)('mock_pynvml')
        sys.modules['pynvml'].nvmlInit = lambda: None
        sys.modules['pynvml'].nvmlDeviceGetCount = lambda: 0
        sys.modules['pynvml'].NVMLError = Exception
        
        from gpu_monitor import GPUMetrics
        print("✓ gpu_monitor 模組導入成功")
        
        # 測試 GPUMetrics 創建
        metrics = GPUMetrics(
            timestamp="2024-01-01 12:00:00",
            gpu_id=0,
            gpu_name="Test GPU",
            utilization_gpu=50.0,
            utilization_memory=60.0,
            memory_total=8192,
            memory_used=4096,
            memory_free=4096,
            temperature=65,
            power_draw=150.0,
            power_limit=200.0,
            fan_speed=40
        )
        print("✓ GPUMetrics 物件創建成功")
        return True
    except Exception as e:
        print(f"❌ gpu_monitor 模組導入失敗: {e}")
        return False

def test_data_logger_import():
    """測試 data_logger 模組導入"""
    try:
        from data_logger import DataLogger
        print("✓ data_logger 模組導入成功")
        return True
    except Exception as e:
        print(f"❌ data_logger 模組導入失敗: {e}")
        return False

def test_main_import():
    """測試 main 模組導入"""
    try:
        # 暫時模擬 pynvml
        import sys
        if 'pynvml' not in sys.modules:
            sys.modules['pynvml'] = type(sys)('mock_pynvml')
            sys.modules['pynvml'].nvmlInit = lambda: None
            sys.modules['pynvml'].nvmlDeviceGetCount = lambda: 0
            sys.modules['pynvml'].NVMLError = Exception
        
        from main import MainController, parse_arguments
        print("✓ main 模組導入成功")
        return True
    except Exception as e:
        print(f"❌ main 模組導入失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("=== 導入測試開始 ===")
    
    results = []
    results.append(test_utils_import())
    results.append(test_gpu_monitor_import())
    results.append(test_data_logger_import())
    results.append(test_main_import())
    
    print("\n=== 測試結果 ===")
    if all(results):
        print("🎉 所有模組導入測試通過！")
        print("可以嘗試執行：python run_monitor.py --test")
        return 0
    else:
        print("❌ 部分模組導入失敗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
