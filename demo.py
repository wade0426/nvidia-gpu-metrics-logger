#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger 演示腳本

這個腳本展示如何使用 GPU 監控系統的基本功能
"""

import os
import sys
import time

# 添加 src 目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def demo_basic_usage():
    """演示基本使用方式"""
    print("=== NVIDIA GPU Metrics Logger 演示 ===")
    
    try:
        from utils import load_config
        from gpu_monitor import GPUMonitor
        from data_logger import DataLogger
        
        print("1. 載入設定檔...")
        config = load_config("config/config.ini")
        print("   ✓ 設定檔載入成功")
        
        print("2. 初始化 GPU 監控器...")
        gpu_monitor = GPUMonitor(config)
        gpu_count = gpu_monitor.get_gpu_count()
        print(f"   ✓ 偵測到 {gpu_count} 個 GPU")
        
        if gpu_count == 0:
            print("   ❌ 未偵測到可用的 NVIDIA GPU")
            print("   請確認：")
            print("   - 系統有 NVIDIA GPU")
            print("   - NVIDIA 驅動程式已安裝")
            print("   - nvidia-smi 命令可以正常執行")
            return
        
        # 顯示 GPU 資訊
        print("3. GPU 資訊：")
        for gpu_id in range(gpu_count):
            gpu_name = gpu_monitor.get_gpu_name(gpu_id)
            memory_info = gpu_monitor.get_gpu_memory_info(gpu_id)
            print(f"   GPU {gpu_id}: {gpu_name}")
            print(f"            記憶體: {memory_info['total']} MB")
        
        print("4. 初始化資料記錄器...")
        data_logger = DataLogger(config)
        print("   ✓ 資料記錄器初始化成功")
        
        print("5. 收集 GPU 指標...")
        metrics_list = gpu_monitor.collect_all_gpu_metrics()
        print(f"   ✓ 成功收集 {len(metrics_list)} 個 GPU 的指標")
        
        print("6. 記錄資料到 CSV...")
        success = data_logger.log_metrics(metrics_list)
        if success:
            print("   ✓ 資料記錄成功")
            
            # 顯示檔案資訊
            file_info = data_logger.get_file_info()
            print(f"   檔案: {file_info['file_path']}")
            print(f"   大小: {file_info['file_size_mb']:.3f} MB")
            print(f"   記錄數: {file_info['record_count']}")
        else:
            print("   ❌ 資料記錄失敗")
        
        print("7. 顯示即時指標：")
        for metrics in metrics_list:
            print(f"   GPU {metrics.gpu_id} ({metrics.gpu_name}):")
            print(f"     使用率: GPU {metrics.utilization_gpu}%, 記憶體 {metrics.utilization_memory}%")
            print(f"     記憶體: {metrics.memory_used}/{metrics.memory_total} MB")
            print(f"     溫度: {metrics.temperature}°C")
            print(f"     功耗: {metrics.power_draw:.1f}W / {metrics.power_limit:.1f}W")
            print(f"     風扇: {metrics.fan_speed}%")
        
        print("8. 清理資源...")
        data_logger.cleanup()
        gpu_monitor.cleanup()
        print("   ✓ 清理完成")
        
        print("\n=== 演示完成 ===")
        print("如需持續監控，請執行：python run_monitor.py")
        
    except ImportError as e:
        print(f"❌ 導入模組失敗: {e}")
        print("請確認已安裝必要套件：pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")


def demo_continuous_monitoring():
    """演示持續監控（短時間）"""
    print("\n=== 持續監控演示（10 秒） ===")
    
    try:
        from main import MainController
        
        print("初始化主控制器...")
        controller = MainController("config/config.ini")
        
        print("開始監控（10 秒後自動停止）...")
        
        # 模擬短時間監控
        start_time = time.time()
        
        try:
            # 啟動監控（在另一個執行緒中）
            import threading
            
            def monitor_worker():
                controller.start_monitoring()
            
            monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
            monitor_thread.start()
            
            # 等待 10 秒
            time.sleep(10)
            
            # 停止監控
            controller.stop()
            
            print("✓ 短時間監控演示完成")
            
        except KeyboardInterrupt:
            print("監控被使用者中斷")
            controller.stop()
        
    except Exception as e:
        print(f"❌ 持續監控演示失敗: {e}")


if __name__ == "__main__":
    # 基本使用演示
    demo_basic_usage()
    
    # 詢問是否要進行持續監控演示
    print("\n是否要進行持續監控演示？(y/N): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        demo_continuous_monitoring()
    
    print("\n感謝使用 NVIDIA GPU Metrics Logger！")
