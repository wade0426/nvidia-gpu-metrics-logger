#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - 主控程式

整合 GPU 監控器與資料記錄器，提供完整的監控服務
"""

import os
import sys
import time
import signal
import argparse
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# 添加 src 目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 嘗試相對導入（作為模組時）
    from .utils import load_config, setup_logging, format_timestamp
    from .gpu_monitor import GPUMonitor, GPUMetrics
    from .data_logger import DataLogger
except ImportError:
    # 絕對導入（作為腳本直接執行時）
    from utils import load_config, setup_logging, format_timestamp
    from gpu_monitor import GPUMonitor, GPUMetrics
    from data_logger import DataLogger


class MainController:
    """
    主控制器
    
    負責協調 GPU 監控器與資料記錄器，管理整體監控流程
    """
    
    def __init__(self, config_path: str = "config/config.ini"):
        """
        初始化主控制器
        
        Args:
            config_path: 設定檔路徑
        """
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.gpu_monitor = None
        self.data_logger = None
        self.is_running = False
        self.restart_count = 0
        self.max_restart_count = 5
        
        # 註冊訊號處理器
        self._register_signal_handlers()
        
        # 載入設定
        self._load_configuration()
        
        # 設定日誌
        self._setup_logging()
        
        # 初始化組件
        self._initialize_components()
    
    def _load_configuration(self) -> None:
        """
        載入設定檔
        """
        try:
            self.config = load_config(self.config_path)
            print(f"設定檔載入成功: {self.config_path}")
        except Exception as e:
            print(f"載入設定檔失敗: {e}")
            sys.exit(1)
    
    def _setup_logging(self) -> None:
        """
        設定日誌記錄
        """
        try:
            self.logger = setup_logging(
                log_level=self.config.get('log_level', 'INFO'),
                log_file=self.config.get('log_file'),
                console_output=self.config.get('console_output', True)
            )
            self.logger.info("GPU 監控系統啟動")
            self.logger.info(f"設定檔: {self.config_path}")
        except Exception as e:
            print(f"設定日誌失敗: {e}")
            sys.exit(1)
    
    def _initialize_components(self) -> None:
        """
        初始化監控組件
        """
        try:
            # 初始化 GPU 監控器
            self.logger.info("初始化 GPU 監控器...")
            self.gpu_monitor = GPUMonitor(self.config)
            
            if self.gpu_monitor.get_gpu_count() == 0:
                raise RuntimeError("未偵測到可用的 NVIDIA GPU")
            
            # 初始化資料記錄器
            self.logger.info("初始化資料記錄器...")
            self.data_logger = DataLogger(self.config)
            
            self.logger.info("所有組件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化組件失敗: {e}")
            raise
    
    def _register_signal_handlers(self) -> None:
        """
        註冊訊號處理器
        """
        def signal_handler(signum, frame):
            print(f"\\n接收到訊號 {signum}，正在停止監控...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _metrics_callback(self, metrics_list: List[GPUMetrics]) -> None:
        """
        GPU 指標回調函數
        
        Args:
            metrics_list: GPU 指標列表
        """
        try:
            # 記錄指標到檔案
            success = self.data_logger.log_metrics(metrics_list)
            
            if success:
                gpu_count = len(metrics_list)
                self.logger.debug(f"成功記錄 {gpu_count} 個 GPU 的指標資料")
                
                # 顯示即時資訊（每 10 次記錄顯示一次詳細資訊）
                if hasattr(self, '_log_counter'):
                    self._log_counter += 1
                else:
                    self._log_counter = 1
                
                if self._log_counter % 10 == 0:
                    self._display_metrics_summary(metrics_list)
            else:
                self.logger.warning("記錄指標資料失敗")
                
        except Exception as e:
            self.logger.error(f"處理指標回調時發生錯誤: {e}")
    
    def _display_metrics_summary(self, metrics_list: List[GPUMetrics]) -> None:
        """
        顯示指標摘要資訊
        
        Args:
            metrics_list: GPU 指標列表
        """
        try:
            summary_lines = [f"=== GPU 監控摘要 ({format_timestamp()}) ==="]
            
            for metrics in metrics_list:
                line = (
                    f"GPU {metrics.gpu_id} ({metrics.gpu_name}): "
                    f"使用率 {metrics.utilization_gpu}%, "
                    f"記憶體 {metrics.memory_used}/{metrics.memory_total}MB "
                    f"({metrics.utilization_memory}%), "
                    f"溫度 {metrics.temperature}°C, "
                    f"功耗 {metrics.power_draw:.1f}W"
                )
                summary_lines.append(line)
            
            file_info = self.data_logger.get_file_info()
            summary_lines.append(
                f"檔案: {os.path.basename(file_info['file_path'])}, "
                f"大小: {file_info['file_size_mb']:.1f}MB, "
                f"記錄數: {file_info['record_count']}"
            )
            
            for line in summary_lines:
                self.logger.info(line)
                
        except Exception as e:
            self.logger.warning(f"顯示摘要資訊時發生錯誤: {e}")
    
    def start_monitoring(self) -> None:
        """
        開始監控
        """
        if self.is_running:
            self.logger.warning("監控已在執行中")
            return
        
        try:
            self.is_running = True
            interval = self.config.get('interval_seconds', 5)
            
            self.logger.info(f"開始監控 {self.gpu_monitor.get_gpu_count()} 個 GPU")
            self.logger.info(f"監控間隔: {interval} 秒")
            self.logger.info(f"輸出目錄: {self.config.get('output_directory')}")
            
            # 顯示初始 GPU 資訊
            self._display_initial_gpu_info()
            
            # 開始監控（會阻塞在這裡）
            self.gpu_monitor.start_monitoring(
                interval_seconds=interval,
                callback=self._metrics_callback
            )
            
        except Exception as e:
            self.logger.error(f"監控過程中發生錯誤: {e}")
            self.is_running = False
            raise
    
    def _display_initial_gpu_info(self) -> None:
        """
        顯示初始 GPU 資訊
        """
        try:
            self.logger.info("=== 偵測到的 GPU 資訊 ===")
            for gpu_id in range(self.gpu_monitor.get_gpu_count()):
                gpu_name = self.gpu_monitor.get_gpu_name(gpu_id)
                memory_info = self.gpu_monitor.get_gpu_memory_info(gpu_id)
                self.logger.info(
                    f"GPU {gpu_id}: {gpu_name}, "
                    f"記憶體: {memory_info['total']} MB"
                )
        except Exception as e:
            self.logger.warning(f"顯示 GPU 資訊時發生錯誤: {e}")
    
    def stop(self) -> None:
        """
        停止監控
        """
        if not self.is_running:
            return
        
        self.logger.info("正在停止監控...")
        self.is_running = False
        
        try:
            # 清空資料記錄器緩存
            if self.data_logger:
                self.logger.info("正在清空資料緩存...")
                self.data_logger.cleanup()
            
            # 清理 GPU 監控器
            if self.gpu_monitor:
                self.gpu_monitor.cleanup()
            
            self.logger.info("監控已停止")
            
        except Exception as e:
            self.logger.error(f"停止監控時發生錯誤: {e}")
    
    def restart(self) -> bool:
        """
        重啟監控
        
        Returns:
            True 如果重啟成功，False 如果失敗
        """
        if self.restart_count >= self.max_restart_count:
            self.logger.error(f"重啟次數已達上限 ({self.max_restart_count})，停止重啟")
            return False
        
        self.restart_count += 1
        self.logger.info(f"正在重啟監控 (第 {self.restart_count} 次)...")
        
        try:
            # 停止當前監控
            self.stop()
            
            # 等待一段時間
            time.sleep(5)
            
            # 重新初始化組件
            self._initialize_components()
            
            # 重新開始監控
            self.start_monitoring()
            
            self.logger.info("重啟成功")
            return True
            
        except Exception as e:
            self.logger.error(f"重啟失敗: {e}")
            return False
    
    def run_with_auto_restart(self) -> None:
        """
        執行監控並支援自動重啟
        """
        while True:
            try:
                self.start_monitoring()
                break  # 正常結束
                
            except KeyboardInterrupt:
                self.logger.info("接收到中斷訊號，停止監控")
                break
                
            except Exception as e:
                self.logger.error(f"監控發生錯誤: {e}")
                
                if not self.restart():
                    self.logger.error("自動重啟失敗，程式退出")
                    break
        
        # 確保清理
        self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """
        取得監控狀態
        
        Returns:
            包含狀態資訊的字典
        """
        status = {
            'is_running': self.is_running,
            'restart_count': self.restart_count,
            'config_path': self.config_path,
            'gpu_count': 0,
            'file_info': None
        }
        
        if self.gpu_monitor:
            status['gpu_count'] = self.gpu_monitor.get_gpu_count()
        
        if self.data_logger:
            status['file_info'] = self.data_logger.get_file_info()
        
        return status


def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數
    
    Returns:
        解析後的參數
    """
    parser = argparse.ArgumentParser(
        description="NVIDIA GPU Metrics Logger - GPU 監控系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py                              # 使用預設設定檔
  python main.py --config custom_config.ini  # 使用自訂設定檔
  python main.py --test                       # 執行測試模式
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.ini',
        help='設定檔路徑 (預設: config/config.ini)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='執行測試模式（收集一次指標後退出）'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='NVIDIA GPU Metrics Logger 1.0.0'
    )
    
    return parser.parse_args()


def test_mode(config_path: str) -> None:
    """
    測試模式：收集一次指標並顯示結果
    
    Args:
        config_path: 設定檔路徑
    """
    try:
        print("=== 測試模式 ===")
        
        # 載入設定
        config = load_config(config_path)
        print(f"設定檔載入成功: {config_path}")
        
        # 初始化監控器
        gpu_monitor = GPUMonitor(config)
        gpu_count = gpu_monitor.get_gpu_count()
        print(f"偵測到 {gpu_count} 個 GPU")
        
        if gpu_count == 0:
            print("錯誤: 未偵測到可用的 NVIDIA GPU")
            return
        
        # 收集一次指標
        print("\\n正在收集 GPU 指標...")
        metrics_list = gpu_monitor.collect_all_gpu_metrics()
        
        # 顯示結果
        print(f"\\n=== GPU 指標結果 ({format_timestamp()}) ===")
        for metrics in metrics_list:
            print(f"GPU {metrics.gpu_id} ({metrics.gpu_name}):")
            print(f"  使用率: GPU {metrics.utilization_gpu}%, 記憶體 {metrics.utilization_memory}%")
            print(f"  記憶體: {metrics.memory_used}/{metrics.memory_total} MB")
            print(f"  溫度: {metrics.temperature}°C")
            print(f"  功耗: {metrics.power_draw:.1f}W / {metrics.power_limit:.1f}W")
            print(f"  風扇: {metrics.fan_speed}%")
            print()
        
        # 清理
        gpu_monitor.cleanup()
        print("測試完成")
        
    except Exception as e:
        print(f"測試模式失敗: {e}")


def main() -> None:
    """
    主函數
    """
    args = parse_arguments()
    
    try:
        if args.test:
            # 測試模式
            test_mode(args.config)
        else:
            # 正常監控模式
            controller = MainController(args.config)
            controller.run_with_auto_restart()
            
    except KeyboardInterrupt:
        print("\\n程式被使用者中斷")
    except Exception as e:
        print(f"程式發生錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
