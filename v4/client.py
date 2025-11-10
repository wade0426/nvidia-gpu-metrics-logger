#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - Client 端

Author: GPU Metrics Logger Team
Date: 2025-09-14
"""

import json
import time
import logging
import datetime
import threading
import configparser
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pynvml
except ImportError:
    print("錯誤: 找不到 pynvml 模組。請執行: pip install nvidia-ml-py3")
    exit(1)

try:
    import requests
except ImportError:
    print("錯誤: 找不到 requests 模組。請執行: pip install requests")
    exit(1)


class GPUMetricsClient:
    """GPU 指標收集和傳送客戶端"""
    
    def __init__(self, config_path: str = "./config/client_config.ini"):
        """初始化 GPU 指標客戶端"""
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.data_queue = Queue()
        self.running = False
        self.gpu_count = 0
        
        # 載入配置
        self._load_config()
        # 設置日誌
        self._setup_logging()
        # 初始化 NVIDIA ML
        self._init_nvml()
        # 驗證服務器連線
        self._check_server_connection()
        
    def _load_config(self):
        """載入配置文件"""
        try:
            self.config = configparser.ConfigParser()
            
            if not Path(self.config_path).exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            self.config.read(self.config_path, encoding='utf-8')
            
            # 驗證必要的配置段落
            required_sections = ['AUTHENTICATION', 'MONITORING', 'SERVER', 'NETWORK', 'GPU_METRICS', 'LOGGING']
            for section in required_sections:
                if not self.config.has_section(section):
                    raise ValueError(f"配置文件缺少必要段落: {section}")
                    
        except Exception as e:
            print(f"載入配置文件失敗: {e}")
            exit(1)
    
    def _setup_logging(self):
        """設置日誌系統"""
        try:
            log_level = self.config.get('LOGGING', 'log_level', fallback='INFO')
            log_file = self.config.get('LOGGING', 'log_file', fallback='./logs/client.log')
            console_output = self.config.getboolean('LOGGING', 'console_output', fallback=True)
            log_max_size = self.config.getint('LOGGING', 'log_max_size', fallback=50) * 1024 * 1024
            log_backup_count = self.config.getint('LOGGING', 'log_backup_count', fallback=10)
            
            # 創建日誌目錄
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 設置日誌格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 配置 logger
            self.logger = logging.getLogger('GPUMetricsClient')
            self.logger.setLevel(getattr(logging, log_level.upper()))
            self.logger.handlers.clear()
            
            # 文件處理器
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=log_max_size, backupCount=log_backup_count, encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # 控制台處理器
            if console_output:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                
            self.logger.info("日誌系統初始化完成")
            
        except Exception as e:
            print(f"設置日誌系統失敗: {e}")
            exit(1)
    
    def _init_nvml(self):
        """初始化 NVIDIA ML"""
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.logger.info(f"成功初始化 NVIDIA ML，偵測到 {self.gpu_count} 個 GPU")
            
            if self.gpu_count == 0:
                raise RuntimeError("未偵測到 NVIDIA GPU")
                
        except Exception as e:
            self.logger.error(f"初始化 NVIDIA ML 失敗: {e}")
            exit(1)
    
    def _check_server_connection(self):
        """檢查服務器連線"""
        try:
            server_host = self.config.get('SERVER', 'server_host')
            server_port = self.config.getint('SERVER', 'server_port')
            connection_timeout = self.config.getint('SERVER', 'connection_timeout', fallback=30)
            
            url = f"{server_host}:{server_port}/health"
            response = requests.get(url, timeout=connection_timeout)
            
            if response.status_code == 200:
                self.logger.info(f"服務器連線正常: {server_host}:{server_port}")
            else:
                self.logger.warning(f"服務器回應異常，狀態碼: {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"無法連線到服務器: {e}")
    
    def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        """收集 GPU 指標"""
        metrics_list = []
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        client_name = self.config.get('AUTHENTICATION', 'client_name')
        
        try:
            for gpu_id in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    metrics = {
                        'timestamp': timestamp,
                        'gpu_id': gpu_id,
                        'gpu_name': gpu_name,
                        'client_name': client_name
                    }
                    
                    # GPU 使用率
                    if self.config.getboolean('GPU_METRICS', 'include_utilization', fallback=True):
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            metrics['utilization_gpu'] = util.gpu
                            metrics['utilization_memory'] = util.memory
                        except Exception as e:
                            self.logger.warning(f"無法取得 GPU {gpu_id} 使用率: {e}")
                            metrics['utilization_gpu'] = None
                            metrics['utilization_memory'] = None
                    
                    # 記憶體資訊
                    if self.config.getboolean('GPU_METRICS', 'include_memory', fallback=True):
                        try:
                            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            metrics['memory_total'] = memory_info.total // (1024 * 1024)  # MB
                            metrics['memory_used'] = memory_info.used // (1024 * 1024)   # MB
                            metrics['memory_free'] = memory_info.free // (1024 * 1024)   # MB
                        except Exception as e:
                            self.logger.warning(f"無法取得 GPU {gpu_id} 記憶體資訊: {e}")
                            metrics['memory_total'] = None
                            metrics['memory_used'] = None
                            metrics['memory_free'] = None
                    
                    # 溫度資訊
                    if self.config.getboolean('GPU_METRICS', 'include_temperature', fallback=True):
                        try:
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            metrics['temperature'] = temperature
                        except Exception as e:
                            self.logger.warning(f"無法取得 GPU {gpu_id} 溫度: {e}")
                            metrics['temperature'] = None
                    
                    # 功耗資訊
                    if self.config.getboolean('GPU_METRICS', 'include_power', fallback=True):
                        try:
                            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0  # mW to W
                            metrics['power_draw'] = round(power_draw, 2)
                            metrics['power_limit'] = round(power_limit, 2)
                        except Exception as e:
                            self.logger.warning(f"無法取得 GPU {gpu_id} 功耗資訊: {e}")
                            metrics['power_draw'] = None
                            metrics['power_limit'] = None
                    
                    # 風扇轉速
                    if self.config.getboolean('GPU_METRICS', 'include_fan_speed', fallback=True):
                        try:
                            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                            metrics['fan_speed'] = fan_speed
                        except Exception as e:
                            self.logger.warning(f"無法取得 GPU {gpu_id} 風扇轉速: {e}")
                            # 當系統無法取得風扇轉速時，預設為0
                            metrics['fan_speed'] = 0
                    
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"收集 GPU {gpu_id} 指標時發生錯誤: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"收集 GPU 指標時發生錯誤: {e}")
            
        return metrics_list
    
    def _send_data_batch(self, data_batch: List[Dict[str, Any]]) -> bool:
        """批次傳送資料到服務器"""
        if not data_batch:
            return True
            
        server_host = self.config.get('SERVER', 'server_host')
        server_port = self.config.getint('SERVER', 'server_port')
        api_endpoint = self.config.get('SERVER', 'api_endpoint')
        connection_timeout = self.config.getint('SERVER', 'connection_timeout', fallback=30)
        read_timeout = self.config.getint('SERVER', 'read_timeout', fallback=60)
        
        max_retries = self.config.getint('NETWORK', 'max_retries', fallback=3)
        retry_delay = self.config.getfloat('NETWORK', 'retry_delay', fallback=5)
        retry_backoff = self.config.getfloat('NETWORK', 'retry_backoff', fallback=2.0)
        
        url = f"{server_host}:{server_port}{api_endpoint}"
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=data_batch,
                    timeout=(connection_timeout, read_timeout),
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if response_data.get("code") == 200:
                            data_info = response_data.get("data", {})
                            received_count = data_info.get("received_count", len(data_batch))
                            failed_count = data_info.get("failed_count", 0)
                            self.logger.info(f"成功傳送資料到服務器:{received_count} 成功,{failed_count} 失敗")
                            return True
                        else:
                            self.logger.warning(f"服務器處理失敗: {response_data.get('message', 'Unknown error')}")
                    except json.JSONDecodeError:
                        self.logger.info(f"成功傳送 {len(data_batch)} 筆資料到服務器")
                        return True
                else:
                    self.logger.warning(f"服務器回應錯誤，狀態碼: {response.status_code}, 內容: {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"傳送資料超時 (嘗試 {attempt + 1}/{max_retries + 1})")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"無法連線到服務器 (嘗試 {attempt + 1}/{max_retries + 1})")
            except Exception as e:
                self.logger.error(f"傳送資料時發生錯誤: {e} (嘗試 {attempt + 1}/{max_retries + 1})")
            
            if attempt < max_retries:
                sleep_time = retry_delay * (retry_backoff ** attempt)
                self.logger.info(f"等待 {sleep_time:.1f} 秒後重試...")
                time.sleep(sleep_time)
        
        self.logger.error(f"傳送資料失敗，已嘗試 {max_retries + 1} 次")
        return False
    
    def _data_sender_worker(self):
        """資料傳送工作線程"""
        batch_size = self.config.getint('NETWORK', 'batch_size', fallback=10)
        batch_interval = self.config.getfloat('NETWORK', 'batch_interval', fallback=30)
        
        data_batch = []
        last_send_time = time.time()
        
        while self.running or not self.data_queue.empty():
            try:
                # 嘗試從佇列取得資料
                try:
                    data = self.data_queue.get(timeout=1.0)
                    data_batch.extend(data)
                    self.data_queue.task_done()
                except Empty:
                    pass
                
                current_time = time.time()
                
                # 檢查是否需要傳送批次
                if (len(data_batch) >= batch_size or 
                    (data_batch and current_time - last_send_time >= batch_interval)):
                    
                    if self._send_data_batch(data_batch):
                        data_batch.clear()
                        last_send_time = current_time
                    else:
                        # 傳送失敗，保留資料下次重試
                        self.logger.warning("資料傳送失敗，將於下次重試")
                        
            except Exception as e:
                self.logger.error(f"資料傳送工作線程發生錯誤: {e}")
                time.sleep(1)
        
        # 傳送剩餘的資料
        if data_batch:
            self._send_data_batch(data_batch)
    
    def start_monitoring(self):
        """開始監控"""
        if self.running:
            self.logger.warning("監控已經在執行中")
            return
        
        self.running = True
        self.logger.info("開始 GPU 指標監控...")
        
        # 啟動資料傳送工作線程
        sender_thread = threading.Thread(target=self._data_sender_worker, daemon=True)
        sender_thread.start()
        
        interval_seconds = self.config.getfloat('MONITORING', 'interval_seconds', fallback=5)
        
        try:
            while self.running:
                start_time = time.time()
                
                # 收集 GPU 指標
                metrics_data = self._collect_gpu_metrics()
                
                if metrics_data:
                    # 將資料放入傳送佇列
                    self.data_queue.put(metrics_data)
                    self.logger.debug(f"收集了 {len(metrics_data)} 筆 GPU 指標資料")
                else:
                    self.logger.warning("未收集到任何 GPU 指標資料")
                
                # 計算下次執行時間
                elapsed_time = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.info("收到中斷信號，正在停止監控...")
        except Exception as e:
            self.logger.error(f"監控過程中發生錯誤: {e}")
        finally:
            self.stop_monitoring()
            
        # 等待資料傳送完成
        self.logger.info("等待剩餘資料傳送完成...")
        self.data_queue.join()
        sender_thread.join(timeout=30)
        
        self.logger.info("GPU 指標監控已停止")
    
    def stop_monitoring(self):
        """停止監控"""
        self.running = False
        self.logger.info("正在停止 GPU 指標監控...")
    
    def __enter__(self):
        """支援 with 語句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支援 with 語句"""
        self.stop_monitoring()
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def main():
    """主函數"""
    try:
        with GPUMetricsClient() as client:
            client.start_monitoring()
    except Exception as e:
        print(f"程式執行發生錯誤: {e}")
        exit(1)


if __name__ == "__main__":
    main()
