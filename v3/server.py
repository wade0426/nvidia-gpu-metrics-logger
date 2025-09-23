#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger - Server 端

一個基於 FastAPI 的 NVIDIA GPU 監控資料 REST API 服務

Author: GPU Metrics Logger Team  
Date: 2025-09-14
"""

import os
import gc
import csv
import json
import logging
import datetime
import configparser
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor

try:
    import pandas as pd
except ImportError:
    print("錯誤: 找不到 pandas 模組。請執行: pip install pandas")
    exit(1)

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("錯誤: 找不到 Flask 相關模組。請執行: pip install flask flask-cors")
    exit(1)

try:
    from marshmallow import Schema, fields, ValidationError, EXCLUDE
except ImportError:
    print("錯誤: 找不到 marshmallow 模組。請執行: pip install marshmallow")
    exit(1)


# ============================================================================
# Marshmallow 資料模型
# ============================================================================

class ReceiveDataSchema(Schema):
    """接收資料請求模型"""
    class Meta:
        unknown = EXCLUDE
    
    timestamp = fields.Str(required=True)
    gpu_id = fields.Int(required=True)
    gpu_name = fields.Str(required=True)
    utilization_gpu = fields.Float(required=True)
    utilization_memory = fields.Float(required=True)
    memory_total = fields.Int(required=True)
    memory_used = fields.Int(required=True)
    memory_free = fields.Int(required=True)
    temperature = fields.Int(required=True)
    power_draw = fields.Float(required=True)
    power_limit = fields.Float(required=True)
    fan_speed = fields.Int(required=True)
    client_name = fields.Str(required=True)


class StatisticsSchema(Schema):
    """統計資料請求模型"""
    class Meta:
        unknown = EXCLUDE
    
    start_date = fields.Str(allow_none=True, missing=None)
    end_date = fields.Str(allow_none=True, missing=None)
    gpu_id = fields.Int(allow_none=True, missing=None)
    client_name = fields.Str(allow_none=True, missing=None)


class HourlyUsageSchema(Schema):
    """每小時使用率請求模型"""
    class Meta:
        unknown = EXCLUDE
    
    start_date = fields.Str(required=True)
    end_date = fields.Str(required=True)
    gpu_id = fields.Int(allow_none=True, missing=None)
    client_name = fields.Str(allow_none=True, missing=None)


class DailyUsageSchema(Schema):
    """每日使用率請求模型"""  
    class Meta:
        unknown = EXCLUDE
    
    start_date = fields.Str(required=True)
    end_date = fields.Str(required=True)
    gpu_id = fields.Int(allow_none=True, missing=None)
    client_name = fields.Str(allow_none=True, missing=None)


class RealtimeSchema(Schema):
    """即時資料請求模型"""
    class Meta:
        unknown = EXCLUDE
    
    gpu_id = fields.Int(allow_none=True, missing=None)
    client_name = fields.Str(allow_none=True, missing=None)


class GPUListSchema(Schema):
    """GPU 清單請求模型"""
    class Meta:
        unknown = EXCLUDE
    
    client_name = fields.Str(allow_none=True, missing=None)


# ============================================================================
# 配置管理類
# ============================================================================

class ServerConfig:
    """伺服器配置管理"""
    
    def __init__(self, config_path: str = "./config/server_config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        """載入配置文件"""
        try:
            if not Path(self.config_path).exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            self.config.read(self.config_path, encoding='utf-8')
            
            # 驗證必要的配置段落
            required_sections = ['SERVER', 'STORAGE', 'LOGGING', 'PERFORMANCE']
            for section in required_sections:
                if not self.config.has_section(section):
                    raise ValueError(f"配置文件缺少必要段落: {section}")
                    
        except Exception as e:
            print(f"載入配置文件失敗: {e}")
            exit(1)
    
    def get(self, section: str, key: str, fallback=None):
        """取得配置值"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback=None):
        """取得整數配置值"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section: str, key: str, fallback=None):
        """取得浮點數配置值"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback=None):
        """取得布林配置值"""
        return self.config.getboolean(section, key, fallback=fallback)


# ============================================================================
# GPU 監控伺服器類
# ============================================================================

class GPUMetricsServer:
    """GPU 監控伺服器主類"""
    
    def __init__(self, config_path: str = "./config/server_config.ini"):
        self.config = ServerConfig(config_path)
        self.logger = None
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 設置日誌
        self._setup_logging()
        
        # 創建輸出目錄
        self._create_directories()
        
        # 啟動背景任務
        self._start_background_tasks()
    
    def _setup_logging(self):
        """設置日誌系統"""
        try:
            log_level = self.config.get('LOGGING', 'log_level', fallback='INFO')
            log_file = self.config.get('LOGGING', 'log_file', fallback='./logs/server.log')
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
            self.logger = logging.getLogger('GPUMetricsServer')
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
                
            self.logger.info("Server 日誌系統初始化完成")
            
        except Exception as e:
            print(f"設置日誌系統失敗: {e}")
            exit(1)
    
    def _create_directories(self):
        """創建必要的目錄"""
        try:
            output_dir = self.config.get('STORAGE', 'output_directory', fallback='./data')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"資料目錄已創建: {output_dir}")
        except Exception as e:
            self.logger.error(f"創建目錄失敗: {e}")
            
    def _start_background_tasks(self):
        """啟動背景任務"""
        # 定期垃圾回收
        gc_interval = self.config.getint('PERFORMANCE', 'gc_interval', fallback=300)
        
        def gc_worker():
            import time
            while True:
                try:
                    time.sleep(gc_interval)
                    gc.collect()
                    self.logger.debug("執行垃圾回收")
                except Exception as e:
                    self.logger.error(f"垃圾回收錯誤: {e}")
        
        threading.Thread(target=gc_worker, daemon=True).start()
        self.logger.info("背景任務已啟動")
    
    def save_to_csv(self, data: Dict[str, Any]) -> bool:
        """儲存資料到 CSV 檔案"""
        try:
            output_dir = self.config.get('STORAGE', 'output_directory', fallback='./data')
            csv_prefix = self.config.get('STORAGE', 'csv_filename_prefix', fallback='gpu_metrics')
            
            # 生成檔案名稱
            today = datetime.datetime.now().strftime('%Y%m%d')
            csv_filename = f"{csv_prefix}_{today}.csv"
            csv_filepath = Path(output_dir) / csv_filename
            
            # 檢查檔案是否存在，決定是否寫入標題行
            write_header = not csv_filepath.exists()
            
            # 寫入 CSV
            with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'gpu_id', 'gpu_name', 'utilization_gpu', 'utilization_memory',
                    'memory_total', 'memory_used', 'memory_free', 'temperature', 'power_draw',
                    'power_limit', 'fan_speed', 'client_name'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                    
                writer.writerow(data)
            
            self.logger.debug(f"資料已儲存到: {csv_filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"儲存 CSV 檔案失敗: {e}")
            return False
    
    def load_csv_data(self) -> Optional[pd.DataFrame]:
        """載入所有 CSV 監控資料"""
        try:
            output_dir = self.config.get('STORAGE', 'output_directory', fallback='./data')
            csv_prefix = self.config.get('STORAGE', 'csv_filename_prefix', fallback='gpu_metrics')
            
            # 搜尋所有 CSV 檔案
            csv_files = list(Path(output_dir).glob(f"{csv_prefix}_*.csv"))
            
            if not csv_files:
                self.logger.warning("找不到任何 CSV 資料檔案")
                return None
            
            # 載入並合併所有檔案
            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"載入檔案失敗 {csv_file}: {e}")
                    continue
            
            if not dfs:
                return None
            
            # 合併所有資料
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 處理時間戳記
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            
            # 新增日期和小時欄位
            combined_df['date'] = combined_df['timestamp'].dt.date
            combined_df['hour'] = combined_df['timestamp'].dt.hour
            
            self.logger.info(f"成功載入 {len(combined_df)} 筆資料，來自 {len(csv_files)} 個檔案")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"載入 CSV 資料失敗: {e}")
            return None
    
    def filter_data(self, df: pd.DataFrame, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, gpu_id: Optional[int] = None,
                   client_name: Optional[str] = None) -> pd.DataFrame:
        """篩選資料"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        filtered_df = df.copy()
        
        # 依日期範圍篩選
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date).date()
                filtered_df = filtered_df[filtered_df['date'] >= start_dt]
            except Exception as e:
                self.logger.warning(f"無效的開始日期: {start_date}, {e}")
        
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date).date()
                filtered_df = filtered_df[filtered_df['date'] <= end_dt]
            except Exception as e:
                self.logger.warning(f"無效的結束日期: {end_date}, {e}")
        
        # 依 GPU ID 篩選
        if gpu_id is not None:
            filtered_df = filtered_df[filtered_df['gpu_id'] == gpu_id]
        
        # 依客戶端名稱篩選
        if client_name:
            filtered_df = filtered_df[filtered_df['client_name'] == client_name]
        
        self.logger.debug(f"篩選後資料筆數: {len(filtered_df)}")
        return filtered_df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """計算統計數據"""
        if df is None or df.empty:
            return {
                "hourly_average": 0.0,
                "daily_average": 0.0,
                "period_average": 0.0,
                "max_utilization": 0.0,
                "min_utilization": 0.0
            }
        
        try:
            # 基本統計
            period_avg = float(df['utilization_gpu'].mean())
            max_util = float(df['utilization_gpu'].max())
            min_util = float(df['utilization_gpu'].min())
            
            # 每小時平均
            hourly_stats = df.groupby(['date', 'hour'])['utilization_gpu'].mean()
            hourly_avg = float(hourly_stats.mean()) if not hourly_stats.empty else 0.0
            
            # 每日平均
            daily_stats = df.groupby('date')['utilization_gpu'].mean()
            daily_avg = float(daily_stats.mean()) if not daily_stats.empty else 0.0
            
            return {
                "hourly_average": round(hourly_avg, 2),
                "daily_average": round(daily_avg, 2),
                "period_average": round(period_avg, 2),
                "max_utilization": round(max_util, 2),
                "min_utilization": round(min_util, 2)
            }
            
        except Exception as e:
            self.logger.error(f"計算統計數據失敗: {e}")
            return {
                "hourly_average": 0.0,
                "daily_average": 0.0,
                "period_average": 0.0,
                "max_utilization": 0.0,
                "min_utilization": 0.0
            }
    
    def get_hourly_usage(self, df: pd.DataFrame, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """取得每小時使用率資料"""
        try:
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            
            # 生成完整的日期時間範圍
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            full_hours = []
            
            for date in date_range:
                for hour in range(24):
                    full_hours.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'hour': hour,
                        'utilization': 0.0
                    })
            
            # 如果有資料，計算實際使用率
            if df is not None and not df.empty:
                hourly_stats = df.groupby(['date', 'hour'])['utilization_gpu'].mean().reset_index()
                
                # 更新實際資料
                for _, row in hourly_stats.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    hour = int(row['hour'])
                    utilization = round(float(row['utilization_gpu']), 2)
                    
                    # 找到對應的時間點並更新
                    for item in full_hours:
                        if item['date'] == date_str and item['hour'] == hour:
                            item['utilization'] = utilization
                            break
            
            return full_hours
            
        except Exception as e:
            self.logger.error(f"取得每小時使用率失敗: {e}")
            return []
    
    def get_daily_usage(self, df: pd.DataFrame, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """取得每日使用率資料"""
        try:
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            
            # 生成完整的日期範圍
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            daily_data = []
            
            for date in date_range:
                daily_data.append({
                    'date': date.strftime('%m/%d'),
                    'min_utilization': 0.0,
                    'max_utilization': 0.0
                })
            
            # 如果有資料，計算實際值
            if df is not None and not df.empty:
                daily_stats = df.groupby('date')['utilization_gpu'].agg(['min', 'max']).reset_index()
                
                # 更新實際資料
                for _, row in daily_stats.iterrows():
                    date_str = row['date'].strftime('%m/%d')
                    min_util = round(float(row['min']), 2)
                    max_util = round(float(row['max']), 2)
                    
                    # 找到對應的日期並更新
                    for item in daily_data:
                        if item['date'] == date_str:
                            item['min_utilization'] = min_util
                            item['max_utilization'] = max_util
                            break
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"取得每日使用率失敗: {e}")
            return []
    
    def get_gpu_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """取得 GPU 清單"""
        try:
            if df is None or df.empty:
                return []
            
            # 取得最新資料
            latest_data = df.sort_values('timestamp').groupby(['gpu_id', 'client_name']).tail(1)
            
            gpu_list = []
            for _, row in latest_data.iterrows():
                gpu_info = {
                    'gpu_id': int(row['gpu_id']),
                    'gpu_name': str(row['gpu_name']),
                    'client_name': str(row['client_name']),
                    'status': 'active',
                    'last_update': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                }
                gpu_list.append(gpu_info)
            
            return sorted(gpu_list, key=lambda x: (x['client_name'], x['gpu_id']))
            
        except Exception as e:
            self.logger.error(f"取得 GPU 清單失敗: {e}")
            return []
    
    def get_realtime_data(self, df: pd.DataFrame, gpu_id: Optional[int] = None,
                         client_name: Optional[str] = None) -> Dict[str, Any]:
        """取得即時資料"""
        try:
            if df is None or df.empty:
                return {
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "type": "average",
                    "scope": "no_data",
                    "utilization_gpu": 0.0,
                    "temperature": 0,
                    "memory_used_percent": 0.0,
                    "power_draw": 0.0,
                    "gpu_count": 0,
                    "client_count": 0
                }
            
            # 取得最新資料（最近5分鐘內的資料）
            latest_time = df['timestamp'].max()
            recent_time = latest_time - pd.Timedelta(minutes=5)
            recent_df = df[df['timestamp'] >= recent_time]
            
            # 根據篩選條件處理
            filtered_df = self.filter_data(recent_df, gpu_id=gpu_id, client_name=client_name)
            
            if filtered_df.empty:
                scope = "no_recent_data"
                gpu_count = 0
                client_count = 0
                avg_data = {
                    "utilization_gpu": 0.0,
                    "temperature": 0,
                    "memory_used_percent": 0.0,
                    "power_draw": 0.0
                }
            else:
                # 計算平均值
                avg_data = {
                    "utilization_gpu": round(float(filtered_df['utilization_gpu'].mean()), 2),
                    "temperature": round(float(filtered_df['temperature'].mean())),
                    "memory_used_percent": round(float(filtered_df['utilization_memory'].mean()), 2),
                    "power_draw": round(float(filtered_df['power_draw'].mean()), 2)
                }
                
                gpu_count = len(filtered_df.groupby(['client_name', 'gpu_id']))
                client_count = len(filtered_df['client_name'].unique())
                
                # 判斷篩選範圍
                if gpu_id is not None and client_name is not None:
                    scope = "single_client_single_gpu"
                elif client_name is not None:
                    scope = "single_client_all_gpus"
                elif gpu_id is not None:
                    scope = "all_clients_single_gpu"
                else:
                    scope = "all_clients_all_gpus"
            
            return {
                "timestamp": latest_time.strftime('%Y-%m-%d %H:%M:%S'),
                "type": "average",
                "scope": scope,
                **avg_data,
                "gpu_count": gpu_count,
                "client_count": client_count
            }
            
        except Exception as e:
            self.logger.error(f"取得即時資料失敗: {e}")
            return {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "type": "average",
                "scope": "error",
                "utilization_gpu": 0.0,
                "temperature": 0,
                "memory_used_percent": 0.0,
                "power_draw": 0.0,
                "gpu_count": 0,
                "client_count": 0
            }


# 創建全域伺服器實例
server_instance = GPUMetricsServer()

# 創建 Flask 應用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支援中文字符

# 設置 CORS
CORS(app, supports_credentials=True)


# ============================================================================
# 幫助函數
# ============================================================================

def validate_request_data(schema_class):
    """驗證請求資料的裝飾器"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                if request.method == 'POST':
                    data = request.get_json()
                    if data is None:
                        data = {}
                else:
                    data = {}
                
                schema = schema_class()
                validated_data = schema.load(data)
                return f(validated_data, *args, **kwargs)
            except ValidationError as e:
                server_instance.logger.warning(f"資料驗證錯誤: {e.messages}")
                return jsonify({
                    "success": False,
                    "detail": "資料格式驗證失敗",
                    "error_code": "VALIDATION_ERROR",
                    "errors": e.messages
                }), 422
            except Exception as e:
                server_instance.logger.error(f"處理請求時發生錯誤: {e}")
                return jsonify({
                    "success": False,
                    "detail": "伺服器內部錯誤",
                    "error_code": "INTERNAL_SERVER_ERROR"
                }), 500
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


# ============================================================================
# API 端點
# ============================================================================

@app.route("/", methods=["GET"])
def root():
    """根路由"""
    return jsonify({
        "message": "NVIDIA GPU Metrics API",
        "version": "1.0.0",
        "docs_url": "/docs"
    })


@app.route("/health", methods=["GET"])
def health_check():
    """健康檢查端點"""
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()})


@app.route("/api/receive-data", methods=["POST"])
def receive_gpu_data():
    """接收 GPU 監控資料（支援單一或批次）"""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({
                "success": False,
                "detail": "請求資料為空",
                "error_code": "EMPTY_REQUEST"
            }), 400
        
        # 統一處理為列表格式
        if isinstance(data, list):
            data_list = data
        else:
            data_list = [data]
        
        server_instance.logger.info(f"接收 {len(data_list)} 筆 GPU 監控資料")
        
        success_count = 0
        failed_count = 0
        schema = ReceiveDataSchema()
        
        # 批次處理每筆資料
        for data_item in data_list:
            try:
                # 驗證資料格式
                validated_data = schema.load(data_item)
                
                # 儲存到 CSV
                if server_instance.save_to_csv(validated_data):
                    success_count += 1
                else:
                    failed_count += 1
                    
            except ValidationError as e:
                server_instance.logger.warning(f"資料驗證失敗: {e.messages}")
                failed_count += 1
                continue
            except Exception as e:
                server_instance.logger.warning(f"處理單筆資料失敗: {e}")
                failed_count += 1
                continue
        
        if success_count > 0:
            return jsonify({
                "success": True,
                "message": f"Batch data received: {success_count} successful, {failed_count} failed",
                "received_count": success_count,
                "failed_count": failed_count,
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({
                "success": False,
                "detail": "All data failed to save",
                "error_code": "DATA_SAVE_FAILED"
            }), 500
            
    except Exception as e:
        server_instance.logger.error(f"接收批次資料時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/gpu/statistics", methods=["POST"])
@validate_request_data(StatisticsSchema)
def get_gpu_statistics(validated_data):
    """取得 GPU 統計數據"""
    try:
        server_instance.logger.info(f"取得統計數據請求: {validated_data}")
        
        # 載入資料
        df = server_instance.load_csv_data()
        
        # 篩選資料
        filtered_df = server_instance.filter_data(
            df, 
            start_date=validated_data.get('start_date'),
            end_date=validated_data.get('end_date'),
            gpu_id=validated_data.get('gpu_id'),
            client_name=validated_data.get('client_name')
        )
        
        # 計算統計
        stats = server_instance.calculate_statistics(filtered_df)
        
        response_data = {
            **stats,
            "period": {
                "start_date": validated_data.get('start_date'),
                "end_date": validated_data.get('end_date')
            },
            "gpu_id": validated_data.get('gpu_id'),
            "client_name": validated_data.get('client_name')
        }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        server_instance.logger.error(f"取得統計數據時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/gpu/hourly-usage", methods=["POST"])
@validate_request_data(HourlyUsageSchema)
def get_hourly_usage(validated_data):
    """取得每小時使用率"""
    try:
        server_instance.logger.info(f"取得每小時使用率請求: {validated_data}")
        
        # 載入資料
        df = server_instance.load_csv_data()
        
        # 篩選資料
        filtered_df = server_instance.filter_data(
            df,
            start_date=validated_data.get('start_date'),
            end_date=validated_data.get('end_date'),
            gpu_id=validated_data.get('gpu_id'),
            client_name=validated_data.get('client_name')
        )
        
        # 取得每小時資料
        hourly_data = server_instance.get_hourly_usage(
            filtered_df, 
            validated_data.get('start_date'), 
            validated_data.get('end_date')
        )
        
        response_data = {
            "chart_data": hourly_data,
            "period": {
                "start_date": validated_data.get('start_date'),
                "end_date": validated_data.get('end_date')
            },
            "gpu_id": validated_data.get('gpu_id'),
            "client_name": validated_data.get('client_name')
        }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        server_instance.logger.error(f"取得每小時使用率時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/gpu/daily-usage", methods=["POST"])
@validate_request_data(DailyUsageSchema)
def get_daily_usage(validated_data):
    """取得每日使用率"""
    try:
        server_instance.logger.info(f"取得每日使用率請求: {validated_data}")
        
        # 載入資料
        df = server_instance.load_csv_data()
        
        # 篩選資料
        filtered_df = server_instance.filter_data(
            df,
            start_date=validated_data.get('start_date'),
            end_date=validated_data.get('end_date'),
            gpu_id=validated_data.get('gpu_id'),
            client_name=validated_data.get('client_name')
        )
        
        # 取得每日資料
        daily_data = server_instance.get_daily_usage(
            filtered_df,
            validated_data.get('start_date'),
            validated_data.get('end_date')
        )
        
        response_data = {
            "chart_data": daily_data,
            "period": {
                "start_date": validated_data.get('start_date'),
                "end_date": validated_data.get('end_date')
            },
            "gpu_id": validated_data.get('gpu_id'),
            "client_name": validated_data.get('client_name')
        }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        server_instance.logger.error(f"取得每日使用率時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/gpu/list", methods=["POST"])
@validate_request_data(GPUListSchema)
def get_gpu_list(validated_data):
    """取得 GPU 清單"""
    try:
        server_instance.logger.info(f"取得 GPU 清單請求: {validated_data}")
        
        # 載入資料
        df = server_instance.load_csv_data()
        
        # 篩選資料
        filtered_df = server_instance.filter_data(df, client_name=validated_data.get('client_name'))
        
        # 取得 GPU 清單
        gpu_list = server_instance.get_gpu_list(filtered_df)
        
        return jsonify({
            "success": True,
            "data": gpu_list
        })
        
    except Exception as e:
        server_instance.logger.error(f"取得 GPU 清單時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/gpu/realtime", methods=["POST"])
@validate_request_data(RealtimeSchema)
def get_realtime_data(validated_data):
    """取得即時資料"""
    try:
        server_instance.logger.info(f"取得即時資料請求: {validated_data}")
        
        # 載入資料
        df = server_instance.load_csv_data()
        
        # 取得即時資料
        realtime_data = server_instance.get_realtime_data(
            df,
            gpu_id=validated_data.get('gpu_id'),
            client_name=validated_data.get('client_name')
        )
        
        return jsonify({
            "success": True,
            "data": realtime_data
        })
        
    except Exception as e:
        server_instance.logger.error(f"取得即時資料時發生錯誤: {e}")
        return jsonify({
            "success": False,
            "detail": f"Internal server error: {str(e)}",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500


# ============================================================================
# 錯誤處理
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """400 錯誤處理器"""
    server_instance.logger.warning(f"400 錯誤: {error}")
    return jsonify({
        "success": False,
        "detail": "請求格式錯誤",
        "error_code": "BAD_REQUEST"
    }), 400


@app.errorhandler(404)
def not_found(error):
    """404 錯誤處理器"""
    server_instance.logger.warning(f"404 錯誤: {error}")
    return jsonify({
        "success": False,
        "detail": "找不到請求的資源",
        "error_code": "NOT_FOUND"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 錯誤處理器"""
    server_instance.logger.error(f"500 錯誤: {error}")
    return jsonify({
        "success": False,
        "detail": "伺服器內部錯誤",
        "error_code": "INTERNAL_SERVER_ERROR"
    }), 500


# ============================================================================
# 主函數
# ============================================================================

def main():
    """主函數"""
    try:
        host = server_instance.config.get('SERVER', 'host', fallback='0.0.0.0')
        port = server_instance.config.getint('SERVER', 'port', fallback=5000)
        debug = server_instance.config.getboolean('SERVER', 'debug', fallback=False)
        
        server_instance.logger.info(f"啟動 GPU Metrics Server 在 {host}:{port}")
        server_instance.logger.info(f"訪問 API: http://{host}:{port}/")
        
        app.run(
            host=host,
            port=port,
            debug=debug
        )
        
    except Exception as e:
        print(f"啟動伺服器失敗: {e}")
        exit(1)


if __name__ == "__main__":
    main()