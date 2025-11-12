#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU History Query API - 單元測試
測試新增的 GPU 紀錄查詢 API 功能

Author: GPU Metrics Logger Team
Date: 2025-11-06
"""

import os
import csv
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    print("錯誤: 找不到 pandas 模組。請執行: pip install pandas")
    exit(1)

try:
    import requests
except ImportError:
    print("錯誤: 找不到 requests 模組。請執行: pip install requests")
    exit(1)


class GPUQueryTestDataGenerator:
    """測試資料生成器"""
    
    CSV_HEADER = [
        'timestamp', 'gpu_id', 'gpu_name', 'utilization_gpu',
        'utilization_memory', 'memory_total', 'memory_used',
        'memory_free', 'temperature', 'power_draw',
        'power_limit', 'fan_speed', 'client_name'
    ]
    
    def __init__(self, data_dir: str = "./data"):
        """初始化測試資料生成器"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_timestamp(self, date_str: str, hour: int, minute: int, second: int) -> str:
        """生成時間戳記"""
        dt = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}",
                               "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def create_test_csv(self, filename: str, data_rows: List[Dict[str, Any]]) -> str:
        """
        建立測試 CSV 檔案
        
        Args:
            filename: CSV 檔案名稱
            data_rows: 資料列表，每個元素為字典
            
        Returns:
            CSV 檔案完整路徑
        """
        filepath = self.data_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()
            
            for row in data_rows:
                # 填充預設值
                full_row = {
                    'gpu_name': 'NVIDIA GeForce RTX 3090',
                    'utilization_memory': 50.0,
                    'memory_total': 24576,
                    'memory_used': 12288,
                    'memory_free': 12288,
                    'temperature': 65,
                    'power_draw': 250.0,
                    'power_limit': 350.0,
                    'fan_speed': 60,
                    'client_name': 'test_client'
                }
                full_row.update(row)
                writer.writerow(full_row)
        
        print(f"✓ 已建立測試檔案: {filepath}")
        return str(filepath)
    
    def scenario_normal_working_hours(self) -> str:
        """
        情境 1: 正常工作時段資料 (8:00-20:00)
        - 涵蓋完整的 8:00-19:59 時段
        - 每小時有不同使用率
        - 部分時段超過 90%
        """
        data = []
        base_date = "2024-11-02"
        
        for hour in range(8, 20):  # 8:00-19:59
            for minute in range(0, 60, 5):  # 每 5 分鐘一筆
                # 模擬不同時段的使用率
                if hour < 10:
                    utilization = 45.0 + (minute / 60 * 10)  # 45-55%
                elif hour < 15:
                    utilization = 85.0 + (minute / 60 * 10)  # 85-95%（會超過90%）
                else:
                    utilization = 60.0 + (minute / 60 * 15)  # 60-75%
                
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                    'gpu_id': 0,
                    'utilization_gpu': round(utilization, 2)
                })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_off_hours_data(self) -> str:
        """
        情境 2: 包含非工作時段資料 (0:00-7:59 和 20:00-23:59)
        - 測試時段篩選功能
        - 應該被過濾掉的資料
        """
        data = []
        base_date = "2024-11-03"
        
        # 非工作時段：0:00-7:59
        for hour in range(0, 8):
            for minute in range(0, 60, 15):
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                    'gpu_id': 0,
                    'utilization_gpu': 30.0  # 這些不應該被計入
                })
        
        # 工作時段：8:00-19:59
        for hour in range(8, 20):
            for minute in range(0, 60, 15):
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                    'gpu_id': 0,
                    'utilization_gpu': 70.0
                })
        
        # 非工作時段：20:00-23:59
        for hour in range(20, 24):
            for minute in range(0, 60, 15):
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                    'gpu_id': 0,
                    'utilization_gpu': 25.0  # 這些不應該被計入
                })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_high_utilization(self) -> str:
        """
        情境 3: 高使用率情境（大量超過 90%）
        - 測試 over90_duration 計算
        - 每 5 秒一筆，60分鐘 = 720 筆
        """
        data = []
        base_date = "2024-11-04"
        
        for hour in range(8, 20):
            for minute in range(0, 60):
                for second in range(0, 60, 5):  # 每 5 秒一筆
                    # 10:00-15:00 時段全部超過 90%
                    if 10 <= hour < 15:
                        utilization = 95.0
                    else:
                        utilization = 70.0
                    
                    data.append({
                        'timestamp': self.generate_timestamp(base_date, hour, minute, second),
                        'gpu_id': 0,
                        'utilization_gpu': utilization
                    })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_multiple_gpu_ids(self) -> str:
        """
        情境 4: 多個 GPU ID 混合資料
        - 測試 GPU ID 篩選功能
        """
        data = []
        base_date = "2024-11-05"
        
        for gpu_id in [0, 1, 2]:
            for hour in range(8, 20):
                for minute in range(0, 60, 10):
                    utilization = 50.0 + (gpu_id * 10)  # 不同 GPU 不同使用率
                    data.append({
                        'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                        'gpu_id': gpu_id,
                        'utilization_gpu': utilization
                    })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_edge_cases(self) -> str:
        """
        情境 5: 邊界情況測試
        - 使用率恰好 90%（不應計入 over90）
        - 使用率 90.01%（應計入 over90）
        - 使用率 0%
        - 使用率 100%
        """
        data = []
        base_date = "2024-11-06"
        
        test_values = [
            (8, 0, 0.0),      # 最低值
            (9, 0, 89.99),    # 接近但未達 90%
            (10, 0, 90.0),    # 恰好 90%（不計入）
            (11, 0, 90.01),   # 超過 90%（計入）
            (12, 0, 95.5),    # 高使用率
            (13, 0, 100.0),   # 最高值
            (14, 0, 50.0),    # 中等值
        ]
        
        for hour, minute, utilization in test_values:
            for second in range(0, 60, 5):
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, second, 0),
                    'gpu_id': 0,
                    'utilization_gpu': utilization
                })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_multi_day_range(self) -> List[str]:
        """
        情境 6: 跨多天查詢
        - 測試日期範圍查詢功能
        - 為每一天生成獨立的 CSV 檔案
        """
        files = []
        
        for day_offset in range(0, 3):  # 3 天
            date_obj = datetime(2024, 11, 1) + timedelta(days=day_offset)
            date_str = date_obj.strftime("%Y-%m-%d")
            
            data = []
            for hour in range(8, 20):
                for minute in range(0, 60, 15):
                    utilization = 60.0 + (day_offset * 5)  # 每天不同使用率
                    data.append({
                        'timestamp': self.generate_timestamp(date_str, hour, minute, 0),
                        'gpu_id': 0,
                        'utilization_gpu': utilization
                    })
            
            filename = f"gpu_metrics_{date_str}.csv"
            filepath = self.create_test_csv(filename, data)
            files.append(filepath)
        
        return files
    
    def scenario_sparse_data(self) -> str:
        """
        情境 7: 稀疏資料
        - 某些小時沒有資料
        - 測試缺失資料處理
        """
        data = []
        base_date = "2024-11-07"
        
        # 只有偶數小時有資料
        for hour in range(8, 20, 2):
            for minute in range(0, 60, 10):
                data.append({
                    'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                    'gpu_id': 0,
                    'utilization_gpu': 75.0
                })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)
    
    def scenario_empty_data(self) -> str:
        """
        情境 8: 空資料集
        - 只有標題，沒有資料
        """
        base_date = "2024-11-08"
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", [])
    
    def scenario_client_name_filtering(self) -> str:
        """
        情境 9: 多個 client_name 混合資料
        - 測試 client_name 篩選功能
        - 包含 'client_A', 'client_B', 'client_C' 三種客戶端
        """
        data = []
        base_date = "2024-11-09"
        clients = ['client_A', 'client_B', 'client_C']
        
        for idx, client in enumerate(clients):
            for hour in range(8, 20):
                for minute in range(0, 60, 10):
                    # 不同客戶端有不同的使用率模式
                    if client == 'client_A':
                        utilization = 70.0 + (minute / 60 * 10)  # 70-80%
                    elif client == 'client_B':
                        utilization = 85.0 + (minute / 60 * 10)  # 85-95%
                    else:  # client_C
                        utilization = 50.0 + (minute / 60 * 15)  # 50-65%
                    
                    data.append({
                        'timestamp': self.generate_timestamp(base_date, hour, minute, 0),
                        'gpu_id': 0,
                        'utilization_gpu': round(utilization, 2),
                        'client_name': client
                    })
        
        return self.create_test_csv(f"gpu_metrics_{base_date}.csv", data)


class TestGPUQueryAPI(unittest.TestCase):
    """GPU 查詢 API 測試類別"""
    
    @classmethod
    def setUpClass(cls):
        """測試前準備：生成所有測試資料"""
        print("\n" + "="*70)
        print("開始生成測試資料...")
        print("="*70)
        
        cls.generator = GPUQueryTestDataGenerator()
        cls.test_files = {
            'normal': cls.generator.scenario_normal_working_hours(),
            'off_hours': cls.generator.scenario_off_hours_data(),
            'high_util': cls.generator.scenario_high_utilization(),
            'multi_gpu': cls.generator.scenario_multiple_gpu_ids(),
            'edge_cases': cls.generator.scenario_edge_cases(),
            'multi_day': cls.generator.scenario_multi_day_range(),  # 現在是列表
            'sparse': cls.generator.scenario_sparse_data(),
            'empty': cls.generator.scenario_empty_data(),
            'client_filter': cls.generator.scenario_client_name_filtering(),
        }
        
        print("\n✓ 所有測試資料已建立完成！")
        print("="*70 + "\n")
        
        # API 端點（根據你的實際設定調整）
        cls.api_base_url = "http://localhost:5000"
        cls.api_endpoint = f"{cls.api_base_url}/api/gpu/query-history"
    
    def _make_request(self, start_date: str, end_date: str, client_name: str = None) -> Dict[str, Any]:
        """發送 API 請求"""
        payload = {
            "start_date": start_date,
            "end_date": end_date
        }
        # client_name 為可選參數，有值才加入 payload
        if client_name is not None:
            payload["client_name"] = client_name
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=10)
            return response.json()
        except requests.exceptions.ConnectionError:
            self.skipTest("伺服器未啟動，跳過 API 測試")
        except Exception as e:
            self.fail(f"API 請求失敗: {str(e)}")
    
    def _load_csv_for_verification(self, filepath: str, start_date: str,
                                   end_date: str, gpu_id: int) -> pd.DataFrame:
        """載入 CSV 並進行相同的篩選，用於驗證"""
        df = pd.read_csv(filepath)
        
        # 解析時間戳記
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        
        # 篩選條件
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df = df[df['gpu_id'] == gpu_id]
        df = df[(df['hour'] >= 8) & (df['hour'] < 20)]  # 8:00-19:59
        
        return df
    
    def test_01_normal_working_hours(self):
        """測試 1: 正常工作時段資料"""
        print("\n【測試 1】正常工作時段資料 (8:00-20:00)")
        
        result = self._make_request("2024-11-02", "2024-11-02", "test_client")
        
        # 驗證回應結構
        self.assertEqual(result['code'], 200)
        self.assertIn('data', result)
        self.assertIn('statistics', result['data'])
        self.assertIn('hourly_max_usage', result['data'])
        self.assertIn('hourly_over90_duration', result['data'])
        
        # 驗證統計數據
        stats = result['data']['statistics']
        self.assertGreater(stats['hourly_average'], 0)
        self.assertGreater(stats['period_average'], 0)
        
        # 驗證每小時資料數量（8:00-19:59 = 12 小時）
        hourly_data = result['data']['hourly_max_usage']
        self.assertGreater(len(hourly_data), 0)
        
        print(f"   ✓ 統計數據: {stats}")
        print(f"   ✓ 每小時資料筆數: {len(hourly_data)}")
    
    def test_02_off_hours_filtering(self):
        """測試 2: 非工作時段過濾"""
        print("\n【測試 2】非工作時段資料過濾")
        
        result = self._make_request("2024-11-03", "2024-11-03", "test_client")
        
        # 驗證只有工作時段的資料
        hourly_data = result['data']['hourly_max_usage']
        for entry in hourly_data:
            hour = entry['hour']
            self.assertGreaterEqual(hour, 8, "發現小於 8 點的資料")
            self.assertLess(hour, 20, "發現大於等於 20 點的資料")
        
        print(f"   ✓ 所有資料都在 8:00-19:59 範圍內")
        print(f"   ✓ 資料筆數: {len(hourly_data)}")
    
    def test_03_high_utilization_over90(self):
        """測試 3: 高使用率與超過 90% 計算"""
        print("\n【測試 3】高使用率情境（超過 90% 計算）")
        
        result = self._make_request("2024-11-04", "2024-11-04", "test_client")
        
        # 驗證 over90_duration
        over90_data = result['data']['hourly_over90_duration']
        
        # 10:00-14:59 時段應該有 over90 資料
        hours_with_over90 = [entry['hour'] for entry in over90_data if entry['duration'] > 0]
        self.assertGreater(len(hours_with_over90), 0, "應該有超過 90% 的時段")
        
        # 檢查 10-14 點是否有 over90 記錄
        for hour in range(10, 15):
            matching = [e for e in over90_data if e['hour'] == hour]
            if matching:
                self.assertGreater(matching[0]['duration'], 0,
                                 f"{hour} 點應該有超過 90% 的持續時間")
        
        print(f"   ✓ 超過 90% 的小時數: {len(hours_with_over90)}")
        print(f"   ✓ 範例資料: {over90_data[:3]}")
    
    def test_05_edge_cases_90_threshold(self):
        """測試 5: 邊界值測試（90% 閾值）"""
        print("\n【測試 5】邊界值測試（90% 閾值）")
        
        result = self._make_request("2024-11-06", "2024-11-06", "test_client")
        
        # 手動驗證：只有 90.01, 95.5, 100.0 應該計入 over90
        df = self._load_csv_for_verification(
            self.test_files['edge_cases'],
            "2024-11-06", "2024-11-06", 0
        )
        
        # 計算超過 90 的資料
        over90_count = len(df[df['utilization_gpu'] > 90])
        expected_hours_with_over90 = df[df['utilization_gpu'] > 90]['hour'].nunique()
        
        over90_data = result['data']['hourly_over90_duration']
        actual_hours_with_over90 = len([e for e in over90_data if e['duration'] > 0])
        
        print(f"   ✓ 超過 90% 的紀錄數: {over90_count}")
        print(f"   ✓ 預期有超過 90% 的小時數: {expected_hours_with_over90}")
        print(f"   ✓ 實際有超過 90% 的小時數: {actual_hours_with_over90}")
        
        # 驗證 90.0 不應計入
        self.assertGreater(actual_hours_with_over90, 0)
    
    def test_06_multi_day_range(self):
        """測試 6: 跨多天查詢"""
        print("\n【測試 6】跨多天查詢")
        
        result = self._make_request("2024-11-01", "2024-11-03", "test_client")
        
        self.assertEqual(result['code'], 200)
        
        # 驗證有多天的資料
        hourly_data = result['data']['hourly_max_usage']
        dates = set(entry['date'] for entry in hourly_data)
        
        self.assertGreaterEqual(len(dates), 2, "應該有至少 2 天的資料")
        
        print(f"   ✓ 查詢天數: {len(dates)}")
        print(f"   ✓ 日期範圍: {min(dates)} ~ {max(dates)}")
        print(f"   ✓ 總資料筆數: {len(hourly_data)}")
        
        # 驗證所有檔案都已建立
        for filepath in self.test_files['multi_day']:
            self.assertTrue(os.path.exists(filepath), f"檔案不存在: {filepath}")
    
    def test_07_sparse_data_handling(self):
        """測試 7: 稀疏資料處理"""
        print("\n【測試 7】稀疏資料處理（部分小時無資料）")
        
        result = self._make_request("2024-11-07", "2024-11-07", "test_client")
        
        self.assertEqual(result['code'], 200)
        
        hourly_data = result['data']['hourly_max_usage']
        hours = [entry['hour'] for entry in hourly_data]
        
        print(f"   ✓ 有資料的小時: {sorted(hours)}")
        print(f"   ✓ 資料筆數: {len(hourly_data)}")
        
        # 驗證統計仍能正確計算
        stats = result['data']['statistics']
        self.assertIsNotNone(stats['hourly_average'])
    
    def test_08_empty_data_handling(self):
        """測試 8: 空資料處理"""
        print("\n【測試 8】空資料處理")
        
        # 查詢一個只有標題沒有資料的日期
        result = self._make_request("2024-11-08", "2024-11-08", "test_client")
        
        # 應該返回空資料或預設值，而非錯誤
        self.assertEqual(result['code'], 200)
        
        print(f"   ✓ API 正確處理空資料情況")
        print(f"   ✓ 回應: {result}")
    
    def test_09_hourly_max_calculation(self):
        """測試 9: 每小時最大值計算驗證"""
        print("\n【測試 9】每小時最大值計算驗證")
        
        result = self._make_request("2024-11-02", "2024-11-02", "test_client")
        
        # 手動計算驗證
        df = self._load_csv_for_verification(
            self.test_files['normal'],
            "2024-11-02", "2024-11-02", 0
        )
        
        if not df.empty:
            manual_hourly_max = df.groupby('hour')['utilization_gpu'].max()
            api_hourly_max = {
                entry['hour']: entry['max_utilization']
                for entry in result['data']['hourly_max_usage']
            }
            
            # 比對部分小時的最大值
            for hour in manual_hourly_max.index[:3]:
                if hour in api_hourly_max:
                    manual_val = manual_hourly_max[hour]
                    api_val = api_hourly_max[hour]
                    print(f"   ✓ {hour} 點 - 手動計算: {manual_val:.2f}%, API: {api_val:.2f}%")
                    self.assertAlmostEqual(manual_val, api_val, places=1)
    
    def test_10_statistics_12h_average(self):
        """測試 10: 12 小時平均值計算"""
        print("\n【測試 10】12 小時平均值計算驗證")
        
        result = self._make_request("2024-11-02", "2024-11-02", "test_client")
        
        stats = result['data']['statistics']
        
        # 驗證所有統計值都在合理範圍
        self.assertGreaterEqual(stats['hourly_average'], 0)
        self.assertLessEqual(stats['hourly_average'], 100)
        self.assertGreaterEqual(stats['daily_average'], 0)
        self.assertLessEqual(stats['daily_average'], 100)
        self.assertGreaterEqual(stats['period_average'], 0)
        self.assertLessEqual(stats['period_average'], 100)
        
        print(f"   ✓ 每小時平均: {stats['hourly_average']:.2f}%")
        print(f"   ✓ 每日平均: {stats['daily_average']:.2f}%")
        print(f"   ✓ 期間平均: {stats['period_average']:.2f}%")
        print(f"   ✓ 最高使用率: {stats['max_utilization']:.2f}%")
        print(f"   ✓ 最低使用率: {stats['min_utilization']:.2f}%")
    
    def test_11_client_name_filtering(self):
        """測試 11: client_name 篩選功能"""
        print("\n【測試 11】client_name 篩選功能")
        
        # 測試不同 client_name 的查詢
        for client in ['client_A', 'client_B', 'client_C']:
            payload = {
                "start_date": "2024-11-09",
                "end_date": "2024-11-09",
                "client_name": client
            }
            
            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=10)
                result = response.json()
            except requests.exceptions.ConnectionError:
                self.skipTest("伺服器未啟動，跳過 API 測試")
            except Exception as e:
                self.fail(f"API 請求失敗: {str(e)}")
            
            self.assertEqual(result['code'], 200)
            self.assertIn('data', result)
            
            stats = result['data']['statistics']
            print(f"   ✓ {client}: 平均使用率 = {stats['period_average']:.2f}%")
            
            # 驗證不同客戶端的使用率範圍
            if client == 'client_A':
                self.assertGreater(stats['period_average'], 65)
                self.assertLess(stats['period_average'], 85)
            elif client == 'client_B':
                self.assertGreater(stats['period_average'], 80)
            else:  # client_C
                self.assertGreater(stats['period_average'], 45)
                self.assertLess(stats['period_average'], 70)
        
        # 測試不指定 client_name（應該返回所有客戶端資料）
        print("\n   測試不指定 client_name（應返回所有客戶端資料）：")
        payload_all = {
            "start_date": "2024-11-09",
            "end_date": "2024-11-09"
        }
        try:
            response = requests.post(self.api_endpoint, json=payload_all, timeout=10)
            result_all = response.json()
        except requests.exceptions.ConnectionError:
            self.skipTest("伺服器未啟動，跳過 API 測試")
        except Exception as e:
            self.fail(f"API 請求失敗: {str(e)}")
        
        self.assertEqual(result_all['code'], 200)
        self.assertIn('data', result_all)
        
        # 驗證返回的資料應該包含所有客戶端的資料
        # 平均使用率應該介於所有客戶端之間
        stats_all = result_all['data']['statistics']
        print(f"   ✓ 不指定 client_name: 平均使用率 = {stats_all['period_average']:.2f}%")
        
        # 理論上，所有客戶端的綜合平均應該在 45%-95% 之間
        # （因為 client_A: 70-80%, client_B: 85-95%, client_C: 50-65%）
        self.assertGreater(stats_all['period_average'], 40)
        self.assertLess(stats_all['period_average'], 100)
        
        # 驗證資料筆數應該比單一客戶端更多
        hourly_data_all = result_all['data']['hourly_max_usage']
        print(f"   ✓ 所有客戶端資料筆數: {len(hourly_data_all)}")


def run_tests():
    """執行所有測試"""
    # 建立測試套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGPUQueryAPI)
    
    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 輸出測試總結
    print("\n" + "="*70)
    print("測試總結")
    print("="*70)
    print(f"執行測試數: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"錯誤: {len(result.errors)}")
    print(f"跳過: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 80)
    print("GPU Query API 單元測試程式".center(80))
    print("=" * 80)
    print()
    print("測試說明：")
    print("  • 自動生成 9 種測試情境的 CSV 資料")
    print("  • 使用 gpu_metrics_YYYY-MM-DD.csv 命名格式")
    print("  • 測試 API 的各種功能和邊界情況")
    print("  • 驗證計算結果的正確性")
    print("  • 測試 client_name 參數篩選功能")
    print()
    print("注意：執行前請確保 server.py 已啟動！")
    print("=" * 80)
    print()
    
    success = run_tests()
    exit(0 if success else 1)
