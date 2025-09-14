#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Server 測試工具

簡單的測試腳本，用於驗證 Server API 功能
"""

import requests
import json
import datetime
from typing import Dict, Any


def test_server_health(base_url: str = "http://localhost:5000") -> bool:
    """測試伺服器健康狀態"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 伺服器健康檢查通過")
            return True
        else:
            print(f"❌ 伺服器健康檢查失敗，狀態碼: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 無法連線到伺服器: {e}")
        return False


def test_api_info(base_url: str = "http://localhost:5000") -> bool:
    """測試 API 基本資訊"""
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API 資訊: {data.get('message')} v{data.get('version')}")
            return True
        else:
            print(f"❌ API 資訊請求失敗，狀態碼: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API 資訊請求錯誤: {e}")
        return False


def test_receive_data(base_url: str = "http://localhost:5000") -> bool:
    """測試資料接收端點（單一資料）"""
    try:
        test_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_id": 0,
            "gpu_name": "NVIDIA Test GPU",
            "utilization_gpu": 75.5,
            "utilization_memory": 82.3,
            "memory_total": 8192,
            "memory_used": 6739,
            "memory_free": 1453,
            "temperature": 65,
            "power_draw": 180.5,
            "power_limit": 220.0,
            "fan_speed": 55,
            "client_name": "test-client"
        }
        
        response = requests.post(
            f"{base_url}/api/receive-data",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ 單一資料接收測試通過")
                return True
            else:
                print(f"❌ 單一資料接收失敗: {result}")
                return False
        else:
            print(f"❌ 單一資料接收請求失敗，狀態碼: {response.status_code}")
            print(f"   回應內容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 單一資料接收測試錯誤: {e}")
        return False


def test_receive_batch_data(base_url: str = "http://localhost:5000") -> bool:
    """測試批次資料接收端點"""
    try:
        # 創建批次測試資料
        batch_data = []
        for i in range(3):
            test_data = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gpu_id": i,
                "gpu_name": f"NVIDIA Test GPU {i}",
                "utilization_gpu": 70.0 + i * 5,
                "utilization_memory": 80.0 + i * 2,
                "memory_total": 8192,
                "memory_used": 6000 + i * 200,
                "memory_free": 2192 - i * 200,
                "temperature": 60 + i * 3,
                "power_draw": 170.0 + i * 10,
                "power_limit": 220.0,
                "fan_speed": 50 + i * 5,
                "client_name": "test-client-batch"
            }
            batch_data.append(test_data)
        
        response = requests.post(
            f"{base_url}/api/receive-data",
            json=batch_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                received_count = result.get("received_count", 0)
                failed_count = result.get("failed_count", 0)
                print(f"✅ 批次資料接收測試通過：{received_count} 成功，{failed_count} 失敗")
                return True
            else:
                print(f"❌ 批次資料接收失敗: {result}")
                return False
        else:
            print(f"❌ 批次資料接收請求失敗，狀態碼: {response.status_code}")
            print(f"   回應內容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 批次資料接收測試錯誤: {e}")
        return False


def test_gpu_list(base_url: str = "http://localhost:5000") -> bool:
    """測試 GPU 清單端點"""
    try:
        test_data = {"client_name": None}
        
        response = requests.post(
            f"{base_url}/api/gpu/list",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                gpu_count = len(result.get("data", []))
                print(f"✅ GPU 清單測試通過，找到 {gpu_count} 個 GPU")
                return True
            else:
                print(f"❌ GPU 清單請求失敗: {result}")
                return False
        else:
            print(f"❌ GPU 清單請求失敗，狀態碼: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ GPU 清單測試錯誤: {e}")
        return False


def test_statistics(base_url: str = "http://localhost:5000") -> bool:
    """測試統計數據端點"""
    try:
        test_data = {
            "start_date": None,
            "end_date": None,
            "gpu_id": None,
            "client_name": None
        }
        
        response = requests.post(
            f"{base_url}/api/gpu/statistics",
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                stats = result.get("data", {})
                avg_util = stats.get("period_average", 0)
                print(f"✅ 統計數據測試通過，平均使用率: {avg_util}%")
                return True
            else:
                print(f"❌ 統計數據請求失敗: {result}")
                return False
        else:
            print(f"❌ 統計數據請求失敗，狀態碼: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 統計數據測試錯誤: {e}")
        return False


def main():
    """主測試函數"""
    print("=" * 50)
    print("NVIDIA GPU Metrics Server 測試工具")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    tests = [
        ("伺服器健康檢查", lambda: test_server_health(base_url)),
        ("API 基本資訊", lambda: test_api_info(base_url)),
        ("單一資料接收功能", lambda: test_receive_data(base_url)),
        ("批次資料接收功能", lambda: test_receive_batch_data(base_url)),
        ("GPU 清單功能", lambda: test_gpu_list(base_url)),
        ("統計數據功能", lambda: test_statistics(base_url))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 測試: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   測試失敗")
    
    print("\n" + "=" * 50)
    print(f"測試結果: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("🎉 所有測試都通過！Server 運行正常")
    else:
        print("⚠️  部分測試失敗，請檢查 Server 狀態")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
