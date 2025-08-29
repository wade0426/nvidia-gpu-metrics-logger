#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的 API 測試腳本
用於驗證 GPU Metrics API 的功能
"""

import requests
import json
from datetime import datetime, date

# API 基本 URL
BASE_URL = "http://localhost:8000"

def test_api_endpoint(endpoint, params=None):
    """測試 API 端點"""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.post(url, json=params)
        
        print(f"測試: {endpoint}")
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"回應: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"錯誤: {response.text}")
        
        print("-" * 50)
        
    except requests.exceptions.ConnectionError:
        print(f"連接錯誤: 無法連接到 {BASE_URL}")
        print("請確保 API 服務正在運行")
        return False
    except Exception as e:
        print(f"測試錯誤: {e}")
        return False
    
    return True

def main():
    """主測試函數"""
    print("NVIDIA GPU Metrics API 測試")
    print("=" * 50)
    
    # 測試根路由
    if not test_api_endpoint("/"):
        return
    
    # 測試 GPU 清單
    test_api_endpoint("/api/gpu/list")
    
    # 測試即時資料
    test_api_endpoint("/api/gpu/realtime")
    
    # 測試統計數據
    test_api_endpoint("/api/gpu/statistics", {})
    
    # 測試每小時使用率（使用今天的日期）
    today = date.today().strftime("%Y-%m-%d")
    # test_api_endpoint("/api/gpu/hourly-usage", {"date": today})
    test_api_endpoint("/api/gpu/hourly-usage", {"date": "2025-08-27"})
    
    # 測試每日使用率（使用過去7天）
    from datetime import timedelta
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    test_api_endpoint("/api/gpu/daily-usage", {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    })
    
    print("測試完成！")

if __name__ == "__main__":
    main()
