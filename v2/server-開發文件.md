# NVIDIA GPU Metrics Server 文件

## 文檔說明

本文檔主要說明 NVIDIA GPU Metrics Logger 的 Server 端系統設計。

## 專案概述

這是一個基於 FastAPI 框架開發的 NVIDIA GPU 監控資料 REST API 服務。該 API 提供了完整的 GPU 效能監控數據查詢功能，包括統計分析、使用率趨勢、即時資料等多種監控指標的存取介面。

## 功能特點

- **RESTful API 設計**：提供標準化的 HTTP API 介面
- **CORS 支援**：支援跨域請求，方便前端整合
- **多維度數據查詢**：支援依時間範圍、GPU ID 篩選資料
- **統計分析**：提供小時、每日、期間平均使用率計算
- **即時監控**：取得最新的 GPU 狀態資訊
- **資料視覺化支援**：提供圖表資料格式化輸出
- **錯誤處理機制**：完整的異常處理和錯誤回應

## 技術架構

### 核心技術棧

- **Web 框架**：FastAPI 1.0.0
- **資料處理**：Pandas
- **資料驗證**：Pydantic
- **ASGI 伺服器**：Uvicorn
- **資料格式**：CSV 檔案

## 安裝與設定

### 環境需求

```bash
pip install fastapi
pip install uvicorn
pip install pandas
pip install pydantic
```

### 配置設定

```python
# CSV 資料檔案存放路徑
CSV_FOLDER = r"D:\Code\Python\nvidia-gpu-metrics-logger\logs"

# 伺服器設定
HOST = "0.0.0.0"
PORT = 8000
```

### 啟動服務

```bash
python main.py
```

或使用 Uvicorn 直接啟動：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 端點說明

### 1. 根路由

**端點**：`POST /`

**描述**：提供 API 基本資訊

**回應範例**：
```json
{
  "message": "NVIDIA GPU Metrics API",
  "version": "1.0.0",
  "docs_url": "/docs"
}
```

### 2. GPU 統計數據

**端點**：`POST /api/gpu/statistics`

**描述**：取得指定條件下的 GPU 使用率統計數據

**請求參數**：
```json
{
  "start_date": "2024-01-01",  // 可選，開始日期
  "end_date": "2024-01-31",    // 可選，結束日期
  "gpu_id": 0                   // 可選，GPU ID
}
```

**回應範例**：
```json
{
  "success": true,
  "data": {
    "hourly_average": 45.2,
    "daily_average": 42.8,
    "period_average": 44.1,
    "max_utilization": 98.5,
    "min_utilization": 0.0,
    "period": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    }
  }
}
```

### 3. 每小時使用率

**端點**：`POST /api/gpu/hourly-usage`

**描述**：取得指定日期的每小時 GPU 使用率資料

**請求參數**：
```json
{
  "date": "2024-01-15",  // 必填，查詢日期
  "gpu_id": 0            // 可選，GPU ID
}
```

**回應範例**：
```json
{
  "success": true,
  "data": {
    "chart_data": [
      {"hour": 0, "utilization": 12.5},
      {"hour": 1, "utilization": 8.3},
      // ... 24小時完整資料
    ],
    "date": "2024-01-15"
  }
}
```

### 4. 每日使用率

**端點**：`POST /api/gpu/daily-usage`

**描述**：取得指定期間的每日最高/最低使用率

**請求參數**：
```json
{
  "start_date": "2024-01-01",  // 必填，開始日期
  "end_date": "2024-01-31",    // 必填，結束日期
  "gpu_id": 0                   // 可選，GPU ID
}
```

**回應範例**：
```json
{
  "success": true,
  "data": {
    "chart_data": [
      {
        "date": "01/01",
        "min_utilization": 5.2,
        "max_utilization": 85.6
      },
      // ... 期間內每日資料
    ]
  }
}
```

### 5. GPU 清單

**端點**：`POST /api/gpu/list`

**描述**：取得系統中所有 GPU 的清單資訊

**回應範例**：
```json
{
  "success": true,
  "data": [
    {
      "gpu_id": 0,
      "gpu_name": "NVIDIA GeForce RTX 4090",
      "status": "active"
    },
    {
      "gpu_id": 1,
      "gpu_name": "NVIDIA GeForce RTX 4080",
      "status": "active"
    }
  ]
}
```

### 6. 即時資料

**端點**：`POST /api/gpu/realtime`

**描述**：取得最新的即時 GPU 監控資料

**回應範例**：
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-15 14:30:25",
    "gpu_id": 0,
    "utilization_gpu": 75.2,
    "temperature": 68,
    "memory_used_percent": 82.5
  }
}
```

## 資料模型

### StatisticsRequest
```python
class StatisticsRequest(BaseModel):
    start_date: Optional[str] = None    # 格式：YYYY-MM-DD
    end_date: Optional[str] = None      # 格式：YYYY-MM-DD  
    gpu_id: Optional[int] = None        # GPU 編號
```

### HourlyUsageRequest
```python
class HourlyUsageRequest(BaseModel):
    date: str                           # 必填，格式：YYYY-MM-DD
    gpu_id: Optional[int] = None        # GPU 編號
```

### DailyUsageRequest
```python
class DailyUsageRequest(BaseModel):
    start_date: str                     # 必填，格式：YYYY-MM-DD
    end_date: str                       # 必填，格式：YYYY-MM-DD
    gpu_id: Optional[int] = None        # GPU 編號
```

## 核心功能模組

### 資料載入模組
- **`load_csv_data()`**：載入並合併所有 CSV 監控資料檔案
- 支援多檔案自動合併
- 時間戳記標準化處理
- 新增日期和小時欄位供後續分析使用

### 資料篩選模組
- **`filter_by_date_range()`**：依據起迄日期篩選資料
- **`filter_by_gpu_id()`**：依據 GPU ID 篩選資料
- 支援彈性的篩選條件組合

### 統計計算模組
- 小時平均使用率計算
- 每日平均使用率計算
- 期間總平均使用率計算  
- 最高/最低使用率統計
- 記憶體使用率百分比計算

## 錯誤處理

### 常見錯誤碼

| 狀態碼 | 錯誤類型 | 說明 |
|--------|----------|------|
| 404 | 資料不存在 | 找不到指定條件的監控資料 |
| 404 | 檔案不存在 | CSV 監控資料檔案不存在 |
| 500 | 伺服器錯誤 | 資料處理或系統內部錯誤 |

### 錯誤回應格式
```json
{
  "detail": "錯誤描述訊息"
}
```

## 資料檔案格式

### CSV 檔案結構
預期的 CSV 檔案應包含以下欄位：

| 欄位名稱 | 資料型別 | 說明 | 單位 |
|---------|---------|------|------|
| timestamp | datetime | 記錄時間 | YYYY-MM-DD HH:MM:SS |
| gpu_id | int | GPU 編號 | - |
| gpu_name | string | GPU 型號名稱 | - |
| utilization_gpu | float | GPU 使用率 | % |
| utilization_memory | float | 記憶體使用率 | % |
| memory_total | int | 總記憶體 | MB |
| memory_used | int | 已使用記憶體 | MB |
| memory_free | int | 可用記憶體 | MB |
| temperature | int | GPU 溫度 | °C |
| power_draw | float | 功耗 | W |
| power_limit | float | 功耗限制 | W |
| fan_speed | int | 風扇轉速 | % |
| client_name | str | 客戶端識別名稱 | - |

### 檔案命名規則
```
gpu_metrics_YYYYMMDD.csv
gpu_metrics_20240115.csv
```


## API 文件存取

啟動服務後，可透過以下網址存取 API 文件：

- **Swagger UI**：`http://localhost:8000/docs`
- **ReDoc**：`http://localhost:8000/redoc`


## 效能優化

### 1. 網路傳輸優化
```python
# 批次傳送減少網路開銷
batch_size = 10
data_batch = []
```

### 2. 記憶體使用優化
  - 定期清理無用物件
  ```
  import gc
  gc.collect()
  ```
  - **批次處理**: 大量資料分批處理避免記憶體溢出
  - **資料清理**: 定期清理過期暫存資料
  - **緩存限制**: 設定合理的緩存大小上限

### 3. 資料庫效能優化
- 使用 pandas 的 `to_csv` 模式優化
- 實作資料索引加速查詢
- 定期整理和壓縮歷史資料