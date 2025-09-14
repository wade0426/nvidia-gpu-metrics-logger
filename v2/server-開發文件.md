# NVIDIA GPU Metrics Server 文件

## 文檔說明

本文檔主要說明 NVIDIA GPU Metrics Logger 的 Server 端系統設計。

## 專案概述

這是一個基於 FastAPI 框架開發的 NVIDIA GPU 監控資料 REST API 服務。該 API 提供了完整的 GPU 效能監控數據查詢功能，包括統計分析、使用率趨勢、即時資料等多種監控指標的存取介面，支援多客戶端資料收集和查詢。

## 功能特點

- **RESTful API 設計**：提供標準化的 HTTP API 介面
- **CORS 支援**：支援跨域請求，方便前端整合
- **多維度數據查詢**：支援依時間範圍、GPU ID、客戶端名稱篩選資料
- **多客戶端支援**：區分不同客戶端的監控資料
- **統計分析**：提供小時、每日、期間平均使用率計算
- **即時監控**：取得最新的 GPU 狀態資訊
- **資料接收**：接收客戶端傳送的 GPU 監控資料
- **錯誤處理機制**：完整的異常處理和錯誤回應
- **智能預設值**：查詢無資料時返回預設值而非錯誤

## 技術架構

### 核心技術棧

- **Web 框架**：FastAPI
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

設定檔位於 `config/server_config.ini`，包含以下主要參數：

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

### 2. 接收監控資料

**端點**：`POST /api/gpu/receive-data`

**描述**：接收客戶端傳送的 GPU 監控資料

**請求參數**：
```json
{
  "timestamp": "2024-01-15 14:30:25",
  "gpu_id": 0,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "utilization_gpu": 75.2,
  "utilization_memory": 82.5,
  "memory_total": 24576,
  "memory_used": 20275,
  "memory_free": 4301,
  "temperature": 68,
  "power_draw": 320.5,
  "power_limit": 450.0,
  "fan_speed": 65,
  "client_name": "gpu-client-001"
}
```

**回應範例**：
```json
{
  "success": true,
  "message": "Data received and saved successfully",
  "timestamp": "2024-01-15 14:30:25"
}
```

### 3. GPU 統計數據

**端點**：`POST /api/gpu/statistics`

**描述**：取得指定條件下的 GPU 使用率統計數據

**請求參數**：
```json
{
  "start_date": "2024-01-01",      // 可選，開始日期
  "end_date": "2024-01-31",        // 可選，結束日期
  "gpu_id": 0,                     // 可選，GPU ID
  "client_name": "gpu-client-001"  // 可選，客戶端名稱
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
    },
    "gpu_id": 0,
    "client_name": "gpu-client-001"
  }
}
```

### 4. 每小時使用率

**端點**：`POST /api/gpu/hourly-usage`

**描述**：取得指定日期範圍的每小時 GPU 使用率資料

**請求參數**：
```json
{
  "start_date": "2024-08-27",      // 必填，開始日期
  "end_date": "2024-09-09",        // 必填，結束日期
  "gpu_id": 0,                     // 可選，GPU ID
  "client_name": "gpu-client-001"  // 可選，客戶端名稱
}
```

**回應範例**：
```json
{
  "success": true,
  "data": {
    "chart_data": [
      {
        "date": "2024-08-27",
        "hour": 0,
        "utilization": 12.5
      },
      {
        "date": "2024-08-27",
        "hour": 1,
        "utilization": 8.3
      },
      // ... 指定日期範圍內所有小時資料
    ],
    "period": {
      "start_date": "2024-08-27",
      "end_date": "2024-09-09"
    },
    "gpu_id": 0,
    "client_name": "gpu-client-001"
  }
}
```

**特殊處理**：
- 如果查詢的日期範圍內沒有任何資料，將返回該期間所有小時的預設值 0
- 缺少的小時資料將自動補 0

### 5. 每日使用率

**端點**：`POST /api/gpu/daily-usage`

**描述**：取得指定期間的每日最高/最低使用率

**請求參數**：
```json
{
  "start_date": "2024-01-01",      // 必填，開始日期
  "end_date": "2024-01-31",        // 必填，結束日期
  "gpu_id": 0,                     // 可選，GPU ID
  "client_name": "gpu-client-001"  // 可選，客戶端名稱
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
    ],
    "period": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    },
    "gpu_id": 0,
    "client_name": "gpu-client-001"
  }
}
```

**特殊處理**：
- 如果查詢的日期範圍內沒有任何資料，將返回該期間所有日期的預設值 0

### 6. GPU 清單

**端點**：`POST /api/gpu/list`

**描述**：取得系統中所有 GPU 的清單資訊

**請求參數**：
```json
{
  "client_name": "gpu-client-001"  // 可選，客戶端名稱篩選
}
```

**回應範例**：
```json
{
  "success": true,
  "data": [
    {
      "gpu_id": 0,
      "gpu_name": "NVIDIA GeForce RTX 4090",
      "client_name": "gpu-client-001",
      "status": "active",
      "last_update": "2024-01-15 14:30:25"
    },
    {
      "gpu_id": 1,
      "gpu_name": "NVIDIA GeForce RTX 4080",
      "client_name": "gpu-client-001",
      "status": "active",
      "last_update": "2024-01-15 14:30:20"
    }
  ]
}
```

### 7. 即時資料

**端點**：`POST /api/gpu/realtime`

**描述**：取得最新的即時 GPU 監控資料

**請求參數**：
```json
{
  "gpu_id": 0,                     // 可選，GPU ID
  "client_name": "gpu-client-001"  // 可選，客戶端名稱
}
```

**回應範例**：
```json
{
  "success": true,
  "data": {
    "timestamp": "2024-01-15 14:30:25",
    "type": "average",
    "scope": "all_clients_all_gpus",  // 或 "single_client_all_gpus" 或 "single_client_single_gpu"
    "utilization_gpu": 65.8,
    "temperature": 72,
    "memory_used_percent": 78.3,
    "power_draw": 285.2,
    "gpu_count": 4,
    "client_count": 2
  }
}
```

**篩選邏輯**：
- **無任何參數**：返回所有客戶端所有 GPU 的平均值
- **只有 client_name**：返回指定客戶端所有 GPU 的平均值
- **只有 gpu_id**：返回所有客戶端指定 GPU ID 的平均值
- **兩者都有**：返回指定客戶端指定 GPU 的資料

## 資料模型

### ReceiveDataRequest
```python
class ReceiveDataRequest(BaseModel):
    timestamp: str                      # 格式：YYYY-MM-DD HH:MM:SS
    gpu_id: int                         # GPU 編號
    gpu_name: str                       # GPU 型號名稱
    utilization_gpu: float              # GPU 使用率 (%)
    utilization_memory: float           # 記憶體使用率 (%)
    memory_total: int                   # 總記憶體 (MB)
    memory_used: int                    # 已使用記憶體 (MB)
    memory_free: int                    # 可用記憶體 (MB)
    temperature: int                    # GPU 溫度 (°C)
    power_draw: float                   # 功耗 (W)
    power_limit: float                  # 功耗限制 (W)
    fan_speed: int                      # 風扇轉速 (%)
    client_name: str                    # 客戶端識別名稱
```

### StatisticsRequest
```python
class StatisticsRequest(BaseModel):
    start_date: Optional[str] = None    # 格式：YYYY-MM-DD
    end_date: Optional[str] = None      # 格式：YYYY-MM-DD  
    gpu_id: Optional[int] = None        # GPU 編號
    client_name: Optional[str] = None   # 客戶端名稱
```

### HourlyUsageRequest
```python
class HourlyUsageRequest(BaseModel):
    start_date: str                     # 必填，格式：YYYY-MM-DD
    end_date: str                       # 必填，格式：YYYY-MM-DD
    gpu_id: Optional[int] = None        # GPU 編號
    client_name: Optional[str] = None   # 客戶端名稱
```

### DailyUsageRequest
```python
class DailyUsageRequest(BaseModel):
    start_date: str                     # 必填，格式：YYYY-MM-DD
    end_date: str                       # 必填，格式：YYYY-MM-DD
    gpu_id: Optional[int] = None        # GPU 編號
    client_name: Optional[str] = None   # 客戶端名稱
```

### RealtimeRequest
```python
class RealtimeRequest(BaseModel):
    gpu_id: Optional[int] = None        # GPU 編號
    client_name: Optional[str] = None   # 客戶端名稱
```

### GPUListRequest
```python
class GPUListRequest(BaseModel):
    client_name: Optional[str] = None   # 客戶端名稱篩選
```

## 核心功能模組

### 資料接收模組
- **`receive_gpu_data()`**：接收並驗證客戶端傳送的 GPU 監控資料
- **`save_to_csv()`**：將接收到的資料儲存至 CSV 檔案
- 支援資料格式驗證和錯誤處理
- 自動建立檔案目錄和檔案命名

### 資料載入模組
- **`load_csv_data()`**：載入並合併所有 CSV 監控資料檔案
- 支援多檔案自動合併
- 時間戳記標準化處理
- 新增日期和小時欄位供後續分析使用

### 資料篩選模組
- **`filter_by_date_range()`**：依據起迄日期篩選資料
- **`filter_by_gpu_id()`**：依據 GPU ID 篩選資料
- **`filter_by_client_name()`**：依據客戶端名稱篩選資料
- 支援彈性的篩選條件組合
- 多維度篩選邏輯處理

### 統計計算模組
- 小時平均使用率計算
- 每日平均使用率計算
- 期間總平均使用率計算  
- 最高/最低使用率統計
- 記憶體使用率百分比計算
- 多客戶端資料聚合運算
- 預設值填充機制

### 即時資料處理模組
- **`get_latest_data()`**：取得最新監控資料
- **`calculate_average()`**：計算多 GPU/客戶端平均值
- 支援不同篩選條件的聚合邏輯
- 智能資料匯總功能

## 錯誤處理

### 常見錯誤碼

| 狀態碼 | 錯誤類型 | 說明 |
|--------|----------|------|
| 400 | 請求格式錯誤 | 請求參數格式不正確或缺少必要參數 |
| 404 | 資料不存在 | 找不到指定條件的監控資料（返回預設值） |
| 404 | 檔案不存在 | CSV 監控資料檔案不存在 |
| 422 | 資料驗證錯誤 | 傳送的資料格式不符合規範 |
| 500 | 伺服器錯誤 | 資料處理或系統內部錯誤 |

### 錯誤回應格式
```json
{
  "success": false,
  "detail": "錯誤描述訊息",
  "error_code": "ERROR_CODE"
}
```

### 資料缺失處理
- **查詢無資料**：返回預設值 0 而非錯誤
- **時間範圍缺失**：自動填充缺失時間點的預設值

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
# 批次接收減少處理開銷
@app.post("/api/gpu/receive-data-batch")
async def receive_batch_data(data_list: List[ReceiveDataRequest]):
    # 批次處理多筆資料
    pass
```

### 2. 記憶體使用優化
- 定期清理無用物件
```python
import gc
gc.collect()
```
- **批次處理**: 大量資料分批處理避免記憶體溢出
- **資料清理**: 定期清理過期暫存資料
- **緩存限制**: 設定合理的緩存大小上限

### 3. 資料存取效能優化
- 使用 pandas 的 `to_csv` 模式優化
- 實作資料索引加速查詢
- 定期整理和壓縮歷史資料
- 建立資料快取機制提升查詢速度

### 4. 多客戶端處理優化
```python
# 使用字典快速分組客戶端資料
client_data = df.groupby('client_name').agg({
    'utilization_gpu': 'mean',
    'temperature': 'mean',
    'memory_used': 'sum'
}).to_dict()
```
