# NVIDIA GPU Metrics API 使用說明

## 概述
這是一個基於 FastAPI 的 GPU 監控資料 API 服務，提供 GPU 使用率統計、圖表資料和即時監控功能。

## 安裝依賴
```bash
pip install -r requirements.txt
```

## 啟動服務

### Windows
```bash
# 使用批次檔啟動
start_api.bat

# 或直接使用 Python
python api-main.py
```

### Linux/Mac
```bash
# 添加執行權限（僅需執行一次）
chmod +x start_api.sh

# 使用腳本啟動
./start_api.sh

# 或直接使用 Python
python api-main.py
```

## API 端點

### 1. 根路由
- **端點**: `GET /`
- **描述**: API 基本資訊

### 2. 統計數據 API
- **端點**: `GET /api/gpu/statistics`
- **參數**: 
  - `start_date`: 開始日期 (可選, YYYY-MM-DD)
  - `end_date`: 結束日期 (可選, YYYY-MM-DD)
  - `gpu_id`: GPU ID (可選)
- **描述**: 取得 GPU 使用率統計數據

### 3. 每小時使用率 API
- **端點**: `GET /api/gpu/hourly-usage`
- **參數**: 
  - `date`: 指定日期 (必需, YYYY-MM-DD)
  - `gpu_id`: GPU ID (可選)
- **描述**: 取得指定日期的 24 小時使用率資料

### 4. 每日使用率 API
- **端點**: `GET /api/gpu/daily-usage`
- **參數**: 
  - `start_date`: 開始日期 (必需, YYYY-MM-DD)
  - `end_date`: 結束日期 (必需, YYYY-MM-DD)
  - `gpu_id`: GPU ID (可選)
- **描述**: 取得指定期間內每日的最高最低使用率。對於沒有資料的日期，會預設顯示 0 值

### 5. GPU 清單 API
- **端點**: `GET /api/gpu/list`
- **描述**: 取得所有 GPU 的清單資訊

### 6. 即時資料 API
- **端點**: `GET /api/gpu/realtime`
- **描述**: 取得最新的 GPU 即時監控資料

## 使用範例

### 取得統計數據
```bash
# 取得所有資料的統計
http://localhost:8000/api/gpu/statistics

# 取得指定日期範圍的統計
http://localhost:8000/api/gpu/statistics?start_date=2025-08-20&end_date=2025-08-27

# 取得指定 GPU 的統計
http://localhost:8000/api/gpu/statistics?gpu_id=0
```

### 取得每小時使用率
```bash
# 取得指定日期的每小時使用率
http://localhost:8000/api/gpu/hourly-usage?date=2025-08-27

# 取得指定 GPU 的每小時使用率
http://localhost:8000/api/gpu/hourly-usage?date=2025-08-27&gpu_id=0
```

### 取得每日使用率
```bash
# 取得指定期間的每日使用率
http://localhost:8000/api/gpu/daily-usage?start_date=2025-08-20&end_date=2025-08-27
```

### 取得 GPU 清單
```bash
http://localhost:8000/api/gpu/list
```

### 取得即時資料
```bash
http://localhost:8000/api/gpu/realtime
```

## API 文檔
啟動服務後，可在以下網址查看詳細的 API 文檔：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 注意事項
1. 確保 `logs` 資料夾中有 GPU 監控資料檔案（格式：`gpu_metrics_*.csv`）
2. API 預設運行在 `http://localhost:8000`
3. 所有日期參數格式為 `YYYY-MM-DD`
4. API 支援 CORS，可以從前端應用程式直接調用
5. 錯誤情況會回傳適當的 HTTP 狀態碼和錯誤訊息
6. 每日使用率 API 會為指定期間內的所有日期提供資料，沒有資料的日期會顯示 0 值

## 資料格式
API 回傳的所有資料都遵循統一格式：
```json
{
  "success": true,
  "data": {
    // 具體資料內容
  }
}
```

## 疑難排解
- 如果遇到 "找不到 GPU 監控資料檔案" 錯誤，請確認 `logs` 資料夾中有相應的 CSV 檔案
- 如果 API 無法啟動，請檢查是否已安裝所有依賴項目
- 確保埠口 8000 沒有被其他應用程式佔用
