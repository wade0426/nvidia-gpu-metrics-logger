# NVIDIA GPU Metrics Logger v2

一個基於 FastAPI 的分散式 NVIDIA GPU 監控系統。

## 系統架構

- **Client 端**: 收集本地 GPU 指標並傳送至 Server 端
- **Server 端**: 接收、存儲並提供 GPU 監控資料的 REST API 服務

## 安裝需求

### Python 環境
- Python 3.8 或更高版本

### 依賴套件
```bash
pip install -r requirements.txt
```

### NVIDIA 驅動需求（僅 Client 端）
- NVIDIA Driver 440.33 或更新版本
- 支援 CUDA 的 NVIDIA GPU

## 快速開始

### 1. 啟動 Server 端

#### Windows
```cmd
start_server.bat
```

#### Linux/Mac
```bash
chmod +x start_server.sh
./start_server.sh
```

#### 手動啟動
```bash
python server.py
```

### 2. 啟動 Client 端

```bash
python client.py
```

### 3. 訪問 API 文件

啟動 Server 後，可透過以下網址訪問：
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

## 配置文件

### Server 配置 (`config/server_config.ini`)
- **SERVER**: 伺服器設定（IP、端口、Debug模式）
- **STORAGE**: 資料存儲設定（CSV檔案路徑、命名等）
- **LOGGING**: 日誌系統設定
- **PERFORMANCE**: 效能優化設定

### Client 配置 (`config/client_config.ini`)
- **AUTHENTICATION**: 客戶端識別設定
- **MONITORING**: 監控間隔設定
- **SERVER**: Server 端連線設定
- **NETWORK**: 網路傳輸設定
- **GPU_METRICS**: 指標收集設定
- **LOGGING**: 日誌系統設定

## API 端點

### 資料接收
- `POST /api/receive-data` - 接收 GPU 監控資料

### 資料查詢
- `POST /api/gpu/statistics` - 取得統計數據
- `POST /api/gpu/hourly-usage` - 取得每小時使用率
- `POST /api/gpu/daily-usage` - 取得每日使用率
- `POST /api/gpu/list` - 取得 GPU 清單
- `POST /api/gpu/realtime` - 取得即時資料

### 系統狀態
- `GET /` - API 基本資訊
- `GET /health` - 健康檢查

## 資料格式

監控資料以 CSV 格式存儲，預設路徑為 `./data/gpu_metrics_YYYYMMDD.csv`

## 錯誤處理

系統提供完整的錯誤處理機制：
- 資料驗證錯誤（422）
- 檔案不存在（404）
- 伺服器內部錯誤（500）

## 效能特色

- **批次處理**: 支援批次資料傳輸
- **緩存機制**: 智能資料緩存
- **多執行緒**: 平行處理提升效能
- **記憶體優化**: 自動垃圾回收
- **容錯機制**: 完整的錯誤恢復

## 疑難排解

### 常見問題

1. **無法連線到 Server**
   - 檢查 Server 是否已啟動
   - 確認防火牆設定
   - 檢查配置文件中的 IP 和端口

2. **找不到 NVIDIA GPU**
   - 確認已安裝 NVIDIA 驅動
   - 執行 `nvidia-smi` 檢查 GPU 狀態

3. **依賴套件錯誤**
   - 確認已安裝所有必要套件：`pip install -r requirements.txt`

### Windows 使用者常見問題

4. **NVML Shared Library Not Found 錯誤**
   
   這個錯誤是 Windows 系統上使用 nvidia-ml-py3 套件時的常見問題。
   
   **問題原因**
   
   nvidia-ml-py3 套件需要找到 NVIDIA 的 nvml.dll 檔案才能正常運作，但程式找不到這個檔案。在 Windows 系統上，該套件會在以下位置尋找 nvml.dll：
   - `C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll`
   - `C:\Windows\System32\nvml.dll`
   
   **解決方案**
   
   **方案 1：建立 NVSMI 目錄並複製檔案**
   
   1. 建立目錄：
      ```
      C:\Program Files\NVIDIA Corporation\NVSMI\
      ```
   
   2. 尋找 nvml.dll 檔案：
      - 通常在 `C:\Windows\System32\nvml.dll`
      - 或在 NVIDIA 驅動程式安裝目錄中
   
   3. 複製檔案：
      - 將 nvml.dll 複製到 `C:\Program Files\NVIDIA Corporation\NVSMI\` 目錄中

### 日誌檢查

- Server 日誌: `./logs/server.log`
- Client 日誌: `./logs/client.log`
