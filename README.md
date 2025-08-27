# NVIDIA GPU Metrics Logger

一個基於 Python 的 NVIDIA GPU 監測系統，能夠持續監控所有 NVIDIA GPU 的使用狀況並記錄至 CSV 檔案，適用於深度學習訓練過程監控、系統效能分析等場景。

## 功能特色

- 🔄 **即時監控**: 可自定義時間間隔持續監控 GPU 狀態
- 📊 **完整指標**: 記錄 GPU 使用率、記憶體使用量、溫度、功耗等關鍵指標
- 💾 **資料持久化**: 自動將監控資料寫入 CSV 檔案，包含完整時間戳記
- 🖥️ **多 GPU 支援**: 自動偵測並監控系統中所有 NVIDIA GPU
- ⚡ **輕量化設計**: 最小化系統資源占用
- 🛠️ **彈性設定**: 支援完整的設定檔自訂
- 🔄 **自動重啟**: 內建錯誤處理與自動重啟機制
- 🔧 **相容性強**: 支援多種 NVML 版本，自動處理 API 差異
- 🛡️ **錯誤容錯**: 優雅的錯誤處理，確保監控持續運行

## 系統需求

### 硬體需求
- NVIDIA GPU（支援 CUDA）
- NVIDIA 驅動程式已安裝

### 軟體需求
- Python 3.8 或更高版本
- NVIDIA Driver 440.33+ 或更新版本
- nvidia-ml-py3 11.450.129+ 或更新版本

## 快速開始

### 1. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 2. 檢查相容性（推薦）
```bash
python check_nvml_compatibility.py
```

### 3. 執行測試模式
```bash
python run_monitor.py --test
```

### 4. 開始監控
```bash
python run_monitor.py
```

### 5. 使用自訂設定檔
```bash
python run_monitor.py --config my_config.ini
```

## 專案結構

```
nvidia-gpu-metrics-logger/
├── src/                        # 原始碼目錄
│   ├── gpu_monitor.py          # GPU 監控核心
│   ├── data_logger.py          # 資料記錄器
│   ├── utils.py                # 工具函數
│   └── main.py                 # 主程式
├── config/                     # 設定檔目錄
│   └── config.ini              # 預設設定檔
├── logs/                       # CSV 輸出目錄
├── tests/                      # 單元測試
├── requirements.txt            # Python 套件依賴
├── setup.py                    # 安裝腳本
├── run_monitor.py              # 執行腳本
├── check_nvml_compatibility.py # NVML 相容性檢查
├── test_imports.py             # 導入測試腳本
├── demo.py                     # 功能演示腳本
└── README.md                   # 專案說明
```

## 設定檔

設定檔位於 `config/config.ini`，包含以下主要參數：

```ini
[MONITORING]
interval_seconds = 5            # 監控間隔（秒）
output_directory = ./logs       # 輸出目錄
csv_filename_prefix = gpu_metrics  # CSV 檔名前綴
max_file_size_mb = 100         # 最大檔案大小（MB）

[GPU_METRICS]
include_utilization = true      # 包含使用率
include_memory = true           # 包含記憶體資訊
include_temperature = true      # 包含溫度
include_power = true            # 包含功耗
include_fan_speed = true        # 包含風扇轉速
```

## 輸出資料格式

CSV 檔案包含以下欄位：

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

## 執行測試

```bash
# 檢查導入相容性
python test_imports.py

# 檢查 NVML 相容性
python check_nvml_compatibility.py

# 執行單元測試
python -m pytest tests/

# 執行特定測試
python -m pytest tests/test_gpu_monitor.py

# 執行測試並顯示覆蓋率
python -m pytest tests/ --cov=src
```

## 安裝為套件

```bash
pip install -e .
```

安裝後可以使用命令列工具：

```bash
gpu-monitor --help
nvidia-gpu-logger --config my_config.ini
```

## 使用範例

### 基本監控
```python
from src.gpu_monitor import GPUMonitor
from src.data_logger import DataLogger
from src.utils import load_config

# 載入設定
config = load_config("config/config.ini")

# 初始化組件
gpu_monitor = GPUMonitor(config)
data_logger = DataLogger(config)

# 收集一次指標
metrics = gpu_monitor.collect_all_gpu_metrics()
data_logger.log_metrics(metrics)
```

### 持續監控
```python
from src.main import MainController

# 啟動監控
controller = MainController("config/config.ini")
controller.run_with_auto_restart()
```

### 相容性檢查
```python
# 檢查 NVML 相容性
import pynvml

# 檢查是否支援特定功能
if hasattr(pynvml, 'nvmlDeviceGetPowerUsage'):
    print("支援功耗查詢")
else:
    print("不支援功耗查詢")

# 系統會自動處理相容性問題
```

## 故障排除

### 常見問題

1. **NVML 初始化失敗**
   - 確認 NVIDIA 驅動程式已正確安裝
   - 檢查 `nvidia-smi` 命令是否正常運作
   - 執行相容性檢查：`python check_nvml_compatibility.py`

2. **找不到 GPU**
   - 確認系統有 NVIDIA GPU
   - 確認 GPU 驅動程式版本支援 NVML
   - 檢查 GPU 是否被其他程序佔用

3. **權限錯誤**
   - 確認對日誌目錄有寫入權限
   - 在 Linux 上可能需要將使用者加入適當群組

4. **API 相容性錯誤**
   - 更新 nvidia-ml-py3：`pip install --upgrade nvidia-ml-py3`
   - 系統已內建相容性檢查，會自動處理版本差異
   - 某些舊版本 GPU 可能不支援所有功能（如功耗、風扇轉速）

### 相容性檢查

```bash
# 檢查 NVML 相容性
python check_nvml_compatibility.py

# 檢查導入是否正常
python test_imports.py
```

### 偵錯模式

```bash
# 啟用詳細日誌
python run_monitor.py --config config/config.ini

# 修改設定檔中的日誌等級
[LOGGING]
log_level = DEBUG
```

### 版本相容性

| 功能 | 最低版本要求 | 備註 |
|------|-------------|------|
| 基本 GPU 監控 | nvidia-ml-py3 11.450.129+ | 支援使用率、記憶體、溫度 |
| 功耗監控 | nvidia-ml-py3 11.450.129+ | 部分 GPU 可能不支援 |
| 風扇轉速 | nvidia-ml-py3 11.450.129+ | 部分 GPU 可能不支援 |
| 功耗限制 | nvidia-ml-py3 11.450.129+ | 自動選擇可用 API |

## 更新日誌

### v1.0.1 (2025-08-27)
- 修復相對導入問題
- 增強 NVML API 相容性
- 添加自動版本檢測和替代 API
- 新增相容性檢查工具
- 改善錯誤處理機制

### v1.0.0 (2025-08-27)
- 初始版本發布
- 支援多 GPU 監控
- CSV 資料記錄功能
- 設定檔支援
- 自動重啟機制

