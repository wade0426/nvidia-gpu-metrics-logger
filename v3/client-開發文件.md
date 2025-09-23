# NVIDIA GPU Metrics Logger Client 文件

## 文檔說明

本文檔主要說明 NVIDIA GPU Metrics Logger 的 Client 端系統設計。

## 專案簡介

一個基於 Python 的分散式 NVIDIA GPU 監測系統，採用 client-server 架構設計。Client 端負責收集本地 GPU 指標並傳送至 Server 端，Server 端則統一收集、處理並持久化存儲監控資料。適用於深度學習訓練監控、多節點叢集效能分析、GPU 資源管理等場景。

## 功能特色

- 🔄 **即時監控**: 可自定義時間間隔持續監控 GPU 狀態
- 📊 **完整指標**: 記錄 GPU 使用率、記憶體使用量、溫度、功耗、風扇轉速等關鍵指標
- 🌐 **分散式架構**: 支援多台機器同時監控，集中管理監控資料
- 💾 **資料傳送至 Server 端**: 自動將監控資料傳送到 Server 端，包含完整時間戳記和來源識別等
- 🖥️ **多 GPU 支援**: 自動偵測並監控系統中所有 NVIDIA GPU
- ⚡ **輕量化設計**: 最小化系統資源占用，對目標系統影響極小
- 🔧 **靈活配置**: 透過設定檔案輕鬆調整監控參數和網路設定

## 技術棧

- **Python 3.8+**
- **nvidia-ml-py3**: NVIDIA 管理庫 Python 綁定
- **datetime**: 時間戳記處理
- **configparser**: 設定檔管理

## 系統需求

### 硬體需求
- NVIDIA GPU（支援 CUDA）
- NVIDIA 驅動程式已安裝

### 軟體需求
- Python 3.8 或更高版本
- NVIDIA Driver 440.33+ 或更新版本

## 安裝與配置

### 1. 套件安裝
```
pip install nvidia-ml-py3 pandas
```

### 2. 環境檢查
確認 NVIDIA 驅動與 CUDA 工具包已正確安裝：
```
nvidia-smi
```

### 3. 設定檔案
創建 `config.ini` 設定監控參數

## 資料格式

### 送給 Server 的資料格式
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

## 錯誤處理機制

### 1. GPU 偵測失敗
- 自動重試機制
- 錯誤日誌記錄

### 2. 檔案寫入錯誤
- 目錄權限檢查
- 磁碟空間監控
- 錯誤日誌記錄

### 3. 記憶體不足
- 資料批次寫入
- 緩存大小限制
- 自動清理機制
- 錯誤日誌記錄

## 效能優化

### 1. 記憶體使用
- 定期清理無用物件
- 控制緩存大小