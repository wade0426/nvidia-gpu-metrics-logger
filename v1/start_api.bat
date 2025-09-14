@echo off
echo 正在啟動 NVIDIA GPU Metrics API 服務...
echo 請確保已安裝所需依賴: pip install -r requirements.txt
echo.
echo API 服務將在 http://localhost:8000 啟動
echo API 文檔可在 http://localhost:8000/docs 查看
echo.
python api-main.py
pause
