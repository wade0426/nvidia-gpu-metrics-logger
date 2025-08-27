#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger 執行腳本

這是一個方便的執行腳本，用於啟動 GPU 監控系統
"""

import os
import sys

# 添加 src 目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# 導入主程式
from main import main

if __name__ == "__main__":
    main()
