import os
import glob
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 初始化 FastAPI 應用
app = FastAPI(
    title="NVIDIA GPU Metrics API",
    description="提供 GPU 監控資料的 REST API",
    version="1.0.0"
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義常數 CSV 存放的資料夾
CSV_FOLDER = r"D:\Code\Python\nvidia-gpu-metrics-logger\logs"


# 資料處理函數
def load_csv_data() -> pd.DataFrame:
    """載入所有 CSV 檔案並合併"""
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "gpu_metrics_*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="找不到 GPU 監控資料檔案")
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            print(f"無法讀取檔案 {file}: {e}")
            continue
    
    if not all_data:
        raise HTTPException(status_code=500, detail="無法讀取任何資料檔案")
    
    # 合併所有資料
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 轉換時間格式
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df['date'] = combined_df['timestamp'].dt.date
    combined_df['hour'] = combined_df['timestamp'].dt.hour
    
    return combined_df


def filter_by_date_range(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """根據日期範圍過濾資料"""
    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        df = df[df['date'] >= start_date_obj]
    
    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df[df['date'] <= end_date_obj]
    
    return df


def filter_by_gpu_id(df: pd.DataFrame, gpu_id: Optional[int]) -> pd.DataFrame:
    """根據 GPU ID 過濾資料"""
    if gpu_id is not None:
        df = df[df['gpu_id'] == gpu_id]
    return df


# API 端點實現

@app.get("/")
async def root():
    """根路由，提供 API 基本資訊"""
    return {
        "message": "NVIDIA GPU Metrics API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/api/gpu/statistics")
async def get_gpu_statistics(
    start_date: Optional[str] = Query(None, description="開始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="結束日期 (YYYY-MM-DD)"),
    gpu_id: Optional[int] = Query(None, description="GPU ID")
):
    """取得 GPU 統計數據"""
    try:
        df = load_csv_data()
        df = filter_by_date_range(df, start_date, end_date)
        df = filter_by_gpu_id(df, gpu_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="指定條件下沒有找到資料")
        
        # 計算每小時平均
        hourly_avg = df.groupby(['date', 'hour'])['utilization_gpu'].mean()
        hourly_average = hourly_avg.mean()
        
        # 計算每日平均
        daily_avg = df.groupby('date')['utilization_gpu'].mean()
        daily_average = daily_avg.mean()
        
        # 計算期間平均
        period_average = df['utilization_gpu'].mean()
        
        # 計算最高最低使用率
        max_utilization = df['utilization_gpu'].max()
        min_utilization = df['utilization_gpu'].min()
        
        # 確定期間範圍
        period_start = df['date'].min().strftime("%Y-%m-%d")
        period_end = df['date'].max().strftime("%Y-%m-%d")
        
        return {
            "success": True,
            "data": {
                "hourly_average": round(hourly_average, 1),
                "daily_average": round(daily_average, 1),
                "period_average": round(period_average, 1),
                "max_utilization": round(max_utilization, 1),
                "min_utilization": round(min_utilization, 1),
                "period": {
                    "start_date": period_start,
                    "end_date": period_end
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得統計數據時發生錯誤: {str(e)}")


@app.get("/api/gpu/hourly-usage")
async def get_hourly_usage(
    date: str = Query(..., description="指定日期 (YYYY-MM-DD)"),
    gpu_id: Optional[int] = Query(None, description="GPU ID")
):
    """取得指定日期的每小時使用率"""
    try:
        df = load_csv_data()
        
        # 過濾指定日期
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
        df = df[df['date'] == target_date]
        df = filter_by_gpu_id(df, gpu_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"在 {date} 沒有找到資料")
        
        # 計算每小時平均使用率
        hourly_data = df.groupby('hour')['utilization_gpu'].mean().reset_index()
        
        # 建立 24 小時完整資料（0-23）
        chart_data = []
        for hour in range(24):
            utilization = 0.0
            hour_data = hourly_data[hourly_data['hour'] == hour]
            if not hour_data.empty:
                utilization = round(hour_data['utilization_gpu'].iloc[0], 1)
            
            chart_data.append({
                "hour": hour,
                "utilization": utilization
            })
        
        return {
            "success": True,
            "data": {
                "chart_data": chart_data,
                "date": date
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得每小時使用率時發生錯誤: {str(e)}")


@app.get("/api/gpu/daily-usage")
async def get_daily_usage(
    start_date: str = Query(..., description="開始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="結束日期 (YYYY-MM-DD)"),
    gpu_id: Optional[int] = Query(None, description="GPU ID")
):
    """取得指定期間的每日使用率"""
    try:
        df = load_csv_data()
        df = filter_by_date_range(df, start_date, end_date)
        df = filter_by_gpu_id(df, gpu_id)
        
        # 建立指定期間的完整日期範圍
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # 產生所有日期
        date_range = []
        current_date = start_date_obj
        while current_date <= end_date_obj:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # 計算每日的最高最低使用率（如果有資料的話）
        daily_stats = None
        if not df.empty:
            daily_stats = df.groupby('date')['utilization_gpu'].agg(['min', 'max']).reset_index()
        
        chart_data = []
        for date in date_range:
            min_util = 0.0
            max_util = 0.0
            
            # 如果有資料，則使用計算出的值
            if daily_stats is not None:
                day_data = daily_stats[daily_stats['date'] == date]
                if not day_data.empty:
                    min_val = day_data['min'].iloc[0]
                    max_val = day_data['max'].iloc[0]
                    # 處理 NaN 值，將其轉換為 0
                    min_util = round(min_val, 1) if pd.notna(min_val) else 0.0
                    max_util = round(max_val, 1) if pd.notna(max_val) else 0.0
            
            chart_data.append({
                "date": date.strftime("%m/%d"),
                "min_utilization": min_util,
                "max_utilization": max_util
            })
        
        return {
            "success": True,
            "data": {
                "chart_data": chart_data
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得每日使用率時發生錯誤: {str(e)}")


@app.get("/api/gpu/list")
async def get_gpu_list():
    """取得 GPU 清單"""
    try:
        df = load_csv_data()
        
        # 取得所有獨特的 GPU
        gpu_list = df[['gpu_id', 'gpu_name']].drop_duplicates().reset_index(drop=True)
        
        result = []
        for _, row in gpu_list.iterrows():
            result.append({
                "gpu_id": int(row['gpu_id']),
                "gpu_name": row['gpu_name'],
                "status": "active"
            })
        
        return {
            "success": True,
            "data": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得 GPU 清單時發生錯誤: {str(e)}")


@app.get("/api/gpu/realtime")
async def get_realtime_data():
    """取得最新的即時 GPU 資料"""
    try:
        df = load_csv_data()
        
        # 取得最新的資料記錄
        latest_record = df.loc[df['timestamp'].idxmax()]
        
        # 計算記憶體使用百分比
        memory_used_percent = (latest_record['memory_used'] / latest_record['memory_total']) * 100
        
        return {
            "success": True,
            "data": {
                "timestamp": latest_record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                "gpu_id": int(latest_record['gpu_id']),
                "utilization_gpu": float(latest_record['utilization_gpu']),
                "temperature": int(latest_record['temperature']),
                "memory_used_percent": round(memory_used_percent, 1)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得即時資料時發生錯誤: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)