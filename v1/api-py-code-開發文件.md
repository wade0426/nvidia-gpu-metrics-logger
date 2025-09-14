需要以下 API 端點和資料格式來支援前端功能：

## **核心 API 端點設計**

### **1. 統計數據 API**
**端點**: `POST /api/gpu/statistics`
**參數**: 
- `start_date`: 開始日期 (可選)
- `end_date`: 結束日期 (可選)
- `gpu_id`: GPU ID (可選)

**注意**: 當所有參數都是可選時，即使不需要任何參數，也必須傳送一個空的 JSON 物件 `{}`

**回傳格式**:
```json
{
  "success": true,
  "data": {
    "hourly_average": 50.0,
    "daily_average": 46.0, 
    "period_average": 54.0,
    "max_utilization": 86.0,
    "min_utilization": 42.0,
    "period": {
      "start_date": "2025-06-15",
      "end_date": "2025-06-21"
    }
  }
}
```

hourly_average: 每小時平均使用率
daily_average: 每日平均使用率
period_average: 期間平均使用率
max_utilization: 最高使用率
min_utilization: 最低使用率

### **2. 每小時使用率 API**
**端點**: `POST /api/gpu/hourly-usage`
**參數**: 
- `date`: 指定日期 
- `gpu_id`: GPU ID (可選)

**回傳格式**:
```json
{
  "success": true,
  "data": {
    "chart_data": [
      {"hour": 0, "utilization": 0.0},
      {"hour": 1, "utilization": 0.0},
      {"hour": 2, "utilization": 0.0},
      {"hour": 3, "utilization": 0.0},
      {"hour": 4, "utilization": 0.0},
      {"hour": 5, "utilization": 0.0},
      {"hour": 6, "utilization": 0.0},
      {"hour": 7, "utilization": 0.0},
      {"hour": 8, "utilization": 0.0},
      {"hour": 9, "utilization": 0.0},
      {"hour": 10, "utilization": 25.0},
      {"hour": 11, "utilization": 30.0},
      {"hour": 12, "utilization": 55.0},
      {"hour": 13, "utilization": 75.0},
      {"hour": 14, "utilization": 95.0},
      {"hour": 15, "utilization": 85.0},
      {"hour": 16, "utilization": 60.0},
      {"hour": 17, "utilization": 45.0},
      {"hour": 18, "utilization": 0.0},
      {"hour": 19, "utilization": 0.0},
      {"hour": 20, "utilization": 0.0},
      {"hour": 21, "utilization": 0.0},
      {"hour": 22, "utilization": 0.0},
      {"hour": 23, "utilization": 0.0}
    ],
    "date": "2025-08-27"
  }
}
```

### **3. 每日使用率 API**
**端點**: `POST /api/gpu/daily-usage`
**參數**: 
- `start_date`: 開始日期
- `end_date`: 結束日期
- `gpu_id`: GPU ID (可選)

**描述**: 取得指定期間內每日的最高最低使用率。API 會為指定期間內的所有日期提供資料，對於沒有資料的日期，會預設顯示 0 值。

**回傳格式**:
```json
{
  "success": true,
  "data": {
    "chart_data": [
      {
        "date": "6/15",
        "min_utilization": 28.0,
        "max_utilization": 52.0
      },
      {
        "date": "6/16", 
        "min_utilization": 48.0,
        "max_utilization": 52.0
      },
      {
        "date": "6/17",
        "min_utilization": 72.0,
        "max_utilization": 78.0
      },
      {
        "date": "6/18",
        "min_utilization": 18.0,
        "max_utilization": 30.0
      },
      {
        "date": "6/19",
        "min_utilization": 48.0,
        "max_utilization": 65.0
      },
      {
        "date": "6/20",
        "min_utilization": 38.0,
        "max_utilization": 56.0
      },
      {
        "date": "6/21",
        "min_utilization": 32.0,
        "max_utilization": 32.0
      }
    ]
  }
}
```

## **資料處理邏輯**

### **後端資料聚合需求**：

1. **每小時聚合**：
   - 計算每小時的平均 GPU 使用率
   - 沒有資料的小時會顯示 0 值

2. **每日聚合**：
   - 將資料按日期分組
   - 計算每日的最高/最低使用率
   - 對於沒有資料的日期，會預設顯示 0 值
   - API 會為指定期間內的所有日期提供完整資料

3. **統計計算**：
   - **每小時平均**：所有小時平均值的平均
   - **每日平均**：所有日期平均值的平均
   - **期間平均**：整個時間範圍的總平均
   - **最高/最低**：期間內的極值

## **輔助 API**

### **4. GPU 清單 API**
**端點**: `POST /api/gpu/list`
**參數**: 無

**注意**: 此 API 不需要任何參數，但仍需傳送一個空的 JSON 物件 `{}`
**回傳格式**:
```json
{
  "success": true,
  "data": [
    {
      "gpu_id": 0,
      "gpu_name": "NVIDIA GeForce RTX 4090",
      "status": "active"
    }
  ]
}
```

### **5. 即時資料 API** 
**端點**: `POST /api/gpu/realtime`
**參數**: 無

**注意**: 此 API 不需要任何參數，但仍需傳送一個空的 JSON 物件 `{}`
**回傳格式**:
```json
{
  "success": true,
  "data": {
    "timestamp": "2025-08-27 18:47:35",
    "gpu_id": 0,
    "utilization_gpu": 0.0,
    "temperature": 37,
    "memory_used_percent": 96.8
  }
}
```