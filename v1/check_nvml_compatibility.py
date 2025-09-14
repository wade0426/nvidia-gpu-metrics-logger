#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVML 相容性檢查腳本

檢查當前安裝的 nvidia-ml-py3 版本支援哪些功能
"""

import sys

def check_nvml_compatibility():
    """檢查 NVML 相容性"""
    print("=== NVML 相容性檢查 ===")
    
    try:
        import pynvml
        print(f"✓ nvidia-ml-py3 已安裝")
        
        # 檢查版本
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
            print(f"  NVIDIA 驅動版本: {version.decode('utf-8')}")
        except:
            print("  NVIDIA 驅動版本: 無法取得")
        
        # 檢查基本功能
        basic_functions = [
            'nvmlInit',
            'nvmlShutdown',
            'nvmlDeviceGetCount',
            'nvmlDeviceGetHandleByIndex',
            'nvmlDeviceGetName'
        ]
        
        print("\n基本功能支援:")
        for func in basic_functions:
            if hasattr(pynvml, func):
                print(f"  ✓ {func}")
            else:
                print(f"  ❌ {func}")
        
        # 檢查 GPU 指標功能
        gpu_functions = [
            'nvmlDeviceGetUtilizationRates',
            'nvmlDeviceGetMemoryInfo',
            'nvmlDeviceGetTemperature',
            'nvmlDeviceGetPowerUsage',
            'nvmlDeviceGetPowerManagementLimitDefault',
            'nvmlDeviceGetPowerManagementLimit',
            'nvmlDeviceGetFanSpeed'
        ]
        
        print("\nGPU 指標功能支援:")
        for func in gpu_functions:
            if hasattr(pynvml, func):
                print(f"  ✓ {func}")
            else:
                print(f"  ❌ {func}")
        
        # 嘗試初始化
        try:
            pynvml.nvmlInit()
            print("\n✓ NVML 初始化成功")
            
            # 檢查 GPU 數量
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"  偵測到 {gpu_count} 個 GPU")
                
                if gpu_count > 0:
                    # 檢查第一個 GPU 的詳細資訊
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    print(f"  第一個 GPU: {gpu_name}")
                    
                    # 測試各種指標
                    print("\n  指標測試:")
                    
                    # 使用率
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        print(f"    ✓ GPU 使用率: {util.gpu}%")
                        print(f"    ✓ 記憶體使用率: {util.memory}%")
                    except Exception as e:
                        print(f"    ❌ 使用率查詢失敗: {e}")
                    
                    # 記憶體
                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        total_mb = mem.total // (1024 * 1024)
                        print(f"    ✓ 記憶體總量: {total_mb} MB")
                    except Exception as e:
                        print(f"    ❌ 記憶體查詢失敗: {e}")
                    
                    # 溫度
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        print(f"    ✓ GPU 溫度: {temp}°C")
                    except Exception as e:
                        print(f"    ❌ 溫度查詢失敗: {e}")
                    
                    # 功耗
                    try:
                        if hasattr(pynvml, 'nvmlDeviceGetPowerUsage'):
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                            print(f"    ✓ 當前功耗: {power:.1f}W")
                        else:
                            print("    ❌ 功耗查詢不支援")
                    except Exception as e:
                        print(f"    ❌ 功耗查詢失敗: {e}")
                    
                    # 風扇
                    try:
                        if hasattr(pynvml, 'nvmlDeviceGetFanSpeed'):
                            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                            print(f"    ✓ 風扇轉速: {fan}%")
                        else:
                            print("    ❌ 風扇轉速查詢不支援")
                    except Exception as e:
                        print(f"    ❌ 風扇轉速查詢失敗: {e}")
                
            except Exception as e:
                print(f"  ❌ GPU 偵測失敗: {e}")
            
            pynvml.nvmlShutdown()
            
        except Exception as e:
            print(f"\n❌ NVML 初始化失敗: {e}")
            print("  請確認 NVIDIA 驅動程式已正確安裝")
        
    except ImportError:
        print("❌ nvidia-ml-py3 未安裝")
        print("  請執行: pip install nvidia-ml-py3")
        return False
    
    return True

def main():
    """主函數"""
    success = check_nvml_compatibility()
    
    print("\n=== 建議 ===")
    if success:
        print("✓ 相容性檢查完成，可以嘗試運行 GPU 監控")
        print("  執行: python run_monitor.py --test")
    else:
        print("❌ 相容性檢查失敗，請先解決上述問題")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
