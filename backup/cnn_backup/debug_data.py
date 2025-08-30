"""
诊断脚本：检查HDF5数据的状态
"""

import h5py
import numpy as np
import glob
import os

def diagnose_hdf5_data():
    """诊断HDF5数据文件"""
    data_dir = "data/processed"
    
    # 找到HDF5文件
    hdf5_files = glob.glob(os.path.join(data_dir, "**", "*.hdf5"), recursive=True)
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    if not hdf5_files:
        print("没有找到HDF5文件!")
        return
    
    # 检查第一个文件
    hdf5_file = hdf5_files[0]
    print(f"\n检查文件: {hdf5_file}")
    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"文件中的键: {list(f.keys())}")
        
        # 检查第一个事件
        fire_keys = list(f.keys())
        if fire_keys:
            first_fire = fire_keys[0]
            print(f"\n检查第一个火灾事件: {first_fire}")
            
            data = f[first_fire]['data'][:]
            print(f"数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            
            # 检查NaN值
            nan_count = np.isnan(data).sum()
            total_count = data.size
            print(f"NaN值数量: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)")
            
            if nan_count > 0:
                nan_per_channel = np.isnan(data).sum(axis=(0, 2, 3))
                print("每个通道的NaN数量:")
                for i, count in enumerate(nan_per_channel):
                    if count > 0:
                        print(f"  通道 {i}: {count}")
            
            # 检查inf值
            inf_count = np.isinf(data).sum()
            print(f"Inf值数量: {inf_count}")
            
            # 检查每个通道的值范围
            print("\n通道值范围:")
            for i in range(data.shape[1]):
                channel_data = data[:, i, :, :]
                valid_data = channel_data[np.isfinite(channel_data)]
                if len(valid_data) > 0:
                    print(f"  通道 {i}: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
                else:
                    print(f"  通道 {i}: 没有有效数据")
            
            # 特别检查土地覆盖通道(16)
            print(f"\n土地覆盖通道(16)详细分析:")
            landcover_data = data[:, 16, :, :]
            valid_lc = landcover_data[np.isfinite(landcover_data)]
            if len(valid_lc) > 0:
                unique_values = np.unique(valid_lc)
                print(f"  唯一值数量: {len(unique_values)}")
                print(f"  值范围: [{unique_values.min():.2f}, {unique_values.max():.2f}]")
                print(f"  前10个值: {unique_values[:10]}")
                
                # 检查是否在期望的1-16范围内
                expected_range = np.all((unique_values >= 1) & (unique_values <= 16))
                print(f"  是否在1-16范围: {expected_range}")
            
            # 特别检查火点通道(22)
            print(f"\n火点通道(22)详细分析:")
            fire_data = data[:, 22, :, :]
            valid_fire = fire_data[np.isfinite(fire_data)]
            if len(valid_fire) > 0:
                print(f"  值范围: [{valid_fire.min():.2f}, {valid_fire.max():.2f}]")
                print(f"  平均值: {valid_fire.mean():.2f}")
                fire_positive = (valid_fire > 0.5).sum()
                print(f"  火点像素(>0.5): {fire_positive} / {len(valid_fire)} ({fire_positive/len(valid_fire)*100:.2f}%)")

if __name__ == "__main__":
    diagnose_hdf5_data() 