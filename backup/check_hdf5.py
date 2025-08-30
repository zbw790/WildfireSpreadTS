import h5py
import os

print("=== HDF5 文件分析 ===\n")

# 检查每年的HDF5文件
years = [2018, 2019, 2020, 2021]
for year in years:
    hdf5_path = f"data/processed/{year}/data.hdf5"
    if os.path.exists(hdf5_path):
        print(f"--- {year}年 ---")
        with h5py.File(hdf5_path, 'r') as f:
            print(f"文件大小: {os.path.getsize(hdf5_path)/1024/1024:.1f} MB")
            print(f"HDF5 keys: {list(f.keys())}")
            if 'data' in f:
                print(f"Data shape: {f['data'].shape}")
                print(f"Data type: {f['data'].dtype}")
                print(f"Attributes: {dict(f['data'].attrs)}")
        print()
    else:
        print(f"--- {year}年 ---")
        print("HDF5文件不存在")
        print()

print("=== 原始数据统计 ===")
for year in years:
    data_path = f"data/{year}"
    if os.path.exists(data_path):
        fire_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        print(f"{year}年: {len(fire_dirs)} 个火灾事件")
    else:
        print(f"{year}年: 数据目录不存在") 