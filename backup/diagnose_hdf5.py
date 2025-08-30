"""
HDF5文件结构诊断脚本
检查WildfireSpreadTS数据集的HDF5文件结构
"""

import h5py
import numpy as np
import glob
import os
from pathlib import Path

def diagnose_hdf5_structure():
    """诊断HDF5文件结构"""
    data_dir = Path("data/processed")
    
    # 找到第一个HDF5文件
    hdf5_files = list(data_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        print("❌ 未找到HDF5文件")
        return
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 检查前几个文件的结构
    for i, file_path in enumerate(hdf5_files[:5]):
        print(f"\n{'='*60}")
        print(f"检查文件 {i+1}: {file_path}")
        print('='*60)
        
        try:
            with h5py.File(file_path, 'r') as f:
                print(f"文件根目录键: {list(f.keys())}")
                
                # 递归打印HDF5结构
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj)}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"    - 形状: {obj.shape}")
                        print(f"    - 数据类型: {obj.dtype}")
                        if hasattr(obj, 'attrs'):
                            attrs = dict(obj.attrs)
                            if attrs:
                                print(f"    - 属性: {attrs}")
                
                print("\n文件结构:")
                f.visititems(print_structure)
                
                # 尝试不同的数据读取方式
                print("\n尝试数据读取:")
                
                for key in f.keys():
                    print(f"\n尝试读取键: {key}")
                    try:
                        obj = f[key]
                        print(f"  对象类型: {type(obj)}")
                        
                        if isinstance(obj, h5py.Dataset):
                            data = obj[:]
                            print(f"  数据形状: {data.shape}")
                            print(f"  数据类型: {data.dtype}")
                            print(f"  数据范围: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
                            break
                        elif isinstance(obj, h5py.Group):
                            print(f"  组中的键: {list(obj.keys())}")
                            for subkey in obj.keys():
                                try:
                                    subobj = obj[subkey]
                                    print(f"    {subkey}: {type(subobj)}")
                                    if isinstance(subobj, h5py.Dataset):
                                        subdata = subobj[:]
                                        print(f"      形状: {subdata.shape}")
                                        print(f"      类型: {subdata.dtype}")
                                        if len(subdata.shape) > 0:
                                            print(f"      范围: [{np.nanmin(subdata):.2f}, {np.nanmax(subdata):.2f}]")
                                except Exception as e:
                                    print(f"      读取 {subkey} 时出错: {e}")
                    except Exception as e:
                        print(f"  读取 {key} 时出错: {e}")
                
        except Exception as e:
            print(f"❌ 打开文件时出错: {e}")
        
        if i >= 2:  # 只检查前3个文件
            break

def test_alternative_reading():
    """测试其他数据读取方式"""
    data_dir = Path("data/processed")
    hdf5_files = list(data_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        return
    
    print(f"\n{'='*60}")
    print("测试其他数据读取方式")
    print('='*60)
    
    file_path = hdf5_files[0]
    print(f"测试文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"根键: {list(f.keys())}")
            
            # 方法1: 直接读取数据集
            if 'data' in f:
                try:
                    data = f['data'][:]
                    print(f"✅ 方法1成功 - 直接读取'data': {data.shape}")
                except Exception as e:
                    print(f"❌ 方法1失败: {e}")
            
            # 方法2: 遍历所有键寻找数据
            for key in f.keys():
                try:
                    obj = f[key]
                    if isinstance(obj, h5py.Dataset):
                        data = obj[:]
                        print(f"✅ 方法2成功 - 读取键'{key}': {data.shape}")
                        break
                    elif isinstance(obj, h5py.Group):
                        for subkey in obj.keys():
                            subobj = obj[subkey]
                            if isinstance(subobj, h5py.Dataset):
                                data = subobj[:]
                                print(f"✅ 方法2成功 - 读取'{key}/{subkey}': {data.shape}")
                                break
                except Exception as e:
                    print(f"❌ 方法2在键'{key}'失败: {e}")
            
            # 方法3: 使用原始项目的数据加载方式
            try:
                from src.dataloader.FireSpreadDataset import FireSpreadDataset
                print("✅ 可以导入原始数据加载器")
                
                # 创建数据集实例进行测试
                dataset = FireSpreadDataset(
                    data_dir=str(data_dir.parent),
                    years=[2018],
                    sequence_length=3,
                    prediction_horizon=1,
                    load_from_hdf5=True
                )
                print(f"✅ 原始数据加载器初始化成功")
                
                # 尝试获取一个样本
                sample = dataset[0]
                print(f"✅ 成功获取样本: {[s.shape if hasattr(s, 'shape') else type(s) for s in sample]}")
                
            except Exception as e:
                print(f"❌ 原始数据加载器测试失败: {e}")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    print("🔍 HDF5文件结构诊断")
    print("="*80)
    
    diagnose_hdf5_structure()
    test_alternative_reading()
    
    print("\n" + "="*80)
    print("诊断完成") 