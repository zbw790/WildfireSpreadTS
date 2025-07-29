# WildfireSpreadTS 完整工作流程

## 🚀 推荐的完整流程：下载 → 解压 → 转换

### 存储空间规划
```
需要空间: 约200GB (建议有250GB空余)
├── WildfireSpreadTS.zip      48GB  (可删除)
├── TIF原始文件               48GB  (保留备份)
└── HDF5转换文件            ~100GB  (训练使用)
```

## 第一步：下载ZIP文件

### 浏览器下载 (推荐)
```
1. 访问: https://zenodo.org/records/8006177
2. 点击: WildfireSpreadTS.zip [Download]
3. 保存到: E:\github\WildfireSpreadTS\data\raw\
4. 等待下载完成 (48.36GB)
```

### 验证下载完整性
```powershell
# 检查文件大小和MD5
cd E:\github\WildfireSpreadTS\data\raw
Get-ChildItem WildfireSpreadTS.zip | Select-Object Name, Length
# 应该显示: Length = 48359369821

# 如果有md5sum工具，验证校验和
# 期望值: dc1a04e63ccc70037b277d585b8fe761
```

## 第二步：解压ZIP文件

### Windows解压
```powershell
# 方法1: 使用PowerShell
Expand-Archive -Path "WildfireSpreadTS.zip" -DestinationPath "." -Force

# 方法2: 右键菜单
# 右键zip文件 → "解压到 WildfireSpreadTS\"
```

### 检查解压结果
```powershell
# 查看数据结构
tree /F WildfireSpreadTS
# 应该看到按年份组织的文件夹和TIF文件
```

## 第三步：安装转换依赖

### 最小化安装 (仅转换需要)
```bash
# 只安装必要的包用于转换
pip install h5py==3.7.0 rasterio tqdm numpy
```

### 检查转换脚本
```bash
# 确认转换脚本存在
ls src/preprocess/CreateHDF5Dataset.py
```

## 第四步：执行HDF5转换

### 转换命令
```bash
# 创建输出目录
mkdir data\processed

# 执行转换 (耗时可能几小时)
python src/preprocess/CreateHDF5Dataset.py --data_dir data/raw/WildfireSpreadTS --target_dir data/processed
```

### 转换过程监控
```
预期输出:
- 显示处理每个年份的进度
- 创建年份.hdf5文件在data/processed/
- 转换完成后约100GB
```

## 第五步：空间优化 (可选)

### 策略1: 删除ZIP文件
```bash
# 转换成功后可删除ZIP节省48GB
del data\raw\WildfireSpreadTS.zip
```

### 策略2: 分年转换 (节省空间)
```bash
# 如果空间不够，可以分年处理
for year in 2018 2019 2020 2021:
    python src/preprocess/CreateHDF5Dataset.py --data_dir data/raw/WildfireSpreadTS/$year --target_dir data/processed
    # 转换完成后删除该年的TIF文件
    rm -rf data/raw/WildfireSpreadTS/$year
```

### 策略3: 保留原始数据 (推荐)
```bash
# 如果空间充足，建议保留TIF文件作为备份
# TIF文件可用于：
# - 重新处理和验证
# - 不同的预处理实验
# - 与其他工具兼容
```

## 最终文件结构

```
E:\github\WildfireSpreadTS\
├── data/
│   ├── raw/
│   │   ├── WildfireSpreadTS.zip (可删除)
│   │   ├── WildfireSpreadTS/     (TIF原始文件 48GB)
│   │   └── WildfireSpreadTS_Documentation.pdf
│   └── processed/               (HDF5文件 ~100GB)
│       ├── 2018.hdf5
│       ├── 2019.hdf5  
│       ├── 2020.hdf5
│       └── 2021.hdf5
├── src/
└── cfgs/
```

## 验证转换结果

### 检查HDF5文件
```python
import h5py

# 验证HDF5文件
with h5py.File('data/processed/2018.hdf5', 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Data shape: {f['data'].shape}")
    print(f"Data type: {f['data'].dtype}")
```

### 性能对比测试
```python
import time
from src.dataloader.FireSpreadDataset import FireSpreadDataset

# 测试TIF加载速度
start = time.time()
dataset_tif = FireSpreadDataset(data_dir="data/raw/WildfireSpreadTS", 
                               load_from_hdf5=False, ...)
tif_time = time.time() - start

# 测试HDF5加载速度  
start = time.time()
dataset_hdf5 = FireSpreadDataset(data_dir="data/processed", 
                                load_from_hdf5=True, ...)
hdf5_time = time.time() - start

print(f"TIF加载时间: {tif_time:.2f}秒")
print(f"HDF5加载时间: {hdf5_time:.2f}秒") 
print(f"速度提升: {tif_time/hdf5_time:.1f}倍")
```

## 故障排除

### 常见问题
```
1. 转换中断: 重新运行转换命令，支持断点续传
2. 空间不足: 使用分年转换策略
3. 内存不足: 调整batch_size或重启机器
4. 权限问题: 确保对目录有写权限
```

### 性能优化
```
1. 使用SSD存储提高转换速度
2. 确保足够RAM (推荐16GB+)
3. 转换时关闭其他占用资源的程序
``` 