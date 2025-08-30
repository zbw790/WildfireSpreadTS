# WildfireSpreadTS 数据集下载指南

## 快速下载 (推荐)

### Windows PowerShell
```powershell
# 创建下载目录
mkdir data\raw

# 下载主数据集 (48.36 GB)
Invoke-WebRequest -Uri "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1" -OutFile "data\raw\WildfireSpreadTS.zip"

# 下载文档 (7.84 MB)
Invoke-WebRequest -Uri "https://zenodo.org/records/8006177/files/WildfireSpreadTS_Documentation.pdf?download=1" -OutFile "data\raw\WildfireSpreadTS_Documentation.pdf"
```

### Linux/macOS
```bash
# 创建下载目录
mkdir -p data/raw

# 下载主数据集 (48.36 GB)
wget -O "data/raw/WildfireSpreadTS.zip" "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1"

# 下载文档 (7.84 MB)  
wget -O "data/raw/WildfireSpreadTS_Documentation.pdf" "https://zenodo.org/records/8006177/files/WildfireSpreadTS_Documentation.pdf?download=1"
```

### 使用curl (跨平台)
```bash
# 下载主数据集
curl -L -o "data/raw/WildfireSpreadTS.zip" "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1"

# 下载文档
curl -L -o "data/raw/WildfireSpreadTS_Documentation.pdf" "https://zenodo.org/records/8006177/files/WildfireSpreadTS_Documentation.pdf?download=1"
```

## 下载完成后的处理

### 1. 解压数据
```bash
# 解压到data/raw目录
unzip data/raw/WildfireSpreadTS.zip -d data/raw/
```

### 2. 检查数据完整性
```bash
# 检查MD5校验和
# 期望值: dc1a04e63ccc70037b277d585b8fe761
md5sum data/raw/WildfireSpreadTS.zip
```

### 3. 转换为HDF5格式 (强烈推荐)
```bash
# 安装必要的Python包
pip install h5py rasterio

# 转换数据格式 (需要约100GB额外空间)
python src/preprocess/CreateHDF5Dataset.py --data_dir data/raw/WildfireSpreadTS --target_dir data/processed/
```

## 重要提示

- **存储空间**: 确保至少有150GB可用空间 (原始48GB + HDF5约100GB)
- **下载时间**: 48GB文件可能需要几小时，建议使用稳定网络
- **校验完整性**: 下载后务必检查文件大小和MD5值
- **格式转换**: HDF5转换会显著提高后续训练速度

## 数据集结构预览

```
WildfireSpreadTS/
├── 2018/
│   ├── fire_event_001/
│   │   ├── 2018-01-15_features.tif
│   │   ├── 2018-01-16_features.tif
│   │   └── ...
│   └── fire_event_002/
├── 2019/
├── 2020/
└── 2021/
```

每个TIF文件包含40个特征通道：
- VIIRS遥感数据 (3通道)
- 植被指数 (NDVI, EVI2)
- 气象数据 (降水、温度、风速等)
- 地形数据 (坡度、坡向、海拔)
- 土地覆盖类型 (17类one-hot编码)
- 活跃火点检测数据 