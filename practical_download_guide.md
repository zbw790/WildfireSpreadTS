# WildfireSpreadTS 实用下载方案

## 推荐的分步骤方案

### 阶段1: 快速开始 (立即可做)
```
1. 浏览器下载: 直接访问 https://zenodo.org/records/8006177
2. 点击 "Download" 按钮下载 WildfireSpreadTS.zip (48GB)
3. 下载文档: WildfireSpreadTS_Documentation.pdf (8MB)
4. 解压到 data/raw/ 目录
```

### 阶段2: 数据探索 (TIF格式足够)
```
# 无需安装复杂环境，只需基本包
pip install rasterio matplotlib pandas

# 直接读取TIF文件进行数据探索
# 查看数据结构、统计信息、可视化等
```

### 阶段3: 决定是否转换 (基于实际使用体验)
```
如果发现加载速度可接受 → 继续使用TIF
如果加载速度太慢 → 转换部分数据为HDF5测试
如果确定要大规模训练 → 全部转换为HDF5
```

## 存储空间优化策略

### 策略1: 逐年处理
```
1. 只下载2018年数据先试验 (~12GB)
2. 验证流程后再下载其他年份
3. 处理完一年删除一年的原始数据
```

### 策略2: 特定事件
```
1. 从documentation找到最大的几个火灾事件
2. 只下载这些事件的数据进行研究
3. 避免下载全部数据
```

### 策略3: 渐进转换
```
# 边用边转换，避免占用过多空间
for year in [2018, 2019, 2020, 2021]:
    convert_to_hdf5(f"data/raw/{year}")
    train_on_hdf5(f"data/processed/{year}")
    delete_raw_tif(f"data/raw/{year}")  # 释放空间
```

## 立即开始的最简方案

### Windows用户
```
1. 打开浏览器访问: https://zenodo.org/records/8006177
2. 点击 "WildfireSpreadTS.zip" 旁边的 Download 按钮
3. 等待下载完成 (可能需要2-6小时)
4. 右键zip文件 → 解压到当前文件夹
5. 开始数据探索！
```

### 验证下载完整性
```powershell
# 检查文件大小应为 48,359,369,821 字节 (约48.36GB)
Get-ChildItem "WildfireSpreadTS.zip" | Select-Object Name, Length
```

## 何时需要转换HDF5？

### 继续使用TIF的情况：
- 只做数据分析和可视化
- 偶尔训练小模型
- 存储空间紧张
- 不在意加载时间

### 必须转换HDF5的情况：
- 大规模深度学习训练
- 需要频繁读取数据
- 追求最佳性能
- 有足够存储空间 (150GB+) 