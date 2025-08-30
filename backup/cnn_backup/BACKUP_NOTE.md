# CNN模型备份说明

## 备份时间
2025-01-30

## 备份原因
CNN模型在训练过程中遇到技术问题（数据预处理和模型架构问题），决定重新开始实现。

## 备份内容

### 主要Python文件
- `train_wildfire_cnn.py` - 主训练脚本
- `test_cnn_framework.py` - 框架测试脚本
- `debug_data.py` - 数据诊断脚本

### 模型相关文件
- `wildfire_cnn_model.py` - CNN模型架构（U-Net + ConvLSTM）
- `wildfire_cnn_dataloader.py` - 数据加载器
- `wildfire_losses.py` - 损失函数实现
- `wildfire_metrics.py` - 评估指标
- `wildfire_preprocessing.py` - 数据预处理

### 配置文件
- `wildfire_cnn_config.json` - 训练配置

### 文档
- `CNN_MODELING_README.md` - CNN建模框架说明

### 输出目录
- `wildfire_cnn/` - 主要训练输出
- `debug_run/` - 调试运行输出
- `test_run/` - 测试运行输出

### 缓存文件
- `*.pyc` - Python编译缓存文件

## 遇到的主要问题
1. 输入数据包含NaN值
2. 土地覆盖分类特征被错误标准化
3. 模型输出包含NaN导致损失计算失败
4. 数据增强中的negative stride问题

## 下一步计划
重新设计更简洁的CNN实现，专注于解决数据预处理和模型稳定性问题。 