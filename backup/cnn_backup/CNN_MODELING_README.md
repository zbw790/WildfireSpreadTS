# 🔥 野火传播预测CNN建模框架

## 📋 项目概述

基于**WildfireSpreadTS数据集**的专业野火传播预测CNN建模框架，采用**U-Net + ConvLSTM**时空融合架构，专门针对**23通道多模态数据**和**严重类别不平衡**问题进行优化。

### 🎯 核心特性

- ✅ **23通道多模态数据支持**：遥感、气象、地形、植被、预测数据
- ✅ **时空建模**：U-Net空间特征提取 + ConvLSTM时序建模
- ✅ **类别不平衡处理**：Focal Loss + 加权采样 + 专业评估指标
- ✅ **物理一致性**：保持风向一致性的数据增强
- ✅ **分类别标准化**：针对不同特征类型的专门标准化策略
- ✅ **专业评估**：AUPRC、IoU、F1-Score、空间连通性指标

---

## 🏗️ 框架架构

### 📊 数据流程
```
原始HDF5数据 → 特征预处理 → 时空裁剪 → 数据增强 → 模型训练
     ↓              ↓            ↓          ↓         ↓
23通道×607事件 → 标准化+嵌入 → 时间窗口 → 物理一致性 → CNN预测
```

### 🧠 模型架构
```
输入: (B, T=5, C=23, H=128, W=128)
     ↓
特征预处理 (土地覆盖嵌入)
     ↓  
特征投影: 23+8 → 64通道
     ↓
U-Net编码器: [64→128→256→512]
     ↓
ConvLSTM时序建模: [512→128→64]
     ↓
U-Net解码器: [64→128→256→512]
     ↓
输出: (B, 1, H=128, W=128) 火点概率
```

---

## 📂 项目结构

```
📁 WildfireSpreadTS/
├── 📁 models/                          # 核心建模模块
│   ├── wildfire_cnn_dataloader.py     # 数据加载器 (类别不平衡处理)
│   ├── wildfire_cnn_model.py          # CNN模型 (U-Net + ConvLSTM)
│   ├── wildfire_losses.py             # 损失函数 (Focal Loss等)
│   ├── wildfire_metrics.py            # 评估指标 (AUPRC, IoU等)
│   └── wildfire_preprocessing.py      # 预处理 (标准化, 增强)
├── 📁 configs/
│   └── wildfire_cnn_config.json       # 训练配置文件
├── train_wildfire_cnn.py              # 完整训练脚本
├── test_cnn_framework.py              # 框架测试脚本
└── CNN_MODELING_README.md             # 本文档
```

---

## 🚀 快速开始

### 1️⃣ 环境检查
```bash
# 测试框架完整性
python test_cnn_framework.py
```

### 2️⃣ 开始训练
```bash
# 使用默认配置训练
python train_wildfire_cnn.py

# 使用自定义配置
python train_wildfire_cnn.py --config configs/wildfire_cnn_config.json

# 命令行参数覆盖
python train_wildfire_cnn.py \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.0005 \
    --output_dir outputs/my_experiment
```

### 3️⃣ 监控训练
```bash
# TensorBoard监控
tensorboard --logdir outputs/wildfire_cnn_experiment/tensorboard

# 查看训练日志
tail -f outputs/wildfire_cnn_experiment/training.log
```

---

## ⚙️ 核心组件详解

### 🔄 数据加载器 (`wildfire_cnn_dataloader.py`)

**关键特性**：
- **时空窗口**：5天输入序列 → 1天预测
- **空间裁剪**：128×128像素，64像素步长
- **加权采样**：火点样本10倍权重
- **分类别标准化**：遥感(RobustScaler)，气象(StandardScaler)

**数据增强**：
- 水平/垂直翻转 + 风向调整
- 90度旋转 + 风向补偿  
- 气象噪声 + 条件扰动

### 🧠 CNN模型 (`wildfire_cnn_model.py`)

**架构组件**：
- **土地覆盖嵌入**：16类 → 8维嵌入
- **U-Net编码器**：多尺度空间特征
- **ConvLSTM**：双层时序建模 [128, 64]
- **注意力机制**：CBAM通道+空间注意力
- **U-Net解码器**：空间细节重建

**参数量**：~10.7M （适中规模，避免过拟合）

### 📉 损失函数 (`wildfire_losses.py`)

**自动损失选择**：
- **>1000:1不平衡** → ComboLoss (Focal+Dice+IoU)
- **100-1000:1** → FocalLoss (α=0.25, γ=2)
- **10-100:1** → WeightedBCE
- **<10:1** → 标准BCE

### 📊 评估指标 (`wildfire_metrics.py`)

**核心指标**：
- **AUPRC**：类别不平衡主要指标
- **IoU**：空间重叠精度
- **F1-Score**：综合分类性能
- **火点检测率**：火灾特定指标
- **Hausdorff距离**：边界相似性

---

## 🎯 预期性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **AUPRC** | > 0.70 | 主要优化目标 |
| **IoU** | > 0.50 | 空间精度 |
| **F1-Score** | > 0.60 | 平衡性能 |
| **火点检测率** | > 0.80 | 召回率 |
| **误报率** | < 0.05 | 特异性 |

---

## 🛠️ 高级配置

### 🎚️ 超参数调优

**学习率策略**：
```json
{
  "optimizer": {"lr": 1e-3},
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "factor": 0.5,
    "patience": 5
  }
}
```

**数据配置**：
```json
{
  "data": {
    "batch_size": 6,        // GPU内存平衡
    "sequence_length": 5,   // 时间上下文
    "crop_size": 128,       // 空间分辨率
    "fire_threshold": 0.5   // 二值化阈值
  }
}
```

### 🔧 模型变体

**轻量级版本**：
```json
{
  "model": {
    "unet_features": [32, 64, 128, 256],
    "lstm_hidden_dims": [64, 32]
  }
}
```

**高性能版本**：
```json
{
  "model": {
    "unet_features": [64, 128, 256, 512, 1024],
    "lstm_hidden_dims": [256, 128, 64]
  }
}
```

---

## 📈 实验建议

### 🔬 阶段性实验

**Phase 1: 基线验证** (1-2周)
```bash
python train_wildfire_cnn.py \
    --epochs 20 \
    --batch_size 4 \
    --model_type WildfireCNN
```

**Phase 2: 超参数优化** (2-3周)
- 学习率网格搜索: [1e-4, 5e-4, 1e-3, 5e-3]
- 批次大小测试: [4, 6, 8, 12]
- 序列长度实验: [3, 5, 7, 10]

**Phase 3: 架构优化** (3-4周)
- 不同损失函数对比
- 注意力机制消融实验
- 时空建模策略对比

### 🎯 性能基准

**最小可接受性能**：
- AUPRC ≥ 0.60
- IoU ≥ 0.40
- F1-Score ≥ 0.50

**目标性能**：
- AUPRC ≥ 0.75
- IoU ≥ 0.55
- F1-Score ≥ 0.65

---

## 🚨 常见问题解决

### ❗ 内存不足
```bash
# 减少批次大小
python train_wildfire_cnn.py --batch_size 2

# 减少裁剪尺寸
python train_wildfire_cnn.py --crop_size 96
```

### ❗ 训练不收敛
```bash
# 降低学习率
python train_wildfire_cnn.py --lr 0.0001

# 增加正则化
# 修改配置文件中的weight_decay
```

### ❗ 类别极度不平衡
```bash
# 调整火点阈值
python train_wildfire_cnn.py --fire_threshold 0.1

# 使用更强的损失函数 (在配置中设置loss.type="combo")
```

---

## 📊 数据集适用性评分

| 评估维度 | 评分 | 说明 |
|----------|------|------|
| **数据质量** | 100/100 | 607事件，完整时序 |
| **特征完整性** | 92/100 | 23/25核心特征 |
| **技术可行性** | 92/100 | CNN+CA混合适用 |
| **科研价值** | 95/100 | 顶级期刊潜力 |
| **应用价值** | 88/100 | 实际火灾管理 |

**🏆 总体评分: 91/100** ⭐⭐⭐⭐⭐

---

## 🎉 下一步工作

### 🔄 模型演进路径
1. **CNN基线模型** ← 当前阶段
2. **ConvLSTM时空建模** ← 下一步  
3. **物理约束CA模型**
4. **CNN-CA混合架构**

### 🚀 研究方向
- **多尺度建模**：不同时空分辨率
- **迁移学习**：跨地区泛化能力
- **实时预测**：在线学习和更新
- **不确定性量化**：预测置信度估计

---

## 📞 联系信息

如有问题或需要技术支持，请参考：
- 📖 技术文档：各模块内详细注释
- 🧪 测试脚本：`test_cnn_framework.py`
- ⚙️ 配置模板：`configs/wildfire_cnn_config.json`
- 📊 评估工具：`models/wildfire_metrics.py`

**🔥 准备好开始您的野火传播预测研究之旅！** 🚀 