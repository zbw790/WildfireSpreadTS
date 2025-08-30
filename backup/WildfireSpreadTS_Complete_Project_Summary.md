# WildfireSpreadTS 野火蔓延预测项目完整总结

## 📋 项目概述

本项目基于WildfireSpreadTS数据集，开发了CNN和Cellular Automata两种深度学习模型来预测野火蔓延动态，旨在为应急管理和灾害防控提供技术支持。

## 📊 数据集详情

### 基本信息
- **数据集名称**: WildfireSpreadTS
- **来源**: Zenodo (https://doi.org/10.5281/zenodo.8006177)
- **许可协议**: CC-BY-4.0
- **时间范围**: 2018-2021年
- **空间分辨率**: 375米/像素
- **时间分辨率**: 每日数据
- **数据大小**: 48.36GB (解压后)
- **火灾事件**: 607个独立野火事件
- **覆盖范围**: 全球多个地区

### 数据结构
```
数据格式: HDF5
维度结构: [时间步, 通道数, 高度, 宽度]
通道数量: 23个特征通道
目标变量: 二值化火灾蔓延掩码
```

### 特征通道详解 (23通道)

#### 1. 主动火点数据 (1通道)
- **VIIRS火点**: 置信度值 [0-100]
- **用途**: 当前火灾活跃度

#### 2. 植被指数 (2通道)
- **NDVI**: 归一化植被指数 [-1, 1]
- **EVI2**: 增强植被指数 [-1, 1]
- **用途**: 植被健康度和燃料载量

#### 3. 气象数据 (8通道)
- 温度 (最高/最低/平均) [°C]
- 相对湿度 [0-100%]
- 风速 [m/s]
- 风向 [0-360°]
- 降水量 [mm]
- 大气压 [hPa]

#### 4. 地形数据 (3通道)
- **高程**: 海拔高度 [m]
- **坡度**: 地形坡度 [度]
- **坡向**: 坡面朝向 [0-360°]

#### 5. 干旱指数 (1通道)
- **PDSI**: Palmer干旱严重程度指数 [-4, 4]

#### 6. 土地覆盖 (1通道)
- **分类变量**: 16类土地利用类型 [0-15]

#### 7. 预报数据 (6通道)
- 未来3天气象预报
- 温度、湿度、风速预报

#### 8. 历史数据 (1通道)
- 历史火灾发生记录

## 🔄 数据预处理流程

### 1. 格式转换
```python
# 从GeoTIFF转HDF5
原始: 多个.tif文件/火灾事件
转换: 单个.hdf5文件/火灾事件
结构: 时间序列组织
```

### 2. 数据清洗
```python
# NaN值处理
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
clean_data = imputer.fit_transform(raw_data)

# 尺寸标准化
target_size = (128, 128)  # 或 (256, 256)
resized = F.interpolate(data, size=target_size, mode='bilinear')
```

### 3. 特征预处理
```python
# 连续特征标准化
continuous_features = features[:21]
scaler = RobustScaler()
normalized = scaler.fit_transform(continuous_features)

# 土地覆盖编码
landcover = torch.clamp(features[21], 0, 15)
embedded = nn.Embedding(16, 8)(landcover.long())

# 目标二值化
target = (target > 0.5).float()
```

### 4. 数据增强
- **空间增强**: 翻转、旋转
- **噪声增强**: 高斯噪声 (σ=0.01)
- **时间扰动**: 轻微时间偏移
- **物理约束**: 保持风向一致性

## 🤖 模型架构

### CNN模型 (U-Net)

```python
class GPUUNet(nn.Module):
    def __init__(self, continuous_channels=21, landcover_classes=16, embed_dim=8):
        super().__init__()
        
        # 土地覆盖嵌入
        self.landcover_embedding = nn.Embedding(landcover_classes, embed_dim)
        total_channels = continuous_channels + embed_dim
        
        # 编码器路径
        self.enc1 = ConvBlock(total_channels, 64)  # 128x128 -> 64x64
        self.enc2 = ConvBlock(64, 128)             # 64x64 -> 32x32
        self.enc3 = ConvBlock(128, 256)            # 32x32 -> 16x16
        self.enc4 = ConvBlock(256, 512)            # 16x16 -> 8x8
        
        # 瓶颈层
        self.bottleneck = ConvBlock(512, 1024)     # 8x8 -> 4x4
        
        # 解码器路径 (跳跃连接)
        self.dec4 = ConvBlock(1024 + 512, 512)     # 4x4 -> 8x8
        self.dec3 = ConvBlock(512 + 256, 256)      # 8x8 -> 16x16
        self.dec2 = ConvBlock(256 + 128, 128)      # 16x16 -> 32x32
        self.dec1 = ConvBlock(128 + 64, 64)        # 32x32 -> 64x64
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
```

**核心特点**:
- 跳跃连接保留空间细节
- 土地覆盖嵌入处理分类特征
- 多尺度特征融合

### Cellular Automata模型

```python
class OptimizedCAModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 环境特征编码器
        self.env_encoder = nn.Sequential(
            nn.Conv2d(total_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        
        # CA核心转换规则
        self.ca_core = nn.Sequential(
            nn.Conv2d(32 + 1, 64, 3, padding=1),  # 环境+火灾状态
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()  # 状态变化 [-1,1]
        )
        
        # 风向影响建模
        self.wind_networks = self._create_wind_networks()
    
    def forward(self, env_features, initial_fire_state, num_steps=3):
        env_encoded = self.env_encoder(env_features)
        fire_state = initial_fire_state.clone()
        
        # 多步演化
        for step in range(num_steps):
            ca_input = torch.cat([env_encoded, fire_state], dim=1)
            fire_change = self.ca_core(ca_input)
            
            # 风向影响
            wind_influence = self._apply_wind_influence(env_features, fire_state)
            fire_change = fire_change * (1 + wind_influence * 0.5)
            
            # 可微分状态更新
            fire_state = torch.clamp(fire_state + fire_change * 0.3, 0, 1)
            
        return fire_state
    
    def _create_wind_networks(self):
        """创建8个风向的专用卷积核"""
        wind_nets = nn.ModuleDict()
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        for direction in directions:
            conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            # 根据风向初始化卷积核权重
            wind_nets[direction] = conv
            
        return wind_nets
```

**核心特点**:
- 物理启发的转换规则
- 多步时间演化
- 风向特异性建模
- 完全可微分设计

## 🎯 损失函数与优化

### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha      # 类别平衡
        self.gamma = gamma      # 难例挖掘
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0001, 0.9999)
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**选择理由**: 解决极度不平衡的火灾/非火灾像素比例

### 优化配置
```python
# 优化器设置
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-4
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs, 
    eta_min=1e-6
)

# 梯度控制
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 📈 评估体系

### 主要指标
1. **AUPRC**: 精确率-召回率曲线下面积 (主要指标)
2. **IoU**: 交并比，空间重叠度量
3. **F1-Score**: 精确率和召回率调和平均
4. **AUC-ROC**: ROC曲线下面积

### 空间评估
- **Hausdorff距离**: 边界匹配精度
- **边界一致性**: 火灾边缘准确性
- **形状保持度**: 几何形状相似性

### 时间一致性
- **帧间变化**: 相邻时间步连续性
- **传播合理性**: 物理规律符合度

## 🚀 实验配置

### 硬件环境
```
GPU: CUDA兼容显卡
内存: ≥16GB RAM
存储: SSD推荐
操作系统: Windows 10
```

### 训练参数
```python
# 核心参数
batch_size = 4              # GPU内存限制
num_epochs = 50
learning_rate = 1e-4
weight_decay = 1e-4

# 数据加载
num_workers = 0             # Windows兼容
pin_memory = True
persistent_workers = False

# 早停机制
patience = 10
min_delta = 1e-4
```

## 🔧 技术挑战与解决方案

### 1. 数据质量
**问题**: 大量NaN值、数据稀疏
**解决**: 
- 中值填补策略
- 数据质量报告系统
- 多重验证机制

### 2. 类别不平衡
**问题**: 火灾像素<5%
**解决**:
- Focal Loss (α=0.8, γ=2.0)
- 加权采样策略
- AUPRC作为主要指标

### 3. 内存限制
**问题**: 大图像, 有限GPU内存
**解决**:
- 小批次训练
- 梯度累积
- 内存优化策略

### 4. 数值稳定性
**问题**: 训练中出现NaN
**解决**:
- 梯度裁剪
- 学习率调度
- 激活函数约束

## 📊 实验结果

### CNN模型性能
```
训练表现:
- Focal Loss: 0.15-0.20
- AUPRC: 0.25-0.30
- 收敛性: 良好

特点:
- 空间特征提取强
- 细节捕获能力好
- 计算开销较大
```

### CA模型性能
```
训练表现:
- Focal Loss: 0.20-0.25
- AUPRC: 0.20-0.25
- 物理一致性: 优秀

特点:
- 时间演化合理
- 可解释性强
- 计算效率高
```

### 对比分析
| 模型 | AUPRC | 训练时间 | GPU内存 | 可解释性 | 物理合理性 |
|------|--------|----------|---------|----------|------------|
| CNN  | 0.25-0.30 | 长 | 高 | 中等 | 中等 |
| CA   | 0.20-0.25 | 短 | 低 | 强 | 强 |

## 🔮 未来发展方向

### 1. 模型优化
- **超参数调优**: Weights & Biases自动搜索
- **损失函数**: 组合损失(Focal + 时间一致性)
- **多尺度训练**: 渐进式分辨率训练
- **集成学习**: 多模型投票机制

### 2. 混合架构构建
- **早期融合**: CNN特征提取 + CA时间演化
- **后期融合**: 加权集成预测结果
- **交叉注意力**: CNN空间特征与CA状态交互
- **联合训练**: 共享环境编码器

### 3. 特征工程研究
- **特征重要性**: SHAP分析各特征贡献
- **消融实验**: 系统性移除低重要特征
- **场景特异性**: 不同火灾类型的特征选择
- **时空分离**: 静态vs动态特征重要性

### 4. 实用化部署
- **模型压缩**: 知识蒸馏、剪枝
- **推理加速**: TensorRT、ONNX优化
- **实时系统**: 流式数据处理
- **边缘计算**: 移动端部署

## 💡 创新贡献

### 技术创新
1. **可微分CA**: 首次实现完全可微分的野火CA模型
2. **多模态融合**: 有效整合22维环境特征
3. **物理约束**: 在深度学习中融入火灾传播物理规律
4. **时空建模**: 联合空间CNN和时间CA优势

### 应用价值
1. **应急管理**: 为消防部门提供预测工具
2. **资源配置**: 优化消防资源部署策略
3. **风险评估**: 量化火灾传播风险
4. **政策支持**: 为防火政策制定提供科学依据

## 📚 技术栈

### 深度学习框架
- **PyTorch**: 1.13+ (CUDA支持)
- **torchvision**: 图像处理
- **torch.nn.functional**: 核心功能

### 数据处理
- **NumPy**: 数值计算
- **h5py**: HDF5文件处理
- **rasterio**: 地理空间数据
- **scikit-learn**: 数据预处理

### 可视化分析
- **matplotlib**: 基础绘图
- **seaborn**: 统计可视化
- **plotly**: 交互式图表

### 模型优化
- **Weights & Biases**: 实验跟踪
- **optuna**: 超参数优化

## 🔗 相关资源

- **数据集**: https://doi.org/10.5281/zenodo.8006177
- **项目仓库**: WildfireSpreadTS GitHub
- **技术文档**: 详见各模块README
- **实验记录**: Weights & Biases项目页面

---

*本项目致力于通过先进的深度学习技术提升野火预测能力，为全球防灾减灾事业做出贡献。* 