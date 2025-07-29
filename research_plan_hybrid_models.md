# 混合CNN+CA模型适配方案

## 集成架构设计

### CNN特征提取 → CA规则参数估计
```python
# 混合架构设计
hybrid_architecture = {
    "cnn_component": {
        "purpose": "全局模式识别和环境特征提取",
        "input": "多时序多光谱数据 (T, 40, H, W)",
        "output": "局部CA规则参数映射",
        "architecture": "ResNet-UNet with temporal attention"
    },
    "ca_component": {
        "purpose": "基于物理的局部传播动力学",
        "input": "CNN提取的参数 + 当前火灾状态",
        "output": "下一时刻火灾状态预测",
        "rules": "CNN参数化的转移规则"
    },
    "integration_strategy": {
        "parameter_mapping": "CNN输出→CA转移概率",
        "multi_scale_fusion": "不同尺度的CNN特征→不同CA邻域规则",
        "feedback_loop": "CA结果→CNN下一轮输入"
    }
}
```

### 多尺度分析架构
```python
# 多尺度集成方案
multi_scale_design = {
    "global_scale": {
        "cnn_receptive_field": "256x256 pixels (96km x 96km)",
        "purpose": "气候模式、地形大格局分析",
        "ca_influence": "区域级传播倾向参数"
    },
    "regional_scale": {
        "cnn_receptive_field": "128x128 pixels (48km x 48km)", 
        "purpose": "天气系统、地形细节",
        "ca_influence": "传播速率和方向参数"
    },
    "local_scale": {
        "cnn_receptive_field": "64x64 pixels (24km x 24km)",
        "purpose": "微观燃料和风场条件",
        "ca_influence": "具体像素转移概率"
    }
}
```

### 不确定性量化
```python
# 随机CA组件的不确定性处理
uncertainty_framework = {
    "ensemble_methods": {
        "cnn_ensemble": "多个CNN模型投票决定CA参数",
        "ca_monte_carlo": "每次预测运行多次随机CA",
        "combined_uncertainty": "CNN预测不确定性 + CA随机性"
    },
    "confidence_estimation": {
        "pixel_level": "每个像素的点火概率置信区间",
        "spatial_propagation": "传播路径的不确定性量化",
        "temporal_evolution": "时间序列预测的置信度衰减"
    }
}
``` 