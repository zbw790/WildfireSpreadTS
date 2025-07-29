# CNN模型适配方案

## 空间数据结构设计

### 最优补丁尺寸分析
```python
# 基于WildfireSpreadTS的空间分析
patch_size_analysis = {
    "current_dataset_crops": "128x128 pixels (48km x 48km at 375m resolution)",
    "fire_spread_distances": {
        "daily_spread": "typically 1-5km under normal conditions",
        "extreme_conditions": "up to 20km+ in extreme weather",
        "receptive_field_needed": "minimum 64x64 for local dynamics"
    },
    "recommended_patches": {
        "local_dynamics": "64x64 (24km x 24km)",
        "regional_context": "128x128 (48km x 48km)", 
        "large_fires": "256x256 (96km x 96km)"
    }
}
```

### 多时序输入设计
```python
# 时序窗口配置
temporal_windows = {
    "short_term": {
        "window_length": "1-3 days",
        "purpose": "immediate fire behavior prediction",
        "input_shape": "(batch, time_steps, channels, height, width)"
    },
    "medium_term": {
        "window_length": "5-7 days", 
        "purpose": "weather pattern integration",
        "input_shape": "(batch, 5, 40, 128, 128)"
    },
    "long_term": {
        "window_length": "10-14 days",
        "purpose": "seasonal and drought condition analysis"
    }
}
```

### 数据增强策略重新设计
```python
# 针对WildfireSpreadTS的增强策略
augmentation_strategies = {
    "spatial_augmentations": {
        "rotation": "90度倍数旋转（保持风向一致性）",
        "flipping": "水平/垂直翻转（调整风向和坡向）", 
        "cropping": "智能裁剪（优先包含火点区域）"
    },
    "temporal_augmentations": {
        "time_jittering": "±1天时间偏移",
        "interpolation": "线性插值处理缺失数据",
        "sequence_shuffling": "短序列内随机排列"
    },
    "feature_augmentations": {
        "weather_perturbation": "气象数据添加现实噪声",
        "topographic_invariance": "地形特征保持不变"
    }
}
``` 