# 评估框架适配方案

## 指标选择重新设计

### 超越标准精度的指标
```python
# 针对火灾传播预测的专门指标
evaluation_metrics = {
    "pixel_level_metrics": {
        "AUPRC": "处理极度不平衡的火/非火像素",
        "IoU": "火灾区域的空间重叠评估", 
        "F1_Score": "平衡精确率和召回率",
        "Dice_Coefficient": "医学图像分割中的重叠指标"
    },
    "spatial_accuracy_metrics": {
        "Hausdorff_Distance": "火灾边界精度评估",
        "Average_Surface_Distance": "平均表面距离",
        "Boundary_IoU": "专注于火灾边界的IoU",
        "Contour_Matching": "火灾轮廓形状相似度"
    },
    "temporal_progression_metrics": {
        "Spread_Trajectory_Overlap": "时间序列传播路径准确性",
        "Growth_Rate_Accuracy": "火灾增长速率预测精度",
        "Direction_Consistency": "传播方向的角度误差",
        "Timing_Accuracy": "到达特定位置的时间预测"
    },
    "physical_realism_metrics": {
        "Fire_Physics_Consistency": "是否符合火灾物理规律",
        "Connectivity_Preservation": "火灾区域连通性保持",
        "Spread_Rate_Realism": "传播速率的现实性评估"
    }
}
```

### 类别不平衡处理策略
```python
# 适配WildfireSpreadTS的不平衡问题
imbalance_handling = {
    "problem_redefinition": {
        "original_issue": "火灾发生vs非发生（极度不平衡）",
        "new_focus": "火灾传播vs非传播（相对平衡）",
        "active_fire_area": "仅在现有火灾周围进行预测",
        "roi_definition": "定义感兴趣区域缩小预测范围"
    },
    "sampling_strategies": {
        "spatial_sampling": "围绕火灾边界的智能采样",
        "temporal_sampling": "火灾活跃期的重点采样", 
        "balanced_patches": "确保训练补丁包含传播和非传播案例",
        "hard_negative_mining": "重点学习困难的边界案例"
    },
    "loss_function_adaptation": {
        "focal_loss": "处理易/难样本不平衡",
        "weighted_cross_entropy": "根据像素类型调整权重",
        "boundary_enhanced_loss": "增强火灾边界学习",
        "temporal_consistency_loss": "保证时间一致性"
    }
}
```

## 验证策略重新设计

### 时间分割策略
```python
# 基于WildfireSpreadTS 2018-2021数据的分割
temporal_splits = {
    "training_period": {
        "years": [2018, 2019],
        "rationale": "包含不同火灾模式的多样性训练",
        "data_distribution": "正常火灾年份为主"
    },
    "validation_period": {
        "years": [2020],
        "rationale": "2020年极端火灾年份用于调参",
        "special_events": "包含加州极端火灾事件"
    }, 
    "test_period": {
        "years": [2021],
        "rationale": "完全未见过的年份进行最终评估",
        "evaluation_focus": "泛化能力和极端事件处理"
    }
}
```

### 空间交叉验证
```python
# 地理泛化能力评估
spatial_validation = {
    "geographic_holdout": {
        "method": "按火灾地理位置进行留出验证",
        "regions": "美国西部不同生态区",
        "purpose": "评估地理泛化能力"
    },
    "fire_type_splits": {
        "forest_fires": "森林火灾专门评估",
        "grassland_fires": "草地火灾评估", 
        "mixed_vegetation": "混合植被火灾评估"
    },
    "climate_zone_validation": {
        "mediterranean": "地中海气候区",
        "arid": "干旱气候区",
        "temperate": "温带气候区"
    }
}
```

### 极端事件专门评估
```python
# 针对极端火灾事件的评估
extreme_event_evaluation = {
    "identification_criteria": {
        "fire_size": ">10,000 hectares",
        "spread_rate": ">5km/day平均传播速度",
        "duration": ">14天持续时间",
        "weather_conditions": "极端天气条件下的火灾"
    },
    "specialized_metrics": {
        "rapid_spread_detection": "快速传播事件的早期识别",
        "extreme_weather_robustness": "极端天气条件下的模型稳定性",
        "large_fire_boundary_accuracy": "大型火灾边界预测精度"
    },
    "failure_case_analysis": {
        "systematic_underestimation": "系统性低估分析",
        "direction_errors": "传播方向错误模式",
        "timing_delays": "时间预测延迟分析"
    }
}
``` 