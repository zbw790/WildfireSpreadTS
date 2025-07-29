# 细胞自动机（CA）模型适配方案

## 网格分析重新设计

### 邻域效应分析
```python
# 基于375m分辨率的邻域设计
neighborhood_design = {
    "original_1km_resolution": "不适用，需要重新设计",
    "new_375m_resolution": {
        "moore_neighborhood": "8邻域（3x3）= 1.125km x 1.125km",
        "extended_neighborhood": "24邻域（5x5）= 1.875km x 1.875km",
        "fire_physics_range": "考虑飞火传播可达3-5km范围"
    },
    "optimal_neighborhood": {
        "local_spread": "3x3 Moore邻域用于基本传播",
        "wind_effects": "延长的椭圆形邻域考虑风向",
        "ember_transport": "5x5到7x7考虑飞火效应"
    }
}
```

### 转移规则设计
```python
# 基于WildfireSpreadTS特征的转移规则
transition_rules = {
    "environmental_factors": {
        "fuel_moisture": "基于NDVI、EVI2和降水量",
        "wind_effect": "风速和风向的矢量计算",
        "topography": "坡度和坡向的火势加速效应",
        "weather_forecast": "未来天气对传播概率的影响"
    },
    "fire_state_transitions": {
        "ignition_probability": "P(ignition) = f(weather, fuel, neighbors)",
        "spread_rate": "基于Rothermel模型和实际观测数据",
        "extinction_probability": "考虑降雨、湿度、燃料消耗"
    },
    "dynamic_parameters": {
        "daily_updates": "根据天气预报调整规则参数",
        "seasonal_adjustment": "基于历史数据的季节性校正"
    }
}
```

### 状态空间重新定义
```python
# 火灾状态的精细化定义
state_space = {
    "fire_states": {
        "unburned": 0,
        "ignited": 1, 
        "actively_burning": 2,
        "recently_burned": 3,
        "burned_out": 4
    },
    "fuel_states": {
        "high_moisture": "0.0-0.3",
        "medium_moisture": "0.3-0.6", 
        "low_moisture": "0.6-1.0 (high fire risk)"
    },
    "probabilistic_transitions": {
        "stochastic_ignition": "蒙特卡洛采样",
        "uncertainty_quantification": "集成多次运行结果"
    }
}
``` 