"""
WildFire Data Preprocessing
专用于野火传播预测的数据预处理模块，包含特征工程、标准化和数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings

class WildfireFeatureProcessor:
    """
    野火特征处理器
    """
    
    def __init__(self):
        # 23通道特征定义
        self.feature_names = [
            'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1', 'NDVI', 'EVI2',  # 0-4: 遥感
            'Precipitation_Total', 'Wind_Speed', 'Wind_Direction',  # 5-7: 气象历史
            'Temperature_Min', 'Temperature_Max', 'ERC', 'Specific_Humidity',  # 8-11: 气象历史
            'Slope', 'Aspect', 'Elevation', 'PDSI',  # 12-15: 地形+干旱
            'Land_Cover_Class',  # 16: 土地覆盖
            'Forecast_Precipitation', 'Forecast_Wind_Speed',  # 17-18: 预测气象
            'Forecast_Wind_Direction', 'Forecast_Temperature',  # 19-20: 预测气象
            'Forecast_Specific_Humidity',  # 21: 预测气象
            'Active_Fire_Confidence'  # 22: 火点置信度
        ]
        
        # 特征分组
        self.feature_groups = {
            'remote_sensing': [0, 1, 2],  # VIIRS
            'vegetation': [3, 4],  # NDVI, EVI2
            'weather_hist': [5, 6, 7, 8, 9, 10, 11],  # 历史气象
            'topography': [12, 13, 14],  # 地形
            'drought': [15],  # 干旱指数
            'landcover': [16],  # 土地覆盖
            'weather_forecast': [17, 18, 19, 20, 21],  # 预测气象
            'fire': [22]  # 火点
        }
        
        # 特征物理约束
        self.feature_constraints = {
            'VIIRS_M11': {'min': 38, 'max': 11886, 'cyclic': False},
            'VIIRS_I2': {'min': -100, 'max': 15893, 'cyclic': False},
            'VIIRS_I1': {'min': -100, 'max': 15666, 'cyclic': False},
            'NDVI': {'min': -1405, 'max': 9736, 'cyclic': False},
            'EVI2': {'min': -843, 'max': 6075, 'cyclic': False},
            'Precipitation_Total': {'min': 0, 'max': 100, 'cyclic': False},
            'Wind_Speed': {'min': 0, 'max': 50, 'cyclic': False},
            'Wind_Direction': {'min': 0, 'max': 360, 'cyclic': True},
            'Temperature_Min': {'min': 200, 'max': 350, 'cyclic': False},
            'Temperature_Max': {'min': 200, 'max': 350, 'cyclic': False},
            'ERC': {'min': 0, 'max': 200, 'cyclic': False},
            'Specific_Humidity': {'min': 0, 'max': 0.05, 'cyclic': False},
            'Slope': {'min': 0, 'max': 90, 'cyclic': False},
            'Aspect': {'min': 0, 'max': 360, 'cyclic': True},
            'Elevation': {'min': 0, 'max': 5000, 'cyclic': False},
            'PDSI': {'min': -10, 'max': 10, 'cyclic': False},
            'Land_Cover_Class': {'min': 1, 'max': 16, 'cyclic': False},
            'Active_Fire_Confidence': {'min': 0, 'max': 100, 'cyclic': False}
        }
    
    def apply_feature_constraints(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
        """应用特征物理约束"""
        feature_name = self.feature_names[feature_idx]
        constraints = self.feature_constraints.get(feature_name, {})
        
        if 'min' in constraints and 'max' in constraints:
            data = np.clip(data, constraints['min'], constraints['max'])
        
        return data
    
    def compute_derived_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """计算衍生特征"""
        derived = {}
        
        # 温度相关
        if len(data) > 9:  # 确保有温度数据
            temp_min = data[8]  # Temperature_Min
            temp_max = data[9]  # Temperature_Max
            derived['temperature_range'] = temp_max - temp_min
            derived['temperature_mean'] = (temp_max + temp_min) / 2
        
        # 风相关
        if len(data) > 7:
            wind_speed = data[6]  # Wind_Speed
            wind_dir = data[7]    # Wind_Direction (degrees)
            
            # 风矢量分解
            wind_dir_rad = np.deg2rad(wind_dir)
            derived['wind_u'] = wind_speed * np.cos(wind_dir_rad)  # 东西分量
            derived['wind_v'] = wind_speed * np.sin(wind_dir_rad)  # 南北分量
        
        # 湿度相关
        if len(data) > 11 and len(data) > 5:
            humidity = data[11]  # Specific_Humidity
            precipitation = data[5]  # Precipitation_Total
            derived['moisture_index'] = humidity + 0.1 * precipitation
        
        # 火险等级
        if len(data) > 10 and len(data) > 6:
            erc = data[10]  # ERC
            wind_speed = data[6]  # Wind_Speed
            derived['fire_danger_index'] = erc * (1 + 0.1 * wind_speed)
        
        # 地形效应
        if len(data) > 14:
            slope = data[12]    # Slope
            aspect = data[13]   # Aspect
            elevation = data[14]  # Elevation
            
            # 坡向效应（南坡更干燥）
            aspect_rad = np.deg2rad(aspect)
            derived['south_facing_effect'] = np.cos(aspect_rad)  # 1=正南，-1=正北
            
            # 地形复杂度
            derived['terrain_complexity'] = slope * (elevation / 1000)
        
        return derived

class WildfireNormalizer:
    """
    野火数据标准化器
    """
    
    def __init__(self, method: str = 'robust'):
        """
        Args:
            method: 标准化方法 ('standard', 'robust', 'minmax')
        """
        self.method = method
        self.scalers = {}
        self.fitted = False
        
        # 特征分组
        self.feature_groups = {
            'remote_sensing': [0, 1, 2],
            'vegetation': [3, 4], 
            'weather': [5, 6, 8, 9, 10, 11, 17, 18, 20, 21],  # 非方向性气象
            'wind_direction': [7, 19],  # 风向（需要特殊处理）
            'topography': [12, 14],  # 坡度和海拔
            'aspect': [13],  # 坡向（循环特征）
            'drought': [15],
            'landcover': [16],
            'fire': [22]
        }
    
    def _create_scaler(self) -> Any:
        """创建标准化器"""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'robust':
            return RobustScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _normalize_circular_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理循环特征（如风向、坡向）
        转换为sin/cos分量
        """
        # 转换为弧度
        data_rad = np.deg2rad(data)
        
        # 转换为sin/cos分量
        sin_component = np.sin(data_rad)
        cos_component = np.cos(data_rad)
        
        return sin_component, cos_component
    
    def fit(self, data: np.ndarray):
        """
        拟合标准化器
        
        Args:
            data: (N, C) 或 (N, T, C) 或 (N, T, C, H, W) 数据
        """
        # 重塑数据为 (samples, features)
        original_shape = data.shape
        if len(original_shape) == 5:  # (N, T, C, H, W)
            data = data.reshape(-1, original_shape[2])
        elif len(original_shape) == 3:  # (N, T, C)
            data = data.reshape(-1, original_shape[2])
        elif len(original_shape) == 2:  # (N, C)
            pass
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")
        
        # 过滤有效值
        valid_mask = np.isfinite(data).all(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            warnings.warn("No valid data for fitting normalizer")
            return
        
        # 为每个特征组创建标准化器
        for group_name, indices in self.feature_groups.items():
            if group_name in ['wind_direction', 'aspect']:
                # 循环特征不需要标准化
                continue
            
            if all(idx < data.shape[1] for idx in indices):
                group_data = data[:, indices]
                scaler = self._create_scaler()
                scaler.fit(group_data)
                self.scalers[group_name] = scaler
        
        self.fitted = True
        print(f"标准化器拟合完成，使用方法: {self.method}")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        应用标准化
        
        Args:
            data: 输入数据
            
        Returns:
            normalized_data: 标准化后的数据
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        original_shape = data.shape
        
        # 重塑数据
        if len(original_shape) == 5:  # (N, T, C, H, W)
            data = data.reshape(-1, original_shape[2])
            spatial_dims = original_shape[3:]
        elif len(original_shape) == 3:  # (N, T, C)
            data = data.reshape(-1, original_shape[2])
            spatial_dims = None
        elif len(original_shape) == 2:  # (N, C)
            spatial_dims = None
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")
        
        normalized_data = data.copy()
        
        # 应用各组标准化
        for group_name, indices in self.feature_groups.items():
            if group_name in ['wind_direction', 'aspect']:
                # 处理循环特征
                for idx in indices:
                    if idx < data.shape[1]:
                        circular_data = data[:, idx]
                        sin_comp, cos_comp = self._normalize_circular_features(circular_data)
                        
                        # 替换原始数据为sin分量，cos分量需要额外处理
                        normalized_data[:, idx] = sin_comp
                        # 注意：这里我们只保留sin分量，实际使用中可能需要扩展维度
                continue
            
            if group_name in self.scalers and all(idx < data.shape[1] for idx in indices):
                scaler = self.scalers[group_name]
                group_data = data[:, indices]
                normalized_group = scaler.transform(group_data)
                normalized_data[:, indices] = normalized_group
        
        # 恢复原始形状
        if len(original_shape) == 5:
            normalized_data = normalized_data.reshape(original_shape)
        elif len(original_shape) == 3:
            normalized_data = normalized_data.reshape(original_shape)
        
        return normalized_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并应用标准化"""
        self.fit(data)
        return self.transform(data)

class WildfireAugmenter:
    """
    野火数据增强器
    保持物理一致性的数据增强
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        noise_prob: float = 0.2,
        weather_perturb_prob: float = 0.3,
        noise_std: float = 0.01
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.weather_perturb_prob = weather_perturb_prob
        self.noise_std = noise_std
        
        # 可以添加噪声的特征（气象数据）
        self.noise_channels = [5, 6, 8, 9, 10, 11, 17, 18, 20, 21]  # 气象特征
        
        # 风向通道（需要特殊处理）
        self.wind_direction_channels = [7, 19]  # 历史风向，预测风向
    
    def _flip_horizontal(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """水平翻转（保持风向一致性）"""
        # 翻转数据
        data_flipped = np.flip(data, axis=-1)
        target_flipped = np.flip(target, axis=-1)
        
        # 调整风向（水平翻转后风向需要镜像）
        for ch in self.wind_direction_channels:
            if ch < data.shape[-3]:  # 确保通道存在
                wind_dir = data_flipped[..., ch, :, :]
                # 水平翻转: E-W反向，N-S保持
                # 新风向 = 360 - 原风向（对于0-360度）
                wind_dir_new = 360 - wind_dir
                wind_dir_new = wind_dir_new % 360
                data_flipped[..., ch, :, :] = wind_dir_new
        
        return data_flipped, target_flipped
    
    def _flip_vertical(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """垂直翻转（保持风向一致性）"""
        # 翻转数据
        data_flipped = np.flip(data, axis=-2)
        target_flipped = np.flip(target, axis=-2)
        
        # 调整风向（垂直翻转后风向需要调整）
        for ch in self.wind_direction_channels:
            if ch < data.shape[-3]:
                wind_dir = data_flipped[..., ch, :, :]
                # 垂直翻转: N-S反向，E-W保持
                # 新风向 = 180 - 原风向
                wind_dir_new = 180 - wind_dir
                wind_dir_new = wind_dir_new % 360
                data_flipped[..., ch, :, :] = wind_dir_new
        
        return data_flipped, target_flipped
    
    def _rotate_90(self, data: np.ndarray, target: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """90度倍数旋转（保持风向一致性）"""
        # 旋转数据
        data_rotated = np.rot90(data, k, axes=(-2, -1))
        target_rotated = np.rot90(target, k, axes=(-2, -1))
        
        # 调整风向
        angle_adjustment = k * 90  # 旋转角度
        for ch in self.wind_direction_channels:
            if ch < data.shape[-3]:
                wind_dir = data_rotated[..., ch, :, :]
                wind_dir_new = (wind_dir + angle_adjustment) % 360
                data_rotated[..., ch, :, :] = wind_dir_new
        
        return data_rotated, target_rotated
    
    def _add_weather_noise(self, data: np.ndarray) -> np.ndarray:
        """添加气象噪声（模拟测量不确定性）"""
        data_noisy = data.copy()
        
        for ch in self.noise_channels:
            if ch < data.shape[-3]:
                noise = np.random.normal(0, self.noise_std, data[..., ch, :, :].shape)
                data_noisy[..., ch, :, :] += noise
        
        return data_noisy
    
    def _perturb_weather(self, data: np.ndarray) -> np.ndarray:
        """扰动气象条件（模拟天气变化）"""
        data_perturbed = data.copy()
        
        # 温度扰动 (±2K)
        for temp_ch in [8, 9, 20]:  # Temperature_Min, Temperature_Max, Forecast_Temperature
            if temp_ch < data.shape[-3]:
                temp_delta = np.random.uniform(-2, 2, data[..., temp_ch, :, :].shape)
                data_perturbed[..., temp_ch, :, :] += temp_delta
        
        # 风速扰动 (±10%)
        for wind_ch in [6, 18]:  # Wind_Speed, Forecast_Wind_Speed
            if wind_ch < data.shape[-3]:
                wind_factor = np.random.uniform(0.9, 1.1, data[..., wind_ch, :, :].shape)
                data_perturbed[..., wind_ch, :, :] *= wind_factor
        
        # 湿度扰动 (±10%)
        for humid_ch in [11, 21]:  # Specific_Humidity, Forecast_Specific_Humidity
            if humid_ch < data.shape[-3]:
                humid_factor = np.random.uniform(0.9, 1.1, data[..., humid_ch, :, :].shape)
                data_perturbed[..., humid_ch, :, :] *= humid_factor
        
        return data_perturbed
    
    def augment(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强
        
        Args:
            data: (T, C, H, W) 输入特征
            target: (H, W) 目标标签
            
        Returns:
            augmented_data, augmented_target
        """
        augmented_data = data.copy()
        augmented_target = target.copy()
        
        # 水平翻转
        if np.random.random() < self.flip_prob:
            augmented_data, augmented_target = self._flip_horizontal(augmented_data, augmented_target)
        
        # 垂直翻转
        if np.random.random() < self.flip_prob:
            augmented_data, augmented_target = self._flip_vertical(augmented_data, augmented_target)
        
        # 旋转
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)  # 90, 180, 270度
            augmented_data, augmented_target = self._rotate_90(augmented_data, augmented_target, k)
        
        # 添加噪声
        if np.random.random() < self.noise_prob:
            augmented_data = self._add_weather_noise(augmented_data)
        
        # 扰动气象条件
        if np.random.random() < self.weather_perturb_prob:
            augmented_data = self._perturb_weather(augmented_data)
        
        return augmented_data, augmented_target

def test_preprocessing():
    """测试预处理功能"""
    print("🧪 测试野火数据预处理...")
    
    # 创建测试数据
    batch_size, seq_len, channels, height, width = 4, 5, 23, 64, 64
    
    # 模拟真实数据范围
    data = np.random.randn(batch_size, seq_len, channels, height, width)
    
    # 设置真实的数据范围
    data[:, :, 0, :, :] = np.random.uniform(38, 11886, (batch_size, seq_len, height, width))  # VIIRS_M11
    data[:, :, 6, :, :] = np.random.uniform(0, 10, (batch_size, seq_len, height, width))      # Wind_Speed
    data[:, :, 7, :, :] = np.random.uniform(0, 360, (batch_size, seq_len, height, width))    # Wind_Direction
    
    target = np.random.choice([0, 1], size=(batch_size, height, width), p=[0.95, 0.05])
    
    print(f"原始数据形状: {data.shape}")
    print(f"目标形状: {target.shape}")
    
    # 测试特征处理器
    print("\n📊 测试特征处理器...")
    processor = WildfireFeatureProcessor()
    
    # 计算衍生特征
    sample_data = data[0, 0, :, 32, 32]  # 取一个像素的数据
    derived_features = processor.compute_derived_features(sample_data)
    print(f"衍生特征数量: {len(derived_features)}")
    
    # 测试标准化器
    print("\n🔧 测试标准化器...")
    normalizer = WildfireNormalizer(method='robust')
    
    # 拟合并变换
    normalized_data = normalizer.fit_transform(data)
    print(f"标准化前范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"标准化后范围: [{normalized_data.min():.2f}, {normalized_data.max():.2f}]")
    
    # 测试数据增强器
    print("\n🎲 测试数据增强器...")
    augmenter = WildfireAugmenter()
    
    sample_sequence = data[0]  # (T, C, H, W)
    sample_target = target[0]  # (H, W)
    
    aug_data, aug_target = augmenter.augment(sample_sequence, sample_target)
    print(f"增强前数据形状: {sample_sequence.shape}")
    print(f"增强后数据形状: {aug_data.shape}")
    print(f"风向变化: {sample_sequence[0, 7, 32, 32]:.1f}° -> {aug_data[0, 7, 32, 32]:.1f}°")
    
    print("✅ 预处理测试完成！")

if __name__ == "__main__":
    test_preprocessing() 