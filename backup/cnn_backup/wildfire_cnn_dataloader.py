"""
WildFire CNN Data Loader
专用于野火传播预测的CNN数据加载器，处理23通道多模态数据和严重类别不平衡
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
import os
import glob
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
from tqdm import tqdm

class WildfireDataset(Dataset):
    """
    野火传播预测数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        years: List[int],
        sequence_length: int = 5,
        prediction_horizon: int = 1,
        crop_size: int = 128,
        stride: int = 64,
        is_training: bool = True,
        normalize: bool = True,
        fire_threshold: float = 0.5,
        augment: bool = True
    ):
        """
        Args:
            data_dir: HDF5文件根目录
            years: 使用的年份列表
            sequence_length: 输入时间序列长度
            prediction_horizon: 预测时间步长
            crop_size: 裁剪尺寸
            stride: 滑动窗口步长
            is_training: 是否为训练模式
            normalize: 是否标准化
            fire_threshold: 火点二值化阈值
            augment: 是否数据增强
        """
        self.data_dir = data_dir
        self.years = years
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.crop_size = crop_size
        self.stride = stride
        self.is_training = is_training
        self.normalize = normalize
        self.fire_threshold = fire_threshold
        self.augment = augment
        
        # 23通道特征定义
        self.feature_channels = 23
        self.fire_channel = 22  # 火点置信度通道
        
        # 特征类别定义（用于分类别标准化）
        self.remote_sensing_channels = [0, 1, 2]  # VIIRS
        self.vegetation_channels = [3, 4]  # NDVI, EVI2
        self.weather_channels = [5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21]  # 气象+预测
        self.topography_channels = [12, 13, 14]  # 地形
        self.drought_channels = [15]  # PDSI
        self.landcover_channels = [16]  # 土地覆盖
        
        # 收集所有火灾文件
        self.fire_files = self._collect_fire_files()
        
        # 创建样本索引
        self.samples = self._create_sample_indices()
        
        # 初始化标准化器
        if self.normalize:
            self.scalers = self._initialize_scalers()
        
        print(f"数据集初始化完成:")
        print(f"  年份: {years}")
        print(f"  火灾事件: {len(self.fire_files)}")
        print(f"  样本数量: {len(self.samples)}")
        print(f"  序列长度: {sequence_length}")
        print(f"  裁剪尺寸: {crop_size}x{crop_size}")
    
    def _collect_fire_files(self) -> List[str]:
        """收集所有HDF5火灾文件"""
        fire_files = []
        for year in self.years:
            year_dir = os.path.join(self.data_dir, str(year))
            if os.path.exists(year_dir):
                files = glob.glob(os.path.join(year_dir, "*.hdf5"))
                fire_files.extend(files)
        return sorted(fire_files)
    
    def _create_sample_indices(self) -> List[Dict]:
        """创建样本索引，包含火灾文件和时空位置"""
        samples = []
        
        for fire_file in tqdm(self.fire_files, desc="Creating sample indices"):
            try:
                with h5py.File(fire_file, 'r') as f:
                    data = f['data']
                    T, C, H, W = data.shape
                    
                    # 确保有足够的时间步
                    if T < self.sequence_length + self.prediction_horizon:
                        continue
                    
                    # 时间窗口
                    for t_start in range(T - self.sequence_length - self.prediction_horizon + 1):
                        t_end = t_start + self.sequence_length
                        t_target = t_end + self.prediction_horizon - 1
                        
                        # 空间窗口
                        for h_start in range(0, H - self.crop_size + 1, self.stride):
                            for w_start in range(0, W - self.crop_size + 1, self.stride):
                                h_end = h_start + self.crop_size
                                w_end = w_start + self.crop_size
                                
                                samples.append({
                                    'fire_file': fire_file,
                                    't_start': t_start,
                                    't_end': t_end,
                                    't_target': t_target,
                                    'h_start': h_start,
                                    'h_end': h_end,
                                    'w_start': w_start,
                                    'w_end': w_end
                                })
            except Exception as e:
                warnings.warn(f"Error processing {fire_file}: {e}")
                continue
        
        return samples
    
    def _initialize_scalers(self) -> Dict[str, object]:
        """初始化不同类别特征的标准化器"""
        scalers = {}
        
        # 遥感数据用RobustScaler（抗异常值）
        scalers['remote_sensing'] = RobustScaler()
        scalers['vegetation'] = RobustScaler()
        
        # 气象数据用StandardScaler
        scalers['weather'] = StandardScaler()
        scalers['topography'] = StandardScaler()
        scalers['drought'] = StandardScaler()
        
        # 计算标准化参数
        self._fit_scalers(scalers)
        
        return scalers
    
    def _fit_scalers(self, scalers: Dict[str, object]):
        """拟合标准化器参数"""
        print("🔧 计算特征标准化参数...")
        
        # 收集样本数据用于拟合
        sample_data = {
            'remote_sensing': [],
            'vegetation': [],
            'weather': [],
            'topography': [],
            'drought': []
        }
        
        # 采样部分数据计算统计量
        sample_files = self.fire_files[::max(1, len(self.fire_files)//10)]  # 采样10%
        
        for fire_file in tqdm(sample_files, desc="Fitting scalers"):
            try:
                with h5py.File(fire_file, 'r') as f:
                    data = f['data'][:]  # (T, C, H, W)
                    
                    # 随机采样空间位置
                    T, C, H, W = data.shape
                    sample_size = min(1000, H*W//100)
                    
                    # 展平并采样
                    data_flat = data.reshape(T*C, H*W).T  # (pixels, T*C)
                    indices = np.random.choice(data_flat.shape[0], sample_size, replace=False)
                    data_sample = data_flat[indices]  # (sample_size, T*C)
                    
                    # 重塑回时间维度
                    data_sample = data_sample.reshape(sample_size, T, C)
                    data_sample = data_sample.reshape(-1, C)  # (sample_size*T, C)
                    
                    # 过滤有限值
                    valid_mask = np.isfinite(data_sample).all(axis=1)
                    data_sample = data_sample[valid_mask]
                    
                    if len(data_sample) > 0:
                        sample_data['remote_sensing'].append(data_sample[:, self.remote_sensing_channels])
                        sample_data['vegetation'].append(data_sample[:, self.vegetation_channels])
                        sample_data['weather'].append(data_sample[:, self.weather_channels])
                        sample_data['topography'].append(data_sample[:, self.topography_channels])
                        sample_data['drought'].append(data_sample[:, self.drought_channels])
                        
            except Exception as e:
                continue
        
        # 拟合标准化器
        for category, scaler in scalers.items():
            if sample_data[category]:
                combined_data = np.vstack(sample_data[category])
                scaler.fit(combined_data)
                print(f"  {category}: {combined_data.shape}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """标准化特征"""
        if not self.normalize:
            return features
        
        normalized = features.copy()
        
        # 分类别标准化
        if self.remote_sensing_channels:
            normalized[..., self.remote_sensing_channels] = self.scalers['remote_sensing'].transform(
                features[..., self.remote_sensing_channels].reshape(-1, len(self.remote_sensing_channels))
            ).reshape(features[..., self.remote_sensing_channels].shape)
        
        if self.vegetation_channels:
            normalized[..., self.vegetation_channels] = self.scalers['vegetation'].transform(
                features[..., self.vegetation_channels].reshape(-1, len(self.vegetation_channels))
            ).reshape(features[..., self.vegetation_channels].shape)
        
        if self.weather_channels:
            normalized[..., self.weather_channels] = self.scalers['weather'].transform(
                features[..., self.weather_channels].reshape(-1, len(self.weather_channels))
            ).reshape(features[..., self.weather_channels].shape)
        
        if self.topography_channels:
            normalized[..., self.topography_channels] = self.scalers['topography'].transform(
                features[..., self.topography_channels].reshape(-1, len(self.topography_channels))
            ).reshape(features[..., self.topography_channels].shape)
        
        if self.drought_channels:
            normalized[..., self.drought_channels] = self.scalers['drought'].transform(
                features[..., self.drought_channels].reshape(-1, len(self.drought_channels))
            ).reshape(features[..., self.drought_channels].shape)
        
        return normalized
    
    def _augment_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强（保持物理一致性）"""
        if not self.augment or not self.is_training:
            return features, target
        
        # 随机翻转（水平/垂直）
        if np.random.random() > 0.5:
            features = np.flip(features, axis=-1).copy()  # 水平翻转
            target = np.flip(target, axis=-1).copy()
        
        if np.random.random() > 0.5:
            features = np.flip(features, axis=-2).copy()  # 垂直翻转
            target = np.flip(target, axis=-2).copy()
        
        # 随机旋转（90度倍数，保持风向一致性）
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)  # 90, 180, 270度
            features = np.rot90(features, k, axes=(-2, -1)).copy()
            target = np.rot90(target, k, axes=(-2, -1)).copy()
            
            # 调整风向（通道7和19）
            if 7 < features.shape[1]:
                features[:, 7] = (features[:, 7] + k * 90) % 360  # 历史风向
            if 19 < features.shape[1]:
                features[:, 19] = (features[:, 19] + k * 90) % 360  # 预测风向变化
        
        return features, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        try:
            with h5py.File(sample['fire_file'], 'r') as f:
                data = f['data']
                
                # 提取输入特征序列
                features = data[
                    sample['t_start']:sample['t_end'],
                    :self.feature_channels,  # 前23个通道
                    sample['h_start']:sample['h_end'],
                    sample['w_start']:sample['w_end']
                ]
                features = np.array(features, dtype=np.float32)
                
                # 提取目标（火点置信度）
                target_raw = data[
                    sample['t_target'],
                    self.fire_channel,  # 通道22
                    sample['h_start']:sample['h_end'],
                    sample['w_start']:sample['w_end']
                ]
                target_raw = np.array(target_raw, dtype=np.float32)
                
                # 二值化目标
                target = (target_raw > self.fire_threshold).astype(np.float32)
                
                # 标准化特征
                features = self._normalize_features(features)
                
                # 数据增强
                features, target = self._augment_data(features, target)
                
                # 转换为张量
                features = torch.from_numpy(features)  # (T, C, H, W)
                target = torch.from_numpy(target)      # (H, W)
                
                return features, target
                
        except Exception as e:
            # 返回默认样本避免训练中断
            warnings.warn(f"Error loading sample {idx}: {e}")
            features = torch.zeros(self.sequence_length, self.feature_channels, self.crop_size, self.crop_size)
            target = torch.zeros(self.crop_size, self.crop_size)
            return features, target

def create_weighted_sampler(dataset: WildfireDataset, target_fire_ratio: float = 0.1) -> WeightedRandomSampler:
    """
    创建加权采样器处理类别不平衡
    
    Args:
        dataset: 数据集
        target_fire_ratio: 目标火点像素比例
    """
    print("⚖️ 分析类别分布，创建加权采样器...")
    
    fire_counts = []
    total_counts = []
    
    # 采样部分数据计算火点比例
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in tqdm(indices, desc="Analyzing class distribution"):
        try:
            _, target = dataset[idx]
            fire_pixels = target.sum().item()
            total_pixels = target.numel()
            
            fire_counts.append(fire_pixels)
            total_counts.append(total_pixels)
            
        except:
            continue
    
    # 计算样本权重
    fire_counts = np.array(fire_counts)
    total_counts = np.array(total_counts)
    fire_ratios = fire_counts / (total_counts + 1e-8)
    
    # 根据火点比例分配权重
    weights = np.ones(len(dataset))
    
    # 为有火点的样本分配更高权重
    high_fire_mask = fire_ratios > 0.001  # 火点比例 > 0.1%
    medium_fire_mask = (fire_ratios > 0.0001) & (fire_ratios <= 0.001)  # 0.01-0.1%
    
    if high_fire_mask.sum() > 0:
        weights[indices[high_fire_mask]] = 10.0  # 高火点样本10倍权重
    if medium_fire_mask.sum() > 0:
        weights[indices[medium_fire_mask]] = 3.0  # 中等火点样本3倍权重
    
    print(f"  总样本: {len(dataset)}")
    print(f"  高火点样本 (>0.1%): {high_fire_mask.sum()} (权重×10)")
    print(f"  中火点样本 (0.01-0.1%): {medium_fire_mask.sum()} (权重×3)")
    print(f"  低火点样本: {len(dataset) - high_fire_mask.sum() - medium_fire_mask.sum()} (权重×1)")
    
    return WeightedRandomSampler(weights, len(dataset), replacement=True)

def create_dataloaders(
    data_dir: str,
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    """
    print("创建数据加载器...")
    
    # 创建数据集
    train_dataset = WildfireDataset(
        data_dir=data_dir,
        years=train_years,
        is_training=True,
        **dataset_kwargs
    )
    
    val_dataset = WildfireDataset(
        data_dir=data_dir,
        years=val_years,
        is_training=False,
        **dataset_kwargs
    )
    
    test_dataset = WildfireDataset(
        data_dir=data_dir,
        years=test_years,
        is_training=False,
        **dataset_kwargs
    )
    
    # 创建加权采样器
    train_sampler = create_weighted_sampler(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试数据加载器...")
    
    # 配置
    data_dir = "data/processed"
    train_years = [2018, 2019]
    val_years = [2020]
    test_years = [2021]
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        batch_size=4,
        sequence_length=3,
        crop_size=64
    )
    
    # 测试加载一个batch
    print("\n📦 测试数据加载...")
    for features, targets in train_loader:
        print(f"特征形状: {features.shape}")  # (B, T, C, H, W)
        print(f"目标形状: {targets.shape}")   # (B, H, W)
        print(f"火点像素比例: {targets.mean():.6f}")
        break
    
    print("✅ 数据加载器测试完成！") 