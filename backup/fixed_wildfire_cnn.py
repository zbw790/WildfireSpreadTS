#!/usr/bin/env python3
"""
16核CPU优化版野火训练系统
充分利用多核CPU资源进行并行处理
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import h5py
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
import warnings
import gc
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import threading
warnings.filterwarnings('ignore')

# 设置CPU优化
def optimize_cpu_settings():
    """优化CPU设置"""
    cpu_count = psutil.cpu_count(logical=True)
    print(f"🖥️  检测到 {cpu_count} 个逻辑CPU核心")
    
    # 设置线程数
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)
    
    # OpenMP设置
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
    
    # 启用CPU优化
    if hasattr(torch.backends.mkl, 'is_available') and torch.backends.mkl.is_available():
        torch.backends.mkl.enabled = True
        print("✅ 启用MKL优化")
    
    print(f"✅ PyTorch线程数设置为: {torch.get_num_threads()}")
    return cpu_count

class MultiCoreDataProcessor:
    """多核数据处理器"""
    
    def __init__(self, num_workers=None):
        if num_workers is None:
            self.num_workers = min(16, psutil.cpu_count(logical=True))
        else:
            self.num_workers = num_workers
        
        print(f"🚀 多核处理器初始化，使用 {self.num_workers} 个进程")
    
    def parallel_load_files(self, file_paths, target_size=(128, 128)):
        """并行加载多个文件"""
        print(f"📂 并行加载 {len(file_paths)} 个文件...")
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            load_func = partial(self._load_single_file, target_size=target_size)
            results = list(executor.map(load_func, file_paths))
        
        # 过滤失败的结果
        valid_results = [r for r in results if r is not None]
        print(f"✅ 成功加载 {len(valid_results)} / {len(file_paths)} 个文件")
        
        return valid_results
    
    @staticmethod
    def _load_single_file(file_path, target_size):
        """加载单个文件（静态方法，用于多进程）"""
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['data']
                n_timesteps = data.shape[0]
                
                # 采样时间步
                step = max(1, n_timesteps // 5)  # 每个文件5个样本
                samples = []
                
                for t in range(0, min(n_timesteps, 50), step):  # 最多50个时间步
                    try:
                        sample_data = data[t]  # (C, H, W)
                        samples.append({
                            'data': sample_data,
                            'file_path': str(file_path),
                            'timestep': t
                        })
                    except Exception:
                        continue
                
                return samples
                
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return None
    
    def parallel_preprocess_data(self, raw_samples, target_size=(128, 128)):
        """并行预处理数据"""
        print(f"⚙️  并行预处理 {len(raw_samples)} 个样本...")
        
        # 使用线程池（因为主要是numpy操作）
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            preprocess_func = partial(self._preprocess_single_sample, target_size=target_size)
            results = list(executor.map(preprocess_func, raw_samples))
        
        # 过滤有效结果
        valid_results = [r for r in results if r is not None]
        print(f"✅ 成功预处理 {len(valid_results)} / {len(raw_samples)} 个样本")
        
        return valid_results
    
    @staticmethod
    def _preprocess_single_sample(sample, target_size):
        """预处理单个样本"""
        try:
            data = sample['data']  # (C, H, W)
            
            # 转换为torch张量
            data_tensor = torch.from_numpy(data).float()
            
            # Resize
            if data_tensor.shape[1:] != target_size:
                data_resized = F.interpolate(
                    data_tensor.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                data_resized = data_tensor
            
            # 分离特征和目标
            features = data_resized[:22]  # 前22个通道
            target = data_resized[22:23]  # 火点置信度
            
            # 数据清洗
            features = torch.where(torch.isfinite(features), features, torch.tensor(0.0))
            target = torch.where(torch.isfinite(target), target, torch.tensor(0.0))
            target = (target > 0).float()
            
            return {
                'features': features,
                'target': target,
                'metadata': sample
            }
            
        except Exception as e:
            return None

class CPUOptimizedDataset(Dataset):
    """CPU优化的数据集"""
    
    def __init__(self, data_dir, mode='train', max_files=200, target_size=(128, 128), 
                 num_workers=16, preload_data=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        self.num_workers = num_workers
        
        print(f"🔥 初始化CPU优化数据集 ({mode}模式)")
        print(f"🖥️  使用 {num_workers} 个CPU核心")
        
        # 多核数据处理器
        self.processor = MultiCoreDataProcessor(num_workers)
        
        # 收集文件
        self._collect_files(max_files)
        
        # 预加载数据（可选）
        if preload_data:
            self._preload_all_data()
        else:
            self.samples = []
            self._build_sample_index()
        
        print(f"✅ 数据集初始化完成: {len(self)} 个样本")
    
    def _collect_files(self, max_files):
        """收集文件"""
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        # 限制文件数量并分割
        files_to_use = all_files[:max_files]
        n_files = len(files_to_use)
        
        if self.mode == 'train':
            self.files = files_to_use[:int(0.8 * n_files)]
        else:  # val
            self.files = files_to_use[int(0.8 * n_files):]
        
        print(f"📁 {self.mode}模式使用 {len(self.files)} 个文件")
    
    def _preload_all_data(self):
        """预加载所有数据到内存"""
        print("🚀 开始预加载数据...")
        start_time = time.time()
        
        # 并行加载原始数据
        raw_samples = self.processor.parallel_load_files(self.files, self.target_size)
        
        # 展平样本列表
        all_raw_samples = []
        for file_samples in raw_samples:
            if file_samples:
                all_raw_samples.extend(file_samples)
        
        print(f"📊 总共收集到 {len(all_raw_samples)} 个原始样本")
        
        # 并行预处理
        self.samples = self.processor.parallel_preprocess_data(all_raw_samples, self.target_size)
        
        # 计算预处理统计
        self._compute_normalization_stats()
        
        # 应用标准化
        self._normalize_samples()
        
        load_time = time.time() - start_time
        print(f"⏱️  数据预加载完成，耗时: {load_time:.2f}秒")
        print(f"📊 数据加载速度: {len(self.samples) / load_time:.1f} 样本/秒")
    
    def _compute_normalization_stats(self):
        """计算标准化统计量"""
        print("📊 计算标准化统计量...")
        
        # 收集所有特征数据
        all_features = []
        sample_size = min(1000, len(self.samples))
        
        for i in range(0, sample_size, 10):
            sample = self.samples[i]
            features = sample['features'][:21]  # 连续特征（排除land cover）
            all_features.append(features.flatten())
        
        if all_features:
            all_features = torch.cat(all_features)
            
            # 计算统计量
            self.feature_mean = float(torch.median(all_features))
            self.feature_std = float(torch.std(all_features))
            self.feature_min = float(torch.quantile(all_features, 0.05))
            self.feature_max = float(torch.quantile(all_features, 0.95))
            
            print(f"✅ 特征统计: mean={self.feature_mean:.3f}, std={self.feature_std:.3f}")
        else:
            self.feature_mean = 0.0
            self.feature_std = 1.0
            self.feature_min = -5.0
            self.feature_max = 5.0
    
    def _normalize_samples(self):
        """标准化所有样本"""
        print("🔧 标准化样本...")
        
        def normalize_single_sample(sample):
            features = sample['features']
            
            # 分离连续特征和土地覆盖
            continuous_features = features[:21]
            landcover = features[21:22]
            
            # 标准化连续特征
            continuous_features = torch.clamp(continuous_features, self.feature_min, self.feature_max)
            continuous_features = (continuous_features - self.feature_mean) / (self.feature_std + 1e-8)
            
            # 清理土地覆盖
            landcover = torch.clamp(landcover, 1, 16) - 1  # 转换为[0,15]
            
            # 重新组合
            sample['features'] = torch.cat([continuous_features, landcover], dim=0)
            return sample
        
        # 并行标准化
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            self.samples = list(executor.map(normalize_single_sample, self.samples))
        
        print("✅ 样本标准化完成")
    
    def _build_sample_index(self):
        """构建样本索引（如果不预加载数据）"""
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_timesteps = f['data'].shape[0]
                    step = max(1, n_timesteps // 10)
                    for t in range(0, n_timesteps, step):
                        self.samples.append((file_idx, t))
            except Exception:
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if isinstance(sample, dict):
            # 预加载模式
            return sample['features'], sample['target']
        else:
            # 动态加载模式（不推荐，这里为完整性保留）
            file_idx, timestep = sample
            return self._load_sample_dynamically(file_idx, timestep)
    
    def _load_sample_dynamically(self, file_idx, timestep):
        """动态加载样本（备用方案）"""
        try:
            file_path = self.files[file_idx]
            with h5py.File(file_path, 'r') as f:
                data = torch.from_numpy(f['data'][timestep]).float()
                features = data[:22]
                target = data[22:23]
                target = (target > 0).float()
                return features, target
        except Exception:
            return torch.zeros(22, *self.target_size), torch.zeros(1, *self.target_size)

class CPUOptimizedCNN(nn.Module):
    """CPU优化的CNN模型"""
    
    def __init__(self, continuous_channels=21, landcover_classes=16, embed_dim=4):
        super().__init__()
        
        # 减少嵌入维度以适应CPU
        self.landcover_embedding = nn.Embedding(landcover_classes, embed_dim)
        
        total_channels = continuous_channels + embed_dim
        
        # CPU友好的轻量级架构
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(total_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        
        # 轻量级解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),   # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),   # 32 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 8, 2, stride=2),    # 64 -> 128
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 分离连续特征和土地覆盖
        continuous_features = x[:, :21]
        landcover = x[:, 21].long()
        
        # 土地覆盖嵌入
        landcover_embedded = self.landcover_embedding(landcover).permute(0, 3, 1, 2)
        
        # 合并特征
        combined_features = torch.cat([continuous_features, landcover_embedded], dim=1)
        
        # 编码-解码
        encoded = self.encoder(combined_features)
        decoded = self.decoder(encoded)
        
        return decoded

class CPUOptimizedTrainer:
    """CPU优化训练器"""
    
    def __init__(self, model, train_loader, val_loader, num_workers=16):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_workers = num_workers
        
        # CPU优化设置
        optimize_cpu_settings()
        
        # 优化器（CPU友好参数）
        self.criterion = nn.BCELoss()
        self.optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6
        )
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.best_val_loss = float('inf')
        
        # 性能监控
        self.epoch_times = []
        self.cpu_usage = []
        
        print(f"🚀 CPU优化训练器初始化完成")
        print(f"🖥️  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """CPU优化训练epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        epoch_start_time = time.time()
        
        # 监控CPU使用率
        cpu_percent_start = psutil.cpu_percent(interval=None)
        
        print_interval = max(1, len(self.train_loader) // 10)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 数据验证
            if torch.isnan(data).any() or torch.isnan(target).any():
                continue
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # 进度显示
            if batch_idx % print_interval == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = data.size(0) / batch_time
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, "
                      f"Speed: {samples_per_sec:.1f} samples/s")
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        cpu_percent_end = psutil.cpu_percent(interval=None)
        
        self.epoch_times.append(epoch_time)
        self.cpu_usage.append((cpu_percent_start + cpu_percent_end) / 2)
        
        print(f"\n📊 Epoch {epoch} 统计:")
        print(f"  训练时间: {epoch_time:.2f}s")
        print(f"  CPU使用率: {self.cpu_usage[-1]:.1f}%")
        print(f"  吞吐量: {len(self.train_loader.dataset) / epoch_time:.1f} samples/s")
        
        return total_loss / max(valid_batches, 1)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        valid_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    all_preds.append(output.numpy().flatten())
                    all_targets.append(target.numpy().flatten())
        
        # 计算AUPRC
        if all_preds and all_targets:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            try:
                auprc = average_precision_score(targets, preds)
            except:
                auprc = 0.0
        else:
            auprc = 0.0
        
        return total_loss / max(valid_batches, 1), auprc
    
    def train(self, num_epochs=25):
        """完整训练流程"""
        print(f"🔥 开始CPU优化训练 {num_epochs} epochs")
        print(f"🖥️  CPU核心数: {psutil.cpu_count(logical=True)}")
        print(f"🧠 可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        save_dir = Path('cpu_optimized_outputs')
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
            print("=" * 50)
            
            # 训练和验证
            train_loss = self.train_epoch(epoch+1)
            val_loss, val_auprc = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_auprcs.append(val_auprc)
            
            print(f"📊 Results:")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  验证AUPRC: {val_auprc:.4f}")
            print(f"  学习率: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc,
                    'cpu_usage': self.cpu_usage,
                    'epoch_times': self.epoch_times
                }, save_dir / 'best_cpu_model.pth')
                print("💾 保存最佳模型")
            
            # 内存清理
            gc.collect()
        
        # 性能统计
        self._print_performance_stats()
        
        # 绘制训练曲线
        self.plot_curves(save_dir)
        print(f"\n✅ 训练完成!")
        
        return self.best_val_loss
    
    def _print_performance_stats(self):
        """打印性能统计"""
        print(f"\n📊 性能统计:")
        print(f"  平均epoch时间: {np.mean(self.epoch_times):.2f}s")
        print(f"  平均CPU使用率: {np.mean(self.cpu_usage):.1f}%")
        print(f"  总训练时间: {sum(self.epoch_times):.2f}s")
        print(f"  最佳验证损失: {self.best_val_loss:.6f}")
        print(f"  最佳AUPRC: {max(self.val_auprcs):.4f}")
    
    def plot_curves(self, save_dir):
        """绘制训练曲线和性能图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUPRC
        axes[0, 1].plot(epochs, self.val_auprcs, 'g-', label='验证AUPRC')
        axes[0, 1].set_title('AUPRC曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CPU使用率
        axes[0, 2].plot(epochs, self.cpu_usage, 'orange', label='CPU使用率')
        axes[0, 2].set_title('CPU使用率')
        axes[0, 2].set_ylabel('使用率 (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Epoch时间
        axes[1, 0].plot(epochs, self.epoch_times, 'purple', label='Epoch时间')
        axes[1, 0].set_title('每Epoch训练时间')
        axes[1, 0].set_ylabel('时间 (秒)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 吞吐量
        throughput = [len(self.train_loader.dataset) / t for t in self.epoch_times]
        axes[1, 1].plot(epochs, throughput, 'brown', label='训练吞吐量')
        axes[1, 1].set_title('训练吞吐量')
        axes[1, 1].set_ylabel('样本/秒')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 统计信息
        stats_text = f"""
最佳验证损失: {self.best_val_loss:.6f}
最佳AUPRC: {max(self.val_auprcs):.4f}
平均epoch时间: {np.mean(self.epoch_times):.2f}s
平均CPU使用率: {np.mean(self.cpu_usage):.1f}%
总训练时间: {sum(self.epoch_times)/3600:.2f}h
CPU核心数: {psutil.cpu_count(logical=True)}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, fontsize=10)
        axes[1, 2].set_title('训练统计')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'cpu_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数 - 16核CPU优化版"""
    print("🔥 16核CPU优化野火训练系统")
    print("=" * 60)
    
    # CPU优化配置
    cpu_count = optimize_cpu_settings()
    
    config = {
        'data_dir': 'data/processed',
        'batch_size': 16,                    # CPU可以处理更大batch
        'num_epochs': 25,
        'max_files': 300,                    # 增加文件数量
        'target_size': (128, 128),
        'num_workers': cpu_count,            # 使用所有CPU核心
        'preload_data': True,                # 预加载数据到内存
        'device': 'cpu'
    }
    
    print(f"\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查数据目录
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建CPU优化数据集
    print(f"\n1. 创建CPU优化数据集...")
    train_dataset = CPUOptimizedDataset(
        config['data_dir'],
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size'],
    train_dataset = CPUOptimizedDataset(
        config['data_dir'],
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size'],
        num_workers=config['num_workers'],
        preload_data=config['preload_data']
    )
    
    val_dataset = CPUOptimizedDataset(
        config['data_dir'],
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size'],
        num_workers=config['num_workers'],
        preload_data=config['preload_data']
    )
    
    # 复制标准化参数
    if hasattr(train_dataset, 'feature_mean'):
        val_dataset.feature_mean = train_dataset.feature_mean
        val_dataset.feature_std = train_dataset.feature_std
        val_dataset.feature_min = train_dataset.feature_min
        val_dataset.feature_max = train_dataset.feature_max
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器（CPU优化）
    print(f"\n2. 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=min(8, config['num_workers']//2),  # 为数据加载分配一半核心
        pin_memory=False,                              # CPU训练不需要pin_memory
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=min(4, config['num_workers']//4),
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"✅ 训练批次数: {len(train_loader)}")
    print(f"✅ 验证批次数: {len(val_loader)}")
    
    # 创建CPU优化模型
    print(f"\n3. 创建CPU优化模型...")
    model = CPUOptimizedCNN(
        continuous_channels=21,
        landcover_classes=16,
        embed_dim=4  # 减少嵌入维度以适应CPU
    )
    
    print(f"✅ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    print(f"\n4. 创建CPU优化训练器...")
    trainer = CPUOptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_workers=config['num_workers']
    )
    
    # 性能测试
    print(f"\n5. 性能测试...")
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            print(f"📊 数据统计:")
            print(f"  输入形状: {data.shape}")
            print(f"  目标形状: {target.shape}")
            print(f"  输入范围: [{data.min().item():.3f}, {data.max().item():.3f}]")
            print(f"  目标范围: [{target.min().item():.3f}, {target.max().item():.3f}]")
            
            # 测试推理速度
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            
            print(f"  推理时间: {inference_time:.3f}s")
            print(f"  推理速度: {data.size(0) / inference_time:.1f} samples/s")
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            break
    
    # 开始训练
    print(f"\n6. 开始CPU优化训练...")
    print(f"🖥️  系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count(logical=True)}")
    print(f"  可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"  PyTorch线程数: {torch.get_num_threads()}")
    
    # 训练
    best_loss = trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\n🎉 训练完成!")
    print(f"💾 最佳模型: cpu_optimized_outputs/best_cpu_model.pth")
    print(f"📈 训练曲线: cpu_optimized_outputs/cpu_training_curves.png")
    print(f"🏆 最佳验证损失: {best_loss:.6f}")


def benchmark_cpu_performance():
    """CPU性能基准测试"""
    print("\n🔬 CPU性能基准测试")
    print("=" * 40)
    
    # 测试不同核心数的性能
    cpu_counts = [1, 4, 8, 16]
    results = {}
    
    for num_cores in cpu_counts:
        if num_cores <= psutil.cpu_count(logical=True):
            print(f"\n测试 {num_cores} 核心...")
            
            # 设置线程数
            torch.set_num_threads(num_cores)
            
            # 创建测试模型和数据
            model = CPUOptimizedCNN()
            test_data = torch.randn(8, 22, 128, 128)
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_data)
            
            # 计时测试
            start_time = time.time()
            with torch.no_grad():
                for _ in range(50):
                    _ = model(test_data)
            
            total_time = time.time() - start_time
            avg_time = total_time / 50
            throughput = test_data.size(0) / avg_time
            
            results[num_cores] = {
                'avg_time': avg_time,
                'throughput': throughput
            }
            
            print(f"  平均推理时间: {avg_time:.4f}s")
            print(f"  吞吐量: {throughput:.1f} samples/s")
    
    # 显示结果
    print(f"\n📊 性能对比:")
    print("核心数 | 推理时间(s) | 吞吐量(samples/s) | 相对加速")
    print("-" * 50)
    
    baseline_time = results[1]['avg_time'] if 1 in results else None
    
    for cores, result in results.items():
        speedup = baseline_time / result['avg_time'] if baseline_time else 1.0
        print(f"{cores:6d} | {result['avg_time']:11.4f} | {result['throughput']:15.1f} | {speedup:8.2f}x")


def optimize_for_specific_cpu():
    """针对特定CPU架构优化"""
    print("\n🔧 CPU架构优化")
    print("=" * 40)
    
    # 检测CPU信息
    cpu_info = {}
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'model name' in line:
                    cpu_info['name'] = line.split(':')[1].strip()
                    break
    except:
        cpu_info['name'] = "Unknown"
    
    cpu_info['physical_cores'] = psutil.cpu_count(logical=False)
    cpu_info['logical_cores'] = psutil.cpu_count(logical=True)
    cpu_info['frequency'] = psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
    
    print(f"CPU信息:")
    print(f"  型号: {cpu_info['name']}")
    print(f"  物理核心: {cpu_info['physical_cores']}")
    print(f"  逻辑核心: {cpu_info['logical_cores']}")
    print(f"  最大频率: {cpu_info['frequency']} MHz")
    
    # 基于CPU特性提供优化建议
    suggestions = []
    
    if cpu_info['logical_cores'] >= 16:
        suggestions.append("✅ 高核心数CPU，适合大batch size和多进程数据加载")
        suggestions.append("💡 建议: batch_size=16-32, num_workers=12-16")
    
    if cpu_info['logical_cores'] > cpu_info['physical_cores']:
        suggestions.append("✅ 支持超线程，可以设置线程数=逻辑核心数")
        suggestions.append("💡 建议: torch.set_num_threads(logical_cores)")
    
    # Intel特定优化
    if 'Intel' in cpu_info['name']:
        suggestions.append("✅ Intel CPU，可启用MKL优化")
        suggestions.append("💡 建议: 设置MKL_NUM_THREADS=物理核心数")
    
    # AMD特定优化
    if 'AMD' in cpu_info['name']:
        suggestions.append("✅ AMD CPU，建议使用OpenBLAS")
        suggestions.append("💡 建议: 设置OMP_NUM_THREADS=物理核心数")
    
    print(f"\n优化建议:")
    for suggestion in suggestions:
        print(f"  {suggestion}")


def memory_usage_monitor():
    """内存使用监控"""
    print(f"\n💾 内存使用情况:")
    
    # 系统内存
    memory = psutil.virtual_memory()
    print(f"  系统内存:")
    print(f"    总计: {memory.total / 1024**3:.1f} GB")
    print(f"    可用: {memory.available / 1024**3:.1f} GB")
    print(f"    使用率: {memory.percent:.1f}%")
    
    # 进程内存
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"  进程内存:")
    print(f"    RSS: {process_memory.rss / 1024**2:.1f} MB")
    print(f"    VMS: {process_memory.vms / 1024**2:.1f} MB")
    
    # PyTorch内存（如果使用CUDA）
    if torch.cuda.is_available():
        print(f"  GPU内存:")
        print(f"    已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"    缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


if __name__ == "__main__":
    print("🔥 多核CPU野火训练系统")
    print("=" * 60)
    
    # 系统信息
    print(f"🖥️  系统信息:")
    print(f"  Python版本: {os.sys.version}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CPU核心数: {psutil.cpu_count(logical=True)}")
    print(f"  可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # CPU架构优化
    optimize_for_specific_cpu()
    
    # 内存监控
    memory_usage_monitor()
    
    # 性能基准测试(可选)
    benchmark_choice = input("\n是否运行性能基准测试? (y/N): ").lower()
    if benchmark_choice == 'y':
        benchmark_cpu_performance()
    
    # 运行主程序
    print(f"\n" + "="*60)
    main()
    
    # 最终内存监控
    print(f"\n最终内存使用:")
    memory_usage_monitor()
    
    print(f"\n🎉 程序完成!")
    print(f"💡 提示: 如果训练速度仍然较慢，可以:")
    print(f"  1. 减少图像尺寸 (如128x128 -> 96x96)")
    print(f"  2. 增加batch size")
    print(f"  3. 减少模型复杂度")
    print(f"  4. 使用数据预处理缓存")