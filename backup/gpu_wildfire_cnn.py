#!/usr/bin/env python3
"""
GPU优化的野火CNN训练系统
专门解决训练速度递减问题，充分利用RTX 4070 Ti的12GB显存
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# GPU优化设置
torch.backends.cudnn.benchmark = True  # 自动优化卷积算法
torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提高性能

class GPUOptimizedWildfireDataset(Dataset):
    """GPU优化的野火数据集"""
    
    def __init__(self, data_dir, mode='train', max_files=300, target_size=(256, 256)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        
        print(f"🔥 初始化{mode}数据集 (GPU优化版)...")
        
        # 收集HDF5文件
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        # 使用更多文件充分利用GPU
        files_to_use = all_files[:max_files]
        n_files = len(files_to_use)
        
        if mode == 'train':
            self.files = files_to_use[:int(0.8 * n_files)]
        else:
            self.files = files_to_use[int(0.8 * n_files):]
        
        print(f"📁 {mode}模式: {len(self.files)} 文件")
        
        # 构建样本索引
        self.samples = []
        self._build_samples()
        
        # 预计算统计量
        if mode == 'train':
            self._compute_stats()
    
    def _build_samples(self):
        """高效构建样本索引"""
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_timesteps = f['data'].shape[0]
                    # 每个文件采样更多样本
                    step = max(1, n_timesteps // 8)
                    for t in range(0, n_timesteps, step):
                        self.samples.append((file_idx, t))
            except:
                continue
        
        print(f"📊 构建 {len(self.samples)} 个样本")
    
    def _compute_stats(self):
        """预计算数据统计量"""
        print("📈 计算数据统计量...")
        
        sample_data = []
        for i in range(0, min(100, len(self.samples)), 5):
            try:
                file_idx, timestep = self.samples[i]
                with h5py.File(self.files[file_idx], 'r') as f:
                    data = f['data'][timestep][:22]
                    sample_data.append(data.flatten())
            except:
                continue
        
        if sample_data:
            all_data = np.concatenate(sample_data)
            valid_data = all_data[np.isfinite(all_data)]
            
            if len(valid_data) > 0:
                self.data_mean = float(np.median(valid_data))
                self.data_std = float(np.std(valid_data))
                self.data_min = float(np.percentile(valid_data, 1))
                self.data_max = float(np.percentile(valid_data, 99))
            else:
                self._set_default_stats()
        else:
            self._set_default_stats()
        
        print(f"📊 统计: mean={self.data_mean:.3f}, std={self.data_std:.3f}")
    
    def _set_default_stats(self):
        """设置默认统计量"""
        self.data_mean = 0.0
        self.data_std = 1.0
        self.data_min = -5.0
        self.data_max = 5.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, timestep = self.samples[idx]
        
        try:
            with h5py.File(self.files[file_idx], 'r') as f:
                data = f['data'][timestep]
                
                # 快速提取特征和目标
                features = torch.from_numpy(data[:22]).float()
                target = torch.from_numpy(data[22:23]).float()
                
                # GPU优化的resize
                features = F.interpolate(
                    features.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                target = F.interpolate(
                    target.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 高效数据清洗
                features = torch.clamp(features, self.data_min, self.data_max)
                features = (features - self.data_mean) / (self.data_std + 1e-8)
                target = (target > 0).float()
                
                return features, target
                
        except:
            return (torch.zeros(22, *self.target_size), 
                   torch.zeros(1, *self.target_size))


class GPUOptimizedCNN(nn.Module):
    """GPU优化的CNN模型 - 充分利用RTX 4070 Ti"""
    
    def __init__(self, input_channels=22, base_channels=64):
        super().__init__()
        
        # 增大网络容量充分利用GPU
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            # Upsample 1
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Upsample 2
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Upsample 3
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Upsample 4
            nn.ConvTranspose2d(base_channels, base_channels // 2, 2, stride=2),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡 - 修复混合精度兼容性"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 使用logits版本的BCE，兼容混合精度
        # 假设输入pred是经过sigmoid的概率，需要转换回logits
        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        logits = torch.log(pred / (1 - pred))  # 反sigmoid变换
        
        # 计算BCE with logits (混合精度安全)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # 计算pt (使用原始概率)
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # 计算alpha_t
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Focal权重
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


class GPUOptimizedTrainer:
    """GPU优化的训练器 - 解决速度递减问题"""
    
    def __init__(self, model, train_loader, val_loader):
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化的损失函数和优化器
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer = AdamW(
            model.parameters(), 
            lr=2e-4,  # 更大的学习率利用GPU
            weight_decay=1e-4,
            eps=1e-8
        )
        
        # Cosine学习率调度
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.epoch_times = []
        self.best_val_loss = float('inf')
        
        print(f"🚀 GPU优化训练器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"🧠 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    def train_epoch(self):
        """GPU优化的训练epoch"""
        self.model.train()
        
        total_loss = 0.0
        valid_batches = 0
        
        # 每个epoch开始清理GPU缓存
        torch.cuda.empty_cache()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 数据有效性快速检查
            if torch.isnan(data).any() or torch.isnan(target).any():
                continue
            
            # 非阻塞GPU传输
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # 重要：完全清零梯度
            self.optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # 损失有效性检查
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 混合精度反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪和优化器步骤
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            total_loss += loss.item()
            valid_batches += 1
            
            # 定期清理和打印
            if batch_idx % 25 == 0:
                torch.cuda.empty_cache()
                print(f"  📊 Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, "
                      f"GPU内存: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")
            
            # 显式删除变量
            del output, loss
        
        return total_loss / max(valid_batches, 1)
    
    def validate(self):
        """GPU优化的验证"""
        self.model.eval()
        
        total_loss = 0.0
        valid_batches = 0
        
        # 使用更高效的预测收集方式
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # 高效收集预测结果
                    all_preds.append(output.detach().cpu().numpy())
                    all_targets.append(target.detach().cpu().numpy())
                
                del output, loss
        
        # 计算AUPRC
        if all_preds and all_targets:
            preds = np.concatenate(all_preds).flatten()
            targets = np.concatenate(all_targets).flatten()
            
            try:
                auprc = average_precision_score(targets, preds)
            except:
                auprc = 0.0
            
            del preds, targets, all_preds, all_targets
        else:
            auprc = 0.0
        
        # 强制清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss, auprc
    
    def train(self, num_epochs=30):
        """GPU优化的训练流程"""
        print(f"🚀 开始GPU优化训练 {num_epochs} epochs")
        print(f"💾 GPU: {torch.cuda.get_device_name()}")
        
        save_dir = Path('gpu_wildfire_outputs')
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # 训练和验证
            train_loss = self.train_epoch()
            val_loss, val_auprc = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录时间和指标
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_auprcs.append(val_auprc)
            
            # 打印结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"📈 训练损失: {train_loss:.6f}")
            print(f"📉 验证损失: {val_loss:.6f}")
            print(f"🎯 验证AUPRC: {val_auprc:.4f}")
            print(f"⏱️  Epoch时间: {epoch_time:.1f}s")
            print(f"🎛️  学习率: {current_lr:.2e}")
            print(f"💾 GPU内存峰值: {torch.cuda.max_memory_allocated() / (1024**3):.1f} GB")
            
            # 性能监控
            if len(self.epoch_times) >= 5:
                recent_avg = np.mean(self.epoch_times[-3:])
                initial_avg = np.mean(self.epoch_times[:3])
                if recent_avg > initial_avg * 1.3:
                    print(f"⚠️  训练速度下降: 初始 {initial_avg:.1f}s -> 当前 {recent_avg:.1f}s")
                else:
                    print(f"✅ 训练速度稳定: ~{recent_avg:.1f}s/epoch")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc,
                    'epoch_times': self.epoch_times
                }, save_dir / 'best_gpu_model.pth')
                print("💾 保存最佳模型")
            
            # 定期清理
            if (epoch + 1) % 5 == 0:
                print("🧹 清理GPU内存...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        
        # 绘制性能分析
        self.plot_performance_analysis(save_dir)
        print(f"\n🎉 GPU优化训练完成!")
        print(f"💰 最佳验证损失: {self.best_val_loss:.6f}")
    
    def plot_performance_analysis(self, save_dir):
        """绘制详细的性能分析"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_title('损失曲线', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUPRC曲线
        axes[0, 1].plot(epochs, self.val_auprcs, 'g-', label='验证AUPRC', linewidth=2)
        axes[0, 1].set_title('AUPRC曲线', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUPRC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练时间分析
        axes[0, 2].plot(epochs, self.epoch_times, 'purple', marker='o', markersize=3)
        axes[0, 2].set_title('每Epoch训练时间', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('时间 (秒)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加时间趋势线
        if len(self.epoch_times) > 1:
            z = np.polyfit(epochs, self.epoch_times, 1)
            p = np.poly1d(z)
            axes[0, 2].plot(epochs, p(epochs), 'r--', alpha=0.8, 
                           label=f'趋势: {z[0]:.3f}s/epoch')
            axes[0, 2].legend()
        
        # 学习率曲线
        lrs = [self.optimizer.param_groups[0]['lr']]
        for _ in range(1, len(epochs)):
            lrs.append(lrs[-1] * 0.99)  # 近似cosine调度
        
        axes[1, 0].plot(epochs, lrs, 'orange', linewidth=2)
        axes[1, 0].set_title('学习率调度', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 损失改善分析
        train_improvement = [(self.train_losses[0] - loss) / self.train_losses[0] * 100 
                           for loss in self.train_losses]
        val_improvement = [(self.val_losses[0] - loss) / self.val_losses[0] * 100 
                         for loss in self.val_losses]
        
        axes[1, 1].plot(epochs, train_improvement, 'b-', label='训练损失改善%')
        axes[1, 1].plot(epochs, val_improvement, 'r-', label='验证损失改善%')
        axes[1, 1].set_title('损失改善百分比', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('改善 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 统计信息
        if self.epoch_times:
            avg_time = np.mean(self.epoch_times)
            min_time = np.min(self.epoch_times)
            max_time = np.max(self.epoch_times)
            std_time = np.std(self.epoch_times)
            best_auprc = max(self.val_auprcs)
            
            stats_text = f"""🚀 GPU训练统计:
            
平均时间: {avg_time:.1f}s/epoch
最短时间: {min_time:.1f}s
最长时间: {max_time:.1f}s
时间标准差: {std_time:.1f}s
时间变化率: {((max_time-min_time)/min_time*100):.1f}%

最佳AUPRC: {best_auprc:.4f}
最佳验证损失: {self.best_val_loss:.6f}
总训练时间: {sum(self.epoch_times)/3600:.2f}小时

GPU: {torch.cuda.get_device_name()}
显存: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB"""
            
            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('GPU训练统计', fontsize=14)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'gpu_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """GPU优化主函数"""
    print("🔥 GPU优化野火CNN训练系统")
    print("=" * 60)
    print("🎯 专门解决训练速度递减问题")
    print("⚡ 充分利用RTX 4070 Ti 12GB显存")
    print("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ GPU不可用，请安装CUDA版本的PyTorch")
        return
    
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # GPU优化配置
    config = {
        'data_dir': 'data/processed',
        'batch_size': 8,  # 利用大显存
        'num_epochs': 30,
        'max_files': 300,  # 更多数据
        'target_size': (256, 256),  # 更大图像
        'base_channels': 64,  # 更大网络
    }
    
    print(f"\n📋 GPU优化配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 检查数据
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建GPU优化数据集
    print("\n📁 创建GPU优化数据集...")
    train_dataset = GPUOptimizedWildfireDataset(
        config['data_dir'], 
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size']
    )
    
    val_dataset = GPUOptimizedWildfireDataset(
        config['data_dir'], 
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size']
    )
    
    # 复制统计量
    if hasattr(train_dataset, 'data_mean'):
        for attr in ['data_mean', 'data_std', 'data_min', 'data_max']:
            setattr(val_dataset, attr, getattr(train_dataset, attr))
    
    print(f"📊 训练集: {len(train_dataset):,} 样本")
    print(f"📊 验证集: {len(val_dataset):,} 样本")
    
    # 创建GPU优化数据加载器
    print("\n⚡ 创建GPU优化数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,  # 多进程加载
        pin_memory=True,  # 固定内存
        persistent_workers=True,  # 持久化worker
        drop_last=True,
        prefetch_factor=2  # 预取因子
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建GPU优化模型
    print("\n🧠 创建GPU优化模型...")
    model = GPUOptimizedCNN(
        input_channels=22,
        base_channels=config['base_channels']
    )
    
    print(f"🔢 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建GPU优化训练器
    print("\n🚀 创建GPU优化训练器...")
    trainer = GPUOptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 开始GPU优化训练
    print("\n🔥 开始GPU优化训练...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n🎉 GPU优化训练完成!")
    print("📁 结果保存在: gpu_wildfire_outputs/")
    print("💾 最佳模型: best_gpu_model.pth")
    print("📊 性能分析: gpu_performance_analysis.png")


if __name__ == "__main__":
    main() 