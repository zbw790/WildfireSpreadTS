#!/usr/bin/env python3
"""
优化版野火CNN - 解决训练速度递减问题
专门针对以下问题进行优化：
1. 内存泄漏
2. GPU内存碎片化
3. 梯度累积问题
4. 数据加载效率
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import h5py
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 优化CUDA设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

class OptimizedWildfireDataset(Dataset):
    """优化版数据集 - 防止内存泄漏"""
    
    def __init__(self, data_dir, mode='train', max_files=100, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        
        print(f"初始化{mode}数据集...")
        
        # 收集文件
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        # 限制文件数量
        files_to_use = all_files[:max_files]
        n_files = len(files_to_use)
        
        if mode == 'train':
            self.files = files_to_use[:int(0.8 * n_files)]
        else:
            self.files = files_to_use[int(0.8 * n_files):]
        
        print(f"{mode}模式: {len(self.files)} 文件")
        
        # 构建样本索引
        self.samples = []
        self._build_samples()
        
        # 预计算统计量（只训练集）
        if mode == 'train':
            self._compute_stats()
    
    def _build_samples(self):
        """高效构建样本索引"""
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_timesteps = f['data'].shape[0]
                    # 每个文件最多5个样本
                    step = max(1, n_timesteps // 5)
                    for t in range(0, n_timesteps, step):
                        self.samples.append((file_idx, t))
            except Exception as e:
                continue
        
        print(f"构建 {len(self.samples)} 个样本")
    
    def _compute_stats(self):
        """预计算数据统计量"""
        print("计算数据统计量...")
        
        sample_data = []
        for i in range(0, min(50, len(self.samples)), 5):
            try:
                file_idx, timestep = self.samples[i]
                with h5py.File(self.files[file_idx], 'r') as f:
                    data = f['data'][timestep][:22]  # 只取特征通道
                    sample_data.append(data.flatten())
            except:
                continue
        
        if sample_data:
            all_data = np.concatenate(sample_data)
            valid_data = all_data[np.isfinite(all_data)]
            
            if len(valid_data) > 0:
                self.data_mean = float(np.median(valid_data))
                self.data_std = float(np.std(valid_data))
                self.data_min = float(np.percentile(valid_data, 5))
                self.data_max = float(np.percentile(valid_data, 95))
            else:
                self.data_mean = 0.0
                self.data_std = 1.0
                self.data_min = -5.0
                self.data_max = 5.0
        else:
            self.data_mean = 0.0
            self.data_std = 1.0
            self.data_min = -5.0
            self.data_max = 5.0
        
        print(f"数据统计: mean={self.data_mean:.3f}, std={self.data_std:.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, timestep = self.samples[idx]
        
        try:
            with h5py.File(self.files[file_idx], 'r') as f:
                data = f['data'][timestep]
                
                # 快速提取和处理
                features = torch.from_numpy(data[:22]).float()
                target = torch.from_numpy(data[22:23]).float()
                
                # Resize
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
                
                # 快速标准化
                features = torch.clamp(features, self.data_min, self.data_max)
                features = (features - self.data_mean) / (self.data_std + 1e-8)
                
                # 二值化目标
                target = (target > 0).float()
                
                return features, target
                
        except Exception:
            # 返回零张量避免崩溃
            return (torch.zeros(22, *self.target_size), 
                   torch.zeros(1, *self.target_size))


class OptimizedCNN(nn.Module):
    """优化的CNN模型 - 内存高效"""
    
    def __init__(self, input_channels=22):
        super().__init__()
        
        # 简化架构以提高速度
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, 3, padding=1),
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


class OptimizedTrainer:
    """优化的训练器 - 解决速度递减问题"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器设置
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.best_val_loss = float('inf')
        
        # 性能监控
        self.epoch_times = []
        
        print(f"优化训练器初始化 - 设备: {device}")
        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self):
        """优化的训练epoch - 防止内存泄漏"""
        self.model.train()
        
        total_loss = 0.0
        valid_batches = 0
        
        # 每个epoch开始时清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 数据有效性检查
            if torch.isnan(data).any() or torch.isnan(target).any():
                continue
            
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # 重要：确保梯度完全清零
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失（使用.item()避免保留计算图）
            total_loss += loss.item()
            valid_batches += 1
            
            # 定期清理内存
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
            
            # 显式删除变量释放内存
            del output, loss
        
        return total_loss / max(valid_batches, 1)
    
    def validate(self):
        """优化的验证 - 高效内存管理"""
        self.model.eval()
        
        total_loss = 0.0
        valid_batches = 0
        
        # 使用列表而不是不断append，更高效
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # 高效收集预测结果
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                
                # 清理GPU内存
                del output, loss
        
        # 计算AUPRC
        if all_preds and all_targets:
            preds = np.concatenate(all_preds).flatten()
            targets = np.concatenate(all_targets).flatten()
            
            try:
                auprc = average_precision_score(targets, preds)
            except:
                auprc = 0.0
            
            # 清理内存
            del preds, targets, all_preds, all_targets
        else:
            auprc = 0.0
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss, auprc
    
    def train(self, num_epochs=20):
        """优化的训练流程"""
        print(f"开始优化训练 {num_epochs} epochs")
        
        save_dir = Path('optimized_wildfire_outputs')
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_auprc = self.validate()
            
            # 记录时间
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_auprcs.append(val_auprc)
            
            # 打印结果
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证AUPRC: {val_auprc:.4f}")
            print(f"Epoch时间: {epoch_time:.1f}s")
            
            # 性能监控
            if len(self.epoch_times) >= 3:
                recent_avg = np.mean(self.epoch_times[-3:])
                initial_avg = np.mean(self.epoch_times[:3])
                if recent_avg > initial_avg * 1.5:
                    print(f"⚠️  训练速度下降检测! 初始: {initial_avg:.1f}s, 最近: {recent_avg:.1f}s")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc,
                    'epoch_times': self.epoch_times
                }, save_dir / 'best_model.pth')
                print("💾 保存最佳模型")
            
            # 定期清理内存
            if (epoch + 1) % 5 == 0:
                print("🧹 清理内存...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 绘制性能分析
        self.plot_performance_analysis(save_dir)
        print(f"\n✅ 优化训练完成!")
    
    def plot_performance_analysis(self, save_dir):
        """绘制性能分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='训练损失')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUPRC曲线
        axes[0, 1].plot(epochs, self.val_auprcs, 'g-', label='验证AUPRC')
        axes[0, 1].set_title('AUPRC曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUPRC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Epoch时间分析
        if self.epoch_times:
            axes[1, 0].plot(epochs, self.epoch_times, 'purple', marker='o', markersize=4)
            axes[1, 0].set_title('每Epoch训练时间')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('时间 (秒)')
            axes[1, 0].grid(True)
            
            # 添加趋势线
            if len(self.epoch_times) > 1:
                z = np.polyfit(epochs, self.epoch_times, 1)
                p = np.poly1d(z)
                axes[1, 0].plot(epochs, p(epochs), 'r--', alpha=0.8, label=f'趋势: {z[0]:.2f}s/epoch')
                axes[1, 0].legend()
        
        # 性能统计
        if self.epoch_times:
            avg_time = np.mean(self.epoch_times)
            min_time = np.min(self.epoch_times)
            max_time = np.max(self.epoch_times)
            std_time = np.std(self.epoch_times)
            
            stats_text = f"""性能统计:
平均时间: {avg_time:.1f}s
最短时间: {min_time:.1f}s  
最长时间: {max_time:.1f}s
标准差: {std_time:.1f}s
变化率: {((max_time-min_time)/min_time*100):.1f}%"""
            
            axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='center')
            axes[1, 1].set_title('性能统计')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """优化主函数"""
    print("🚀 优化版野火CNN - 解决训练速度递减问题")
    print("=" * 60)
    print("优化内容:")
    print("✅ 防止内存泄漏")
    print("✅ GPU内存管理")
    print("✅ 梯度清理优化")
    print("✅ 性能监控")
    print("=" * 60)
    
    # 配置
    config = {
        'data_dir': 'data/processed',
        'batch_size': 4,
        'num_epochs': 20,
        'max_files': 100,
        'target_size': (128, 128),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n配置: {json.dumps(config, indent=2)}")
    
    # 检查数据
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建数据集
    print("\n1. 创建优化数据集...")
    train_dataset = OptimizedWildfireDataset(
        config['data_dir'], 
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size']
    )
    
    val_dataset = OptimizedWildfireDataset(
        config['data_dir'], 
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size']
    )
    
    # 复制统计量
    if hasattr(train_dataset, 'data_mean'):
        val_dataset.data_mean = train_dataset.data_mean
        val_dataset.data_std = train_dataset.data_std
        val_dataset.data_min = train_dataset.data_min
        val_dataset.data_max = train_dataset.data_max
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器（优化设置）
    print("\n2. 创建优化数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Windows上设为0
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True  # 避免不完整batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    
    # 创建模型
    print("\n3. 创建优化模型...")
    model = OptimizedCNN(input_channels=22)
    
    # 创建训练器
    print("\n4. 创建优化训练器...")
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device']
    )
    
    # 开始训练
    print("\n5. 开始优化训练...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n🎉 优化训练完成!")
    print("📁 结果: optimized_wildfire_outputs/")
    print("📊 性能分析: performance_analysis.png")


if __name__ == "__main__":
    main() 