#!/usr/bin/env python3
"""
WildfireSpreadTS CNN模型 v2.0
完整的野火传播预测深度学习系统

特点:
- 处理所有607个火灾事件
- U-Net架构适合分割任务
- Focal Loss处理类别不平衡
- 完整的训练/验证管道
- 支持时空序列预测

作者: AI Assistant
日期: 2025-01-30
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ================================
# 1. 数据加载器
# ================================

class WildfireDataset(Dataset):
    """野火数据集类"""
    
    def __init__(self, data_dir, sequence_length=5, prediction_horizon=1, 
                 train_split=0.8, mode='train', normalize=True):
        """
        Args:
            data_dir: HDF5文件目录
            sequence_length: 输入时间序列长度 
            prediction_horizon: 预测时间步数
            train_split: 训练集比例
            mode: 'train', 'val', 'test'
            normalize: 是否标准化
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.normalize = normalize
        
        # 找到所有HDF5文件
        self.hdf5_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                self.hdf5_files.extend(year_files)
        
        print(f"发现 {len(self.hdf5_files)} 个HDF5文件")
        
        # 按模式分割文件
        n_files = len(self.hdf5_files)
        train_end = int(n_files * train_split)
        val_end = int(n_files * (train_split + 0.1))
        
        if mode == 'train':
            self.files = self.hdf5_files[:train_end]
        elif mode == 'val':
            self.files = self.hdf5_files[train_end:val_end] 
        else:  # test
            self.files = self.hdf5_files[val_end:]
        
        print(f"{mode}模式: {len(self.files)} 个文件")
        
        # 构建样本索引
        self.samples = []
        self._build_samples()
        
        # 初始化标准化器
        if self.normalize and mode == 'train':
            self._initialize_normalizer()
    
    def _build_samples(self):
        """构建所有可用的样本索引"""
        print(f"构建{self.mode}样本索引...")
        
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data']
                    n_timesteps = data.shape[0]
                    
                    # 计算可用的样本数量
                    max_start = n_timesteps - self.sequence_length - self.prediction_horizon + 1
                    
                    if max_start > 0:
                        for start_idx in range(0, max_start, 2):  # 每2步采样一次
                            self.samples.append((file_idx, start_idx))
                            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        print(f"总计 {len(self.samples)} 个样本")
    
    def _initialize_normalizer(self):
        """初始化数据标准化器"""
        print("初始化数据标准化器...")
        
        # 采样一些数据来计算统计量
        sample_data = []
        n_samples = min(1000, len(self.samples))
        
        for i in range(0, n_samples, 10):
            try:
                sample = self[i]
                if sample is not None:
                    features = sample[0]  # (T, C, H, W)
                    # 排除土地覆盖类别特征 (channel 16)
                    mask = torch.ones(features.shape[1], dtype=torch.bool)
                    mask[16] = False
                    continuous_features = features[:, mask, :, :]
                    sample_data.append(continuous_features.numpy().flatten())
            except:
                continue
        
        if sample_data:
            all_data = np.concatenate(sample_data)
            self.scaler = RobustScaler()
            self.scaler.fit(all_data.reshape(-1, 1))
            print("标准化器初始化完成")
        else:
            self.scaler = None
            print("警告: 无法初始化标准化器")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        file_idx, start_idx = self.samples[idx]
        file_path = self.files[file_idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['data']  # (T, C, H, W)
                
                # 提取输入序列和目标
                end_idx = start_idx + self.sequence_length
                target_idx = end_idx + self.prediction_horizon - 1
                
                input_sequence = torch.from_numpy(data[start_idx:end_idx]).float()
                target = torch.from_numpy(data[target_idx, 22:23]).float()  # 火点置信度通道
                
                # 数据预处理
                input_sequence = self._preprocess_features(input_sequence)
                target = self._preprocess_target(target)
                
                return input_sequence, target
                
        except Exception as e:
            print(f"加载样本 {idx} 时出错: {e}")
            # 返回随机样本
            if len(self.samples) > 1:
                return self[(idx + 1) % len(self.samples)]
            else:
                return None
    
    def _preprocess_features(self, features):
        """预处理输入特征"""
        # features: (T, C, H, W)
        
        # 0. Resize到固定尺寸 (解决不同火灾事件尺寸不一致问题)
        target_size = (256, 256)  # 固定尺寸
        features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
        
        # 1. 处理土地覆盖类别特征 (channel 16)
        landcover = features[:, 16:17, :, :].clone()
        landcover = torch.clamp(landcover, 1, 16) - 1  # 转换为0-15范围
        
        # 2. 标准化连续特征
        continuous_features = torch.cat([
            features[:, :16, :, :], 
            features[:, 17:, :, :]
        ], dim=1)
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            original_shape = continuous_features.shape
            continuous_features = continuous_features.view(-1, 1)
            continuous_features = torch.from_numpy(
                self.scaler.transform(continuous_features.numpy())
            ).float()
            continuous_features = continuous_features.view(original_shape)
        
        # 3. 重新组合特征
        processed_features = torch.cat([
            continuous_features[:, :16, :, :],
            landcover,
            continuous_features[:, 16:, :, :]
        ], dim=1)
        
        return processed_features
    
    def _preprocess_target(self, target):
        """预处理目标变量"""
        # target: (1, H, W)
        
        # 0. Resize到固定尺寸
        target_size = (256, 256)
        target = F.interpolate(target.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        target = target.squeeze(0)
        
        # 1. 将目标转换为二值化 (>0 为火点)
        target = (target > 0).float()
        return target


# ================================
# 2. CNN模型架构
# ================================

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class WildfireCNN(nn.Module):
    """野火预测CNN模型 - 简化的U-Net架构"""
    
    def __init__(self, input_channels=23, sequence_length=5, base_channels=32):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        
        # 土地覆盖嵌入层
        self.landcover_embedding = nn.Embedding(16, 8)  # 16类 -> 8维
        
        # 计算实际输入通道数 (连续特征 + 嵌入特征)
        continuous_channels = input_channels - 1  # 排除土地覆盖
        embedded_channels = continuous_channels + 8  # 加上嵌入维度
        total_input_channels = embedded_channels * sequence_length
        
        # U-Net编码器
        self.encoder1 = ConvBlock(total_input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        
        # U-Net解码器
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(0.3)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size = x.size(0)
        
        # 处理时间序列
        processed_frames = []
        
        for t in range(self.sequence_length):
            frame = x[:, t]  # (B, C, H, W)
            
            # 分离土地覆盖和连续特征
            landcover = frame[:, 16].long()  # (B, H, W)
            continuous = torch.cat([frame[:, :16], frame[:, 17:]], dim=1)  # (B, 22, H, W)
            
            # 嵌入土地覆盖
            landcover_embedded = self.landcover_embedding(landcover)  # (B, H, W, 8)
            landcover_embedded = landcover_embedded.permute(0, 3, 1, 2)  # (B, 8, H, W)
            
            # 组合特征
            combined = torch.cat([continuous, landcover_embedded], dim=1)  # (B, 30, H, W)
            processed_frames.append(combined)
        
        # 连接所有时间步
        x = torch.cat(processed_frames, dim=1)  # (B, T*30, H, W)
        
        # U-Net前向传播
        # 编码器
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        # 瓶颈
        b = self.bottleneck(p3)
        b = self.dropout(b)
        
        # 解码器
        up3 = self.upconv3(b)
        merge3 = torch.cat([up3, e3], dim=1)
        d3 = self.decoder3(merge3)
        
        up2 = self.upconv2(d3)
        merge2 = torch.cat([up2, e2], dim=1)
        d2 = self.decoder2(merge2)
        
        up1 = self.upconv1(d2)
        merge1 = torch.cat([up1, e1], dim=1)
        d1 = self.decoder1(merge1)
        
        # 输出
        output = self.output(d1)
        
        return output


# ================================
# 3. 损失函数
# ================================

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        # pred: (B, 1, H, W), target: (B, 1, H, W)
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 确保pred在[0,1]范围内
        pred = torch.clamp(pred, 0.0001, 0.9999)
        
        # 计算BCE
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # 计算pt
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # 计算alpha_t
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # 计算focal权重
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # 应用focal权重
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ================================
# 4. 训练器
# ================================

class WildfireTrainer:
    """野火CNN训练器"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数和优化器
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # 最佳模型
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if data is None or target is None:
                continue
                
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Debug: 检查输出范围
            if batch_idx == 0:
                print(f"  Debug - Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
                print(f"  Debug - Target range: [{target.min().item():.6f}, {target.max().item():.6f}]")
                print(f"  Debug - Output shape: {output.shape}, Target shape: {target.shape}")
            
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if data is None or target is None:
                    continue
                    
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 收集预测结果
                all_preds.append(output.cpu().numpy().flatten())
                all_targets.append(target.cpu().numpy().flatten())
        
        # 计算指标
        if all_preds and all_targets:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            
            # 计算AUPRC
            try:
                auprc = average_precision_score(targets, preds)
            except:
                auprc = 0.0
            
            # 计算其他指标 (使用阈值0.5)
            pred_binary = (preds > 0.5).astype(int)
            try:
                f1 = f1_score(targets, pred_binary, zero_division=0)
                precision = precision_score(targets, pred_binary, zero_division=0)
                recall = recall_score(targets, pred_binary, zero_division=0)
            except:
                f1 = precision = recall = 0.0
            
            metrics = {
                'auprc': auprc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        else:
            metrics = {'auprc': 0, 'f1': 0, 'precision': 0, 'recall': 0}
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        return avg_loss, metrics
    
    def train(self, num_epochs=50, save_dir='wildfire_cnn_outputs'):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"开始训练 {num_epochs} 个epochs...")
        print(f"模型将保存到: {save_dir.absolute()}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, save_dir / 'best_model.pth')
            
            # 打印进度
            epoch_time = time.time() - epoch_start
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证AUPRC: {val_metrics['auprc']:.4f}")
            print(f"验证F1: {val_metrics['f1']:.4f}")
            print(f"Epoch时间: {epoch_time:.1f}s")
            
            # 每10个epoch保存一次checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_metrics': self.val_metrics
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time/3600:.2f}小时")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics': self.val_metrics,
                'best_val_loss': self.best_val_loss
            }, f, indent=2)
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0,0].plot(epochs, self.train_losses, label='训练损失')
        axes[0,0].plot(epochs, self.val_losses, label='验证损失')
        axes[0,0].set_title('损失曲线')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # AUPRC曲线
        auprc_values = [m['auprc'] for m in self.val_metrics]
        axes[0,1].plot(epochs, auprc_values, label='验证AUPRC', color='green')
        axes[0,1].set_title('AUPRC曲线')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('AUPRC')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # F1曲线
        f1_values = [m['f1'] for m in self.val_metrics]
        axes[1,0].plot(epochs, f1_values, label='验证F1', color='orange')
        axes[1,0].set_title('F1分数曲线')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('F1 Score')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 精确率和召回率
        precision_values = [m['precision'] for m in self.val_metrics]
        recall_values = [m['recall'] for m in self.val_metrics]
        axes[1,1].plot(epochs, precision_values, label='精确率', color='red')
        axes[1,1].plot(epochs, recall_values, label='召回率', color='blue')
        axes[1,1].set_title('精确率和召回率')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Score')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


# ================================
# 5. 主训练函数
# ================================

def main():
    """主训练函数"""
    print("🔥 WildfireSpreadTS CNN模型训练系统 v2.0")
    print("=" * 60)
    
    # 配置
    config = {
        'data_dir': 'data/processed',
        'sequence_length': 5,
        'prediction_horizon': 1,
        'batch_size': 2,  # 减小batch_size适应256x256图像
        'num_epochs': 50,
        'base_channels': 32,
        'num_workers': 0,  # 设置为0避免multiprocessing问题
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"使用设备: {config['device']}")
    
    # 检查数据目录
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建数据集
    print("\n1. 创建数据集...")
    train_dataset = WildfireDataset(
        data_dir=config['data_dir'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        mode='train',
        normalize=True
    )
    
    val_dataset = WildfireDataset(
        data_dir=config['data_dir'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        mode='val',
        normalize=True
    )
    
    # 复制训练集的标准化器到验证集
    if hasattr(train_dataset, 'scaler'):
        val_dataset.scaler = train_dataset.scaler
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    print("\n3. 创建模型...")
    model = WildfireCNN(
        input_channels=23,
        sequence_length=config['sequence_length'],
        base_channels=config['base_channels']
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = WildfireTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device']
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir='wildfire_cnn_v2_outputs'
    )
    
    print("\n✅ 训练完成!")
    print("📁 结果保存在: wildfire_cnn_v2_outputs/")
    print("📊 最佳模型: best_model.pth")
    print("📈 训练曲线: training_curves.png")


if __name__ == "__main__":
    main() 