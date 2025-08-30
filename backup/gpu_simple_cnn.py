#!/usr/bin/env python3
"""
基于fixed_wildfire_cnn.py的简单GPU版本
保持原有稳定架构，只添加GPU支持
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

class GPUWildfireDataset(Dataset):
    """基于fixed版本的GPU数据集"""
    
    def __init__(self, data_dir, mode='train', max_files=607, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        
        print(f"初始化{mode}数据集 (GPU版本)...")
        
        # 收集文件 (与fixed版本相同)
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        files_to_use = all_files[:max_files]
        n_files = len(files_to_use)
        
        if mode == 'train':
            self.files = files_to_use[:int(0.8 * n_files)]
        else:
            self.files = files_to_use[int(0.8 * n_files):]
        
        print(f"{mode}模式: {len(self.files)} 文件")
        
        # 构建样本 (与fixed版本相同)
        self.samples = []
        self._build_samples()
        
        # 初始化预处理 (与fixed版本相同)
        if mode == 'train':
            self._initialize_preprocessing()
    
    def _build_samples(self):
        """构建样本索引 (与fixed版本相同)"""
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data']
                    n_timesteps = data.shape[0]
                    step = max(1, n_timesteps // 5)
                    for t in range(0, n_timesteps, step):
                        self.samples.append((file_idx, t))
            except Exception as e:
                continue
        
        print(f"构建了 {len(self.samples)} 个样本")
    
    def _initialize_preprocessing(self):
        """初始化预处理 (与fixed版本相同)"""
        print("初始化数据预处理器...")
        
        sample_features = []
        sample_size = min(100, len(self.samples))
        
        for i in range(0, sample_size, 10):
            try:
                file_idx, timestep = self.samples[i]
                file_path = self.files[file_idx]
                
                with h5py.File(file_path, 'r') as f:
                    data = f['data'][timestep]
                    continuous_data = np.concatenate([
                        data[:16],
                        data[17:22]
                    ], axis=0)
                    sample_features.append(continuous_data.flatten())
            except Exception as e:
                continue
        
        if sample_features:
            all_features = np.concatenate(sample_features)
            valid_mask = np.isfinite(all_features)
            if valid_mask.sum() > 0:
                valid_features = all_features[valid_mask]
                
                self.feature_mean = np.median(valid_features)
                self.feature_std = np.std(valid_features)
                self.feature_min = np.percentile(valid_features, 5)
                self.feature_max = np.percentile(valid_features, 95)
                
                print(f"特征统计: mean={self.feature_mean:.3f}, std={self.feature_std:.3f}")
            else:
                self._set_default_stats()
        else:
            self._set_default_stats()
    
    def _set_default_stats(self):
        """设置默认统计量"""
        self.feature_mean = 0.0
        self.feature_std = 1.0
        self.feature_min = -5.0
        self.feature_max = 5.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """与fixed版本相同的getitem逻辑"""
        file_idx, timestep = self.samples[idx]
        file_path = self.files[file_idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['data'][timestep]
                
                target = torch.from_numpy(data[22:23]).float()
                landcover = data[16:17]
                continuous_features = torch.cat([
                    torch.from_numpy(data[:16]).float(),
                    torch.from_numpy(data[17:22]).float()
                ], dim=0)
                
                # Resize
                continuous_features = F.interpolate(
                    continuous_features.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                landcover = F.interpolate(
                    torch.from_numpy(landcover).float().unsqueeze(0), 
                    size=self.target_size, 
                    mode='nearest'
                ).squeeze(0)
                
                target = F.interpolate(
                    target.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # 数据清洗 (与fixed版本相同)
                continuous_features = self._clean_and_normalize_features(continuous_features)
                landcover = self._clean_landcover(landcover)
                target = self._clean_target(target)
                
                features = torch.cat([continuous_features, landcover], dim=0)
                
                return features, target
                
        except Exception:
            return (torch.zeros(22, *self.target_size), 
                   torch.zeros(1, *self.target_size))
    
    def _clean_and_normalize_features(self, features):
        """与fixed版本相同"""
        features = torch.where(torch.isfinite(features), features, torch.tensor(self.feature_mean))
        features = torch.clamp(features, self.feature_min, self.feature_max)
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        return features
    
    def _clean_landcover(self, landcover):
        """与fixed版本相同"""
        landcover = torch.where(torch.isfinite(landcover), landcover, torch.tensor(1.0))
        landcover = torch.clamp(landcover, 1, 16)
        landcover = landcover - 1
        return landcover
    
    def _clean_target(self, target):
        """与fixed版本相同"""
        target = torch.where(torch.isfinite(target), target, torch.tensor(0.0))
        target = (target > 0).float()
        return target
    
class GPUUNet(nn.Module):
    def __init__(self, continuous_channels=21, landcover_classes=16, embed_dim=8):
        super().__init__()
        self.landcover_embedding = nn.Embedding(landcover_classes, embed_dim)
        total_channels = continuous_channels + embed_dim

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(total_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        continuous_features = x[:, :21]
        landcover = x[:, 21].long()
        landcover_embedded = self.landcover_embedding(landcover).permute(0, 3, 1, 2)
        x = torch.cat([continuous_features, landcover_embedded], dim=1)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with skip connections
        up1 = self.upconv1(bottleneck)
        up1 = torch.cat([up1, enc3], dim=1)
        dec1 = self.dec_conv1(up1)

        up2 = self.upconv2(dec1)
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec_conv2(up2)

        up3 = self.upconv3(dec2)
        up3 = torch.cat([up3, enc1], dim=1)
        dec3 = self.dec_conv3(up3)

        out = self.final_conv(dec3)
        return torch.sigmoid(out)


class GPUFixedCNN(nn.Module):
    """基于fixed版本的GPU CNN模型"""
    
    def __init__(self, continuous_channels=21, landcover_classes=16, embed_dim=8):
        super().__init__()
        
        # 土地覆盖嵌入 (与fixed版本相同)
        self.landcover_embedding = nn.Embedding(landcover_classes, embed_dim)
        
        total_channels = continuous_channels + embed_dim
        
        # 编码器 (与fixed版本相同)
        self.encoder = nn.Sequential(
            nn.Conv2d(total_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 解码器 (与fixed版本相同)
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
        
        # 权重初始化 (与fixed版本相同)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """与fixed版本相同的权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, x):
        """与fixed版本相同的前向传播"""
        batch_size = x.size(0)
        
        continuous_features = x[:, :21]
        landcover = x[:, 21].long()
        
        landcover_embedded = self.landcover_embedding(landcover)
        landcover_embedded = landcover_embedded.permute(0, 3, 1, 2)
        
        combined_features = torch.cat([continuous_features, landcover_embedded], dim=1)
        
        encoded = self.encoder(combined_features)
        decoded = self.decoder(encoded)
        
        return decoded


class GPUTrainer:
    """基于fixed版本的GPU训练器"""
    
    def __init__(self, model, train_loader, val_loader):
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("GPU不可用，使用CPU")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 使用与fixed版本相同的简单设置
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.best_val_loss = float('inf')
        
        print(f"训练器初始化完成，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self):
        """与fixed版本相同的训练epoch逻辑"""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0
        
        # GPU版本添加缓存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if torch.isnan(data).any() or torch.isnan(target).any():
                continue
            
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 50 == 0:
                gpu_mem = f", GPU: {torch.cuda.memory_allocated() / (1024**3):.1f}GB" if torch.cuda.is_available() else ""
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}{gpu_mem}")
            
            # 定期清理GPU内存
            if torch.cuda.is_available() and batch_idx % 25 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / max(valid_batches, 1)
    
    def validate(self):
        """与fixed版本相同的验证逻辑"""
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if torch.isnan(data).any() or torch.isnan(target).any():
                    continue
                
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
        
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
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss, auprc
    
    def train(self, num_epochs=20):
        """与fixed版本相同的训练流程"""
        print(f"开始训练 {num_epochs} epochs")
        
        save_dir = Path('gpu_simple_outputs')
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            train_loss = self.train_epoch()
            val_loss, val_auprc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_auprcs.append(val_auprc)
            
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证AUPRC: {val_auprc:.4f}")
            print(f"Epoch时间: {epoch_time:.1f}s")
            
            if torch.cuda.is_available():
                print(f"GPU内存: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc
                }, save_dir / 'best_model.pth')
                print("💾 保存最佳模型")
            
            # 定期清理
            if (epoch + 1) % 5 == 0 and torch.cuda.is_available():
                print("🧹 清理GPU内存...")
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"\n✅ 训练完成! 最佳验证损失: {self.best_val_loss:.6f}")


def main():
    """主函数"""
    print("🔥 基于fixed版本的简单GPU野火CNN")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        device_str = "GPU"
    else:
        print("⚠️  GPU不可用，使用CPU")
        device_str = "CPU"
    
    # 配置 (与fixed版本相同)
    config = {
        'data_dir': 'data/processed',
        'batch_size': 4,
        'num_epochs': 20,
        'max_files': 607,
        'target_size': (128, 128),
    }
    
    print(f"\n配置 ({device_str}版本):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查数据
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建数据集 (与fixed版本相同)
    print("\n1. 创建数据集...")
    train_dataset = GPUWildfireDataset(
        config['data_dir'], 
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size']
    )
    
    val_dataset = GPUWildfireDataset(
        config['data_dir'], 
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size']
    )
    
    # 复制统计量
    if hasattr(train_dataset, 'feature_mean'):
        for attr in ['feature_mean', 'feature_std', 'feature_min', 'feature_max']:
            setattr(val_dataset, attr, getattr(train_dataset, attr))
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Windows兼容
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # 创建模型 (与fixed版本相同)
    print("\n3. 创建模型...")
    model = GPUUNet(
        continuous_channels=21,
        landcover_classes=16,
        embed_dim=8
    )
    
    # 创建训练器
    print("\n4. 创建训练器...")
    trainer = GPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 开始训练
    print("\n5. 开始训练...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\n🎉 训练完成!")
    print("📁 结果保存在: gpu_simple_outputs/")


if __name__ == "__main__":
    main() 