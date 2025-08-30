#!/usr/bin/env python3
"""
优化版CA野火模型
针对性能问题进行改进：
1. 简化架构减少过拟合
2. 改进损失函数
3. 更好的数据增强
4. 动态学习率
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

class OptimizedCAModel(nn.Module):
    """优化的CA模型 - 更简单但更有效"""
    
    def __init__(self, input_channels=22, hidden_dim=32):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # 简化的特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # CA核心网络 - 更简单的架构
        self.ca_core = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim//2, 3, padding=1),  # +1 for current fire state
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(hidden_dim//2, hidden_dim//4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim//4, 1, 1),  # 输出火势变化
            nn.Sigmoid()
        )
        
        # 风向影响网络 - 简化版
        self.wind_net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),  # 风速+风向
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, env_features, initial_fire_state, num_steps=3):
        """简化的前向传播"""
        batch_size = env_features.size(0)
        
        # 编码环境特征
        env_encoded = self.feature_encoder(env_features)
        
        # 提取风向信息
        wind_features = env_features[:, 7:9]  # 风向和风速
        wind_influence = self.wind_net(wind_features)
        
        # CA演化
        fire_state = initial_fire_state.clone()
        
        for step in range(num_steps):
            # 准备输入
            ca_input = torch.cat([env_encoded, fire_state], dim=1)
            
            # CA更新
            fire_change = self.ca_core(ca_input)
            
            # 应用风向影响
            fire_change = fire_change * (1 + wind_influence * 0.5)
            
            # 更新火势状态 - 简单的加法更新
            fire_state = torch.clamp(fire_state + fire_change * 0.3, 0, 1)
        
        return fire_state

class FocalLoss(nn.Module):
    """改进的Focal Loss"""
    
    def __init__(self, alpha=0.8, gamma=2.0):  # 增大alpha处理不平衡
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        # BCE损失
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # pt
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # alpha权重
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # focal权重
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()

class EnhancedDataset(Dataset):
    """增强版数据集 - 更好的数据增强"""
    
    def __init__(self, data_dir, mode='train', max_files=100, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        self.training = (mode == 'train')
        
        print(f"🔥 初始化增强版CA数据集 ({mode}模式)")
        
        # 收集文件
        self._collect_files(max_files)
        
        # 构建样本
        self.samples = []
        self._build_samples()
        
        # 计算数据统计
        if mode == 'train':
            self._compute_stats()
        
        print(f"✅ 数据集初始化完成: {len(self)} 个样本")
    
    def _collect_files(self, max_files):
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        files_to_use = all_files[:max_files]
        n_files = len(files_to_use)
        
        if self.mode == 'train':
            self.files = files_to_use[:int(0.8 * n_files)]
        else:
            self.files = files_to_use[int(0.8 * n_files):]
        
        print(f"📁 {self.mode}模式使用 {len(self.files)} 个文件")
    
    def _build_samples(self):
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_timesteps = f['data'].shape[0]
                    
                    # 增加采样密度
                    step = 1 if self.training else 2
                    for t in range(0, n_timesteps - 1, step):
                        self.samples.append((file_idx, t))
                        
            except Exception as e:
                continue
    
    def _compute_stats(self):
        """计算数据统计量"""
        print("计算数据统计...")
        
        sample_data = []
        for i in range(0, min(200, len(self.samples)), 10):
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
                self.data_mean = float(np.mean(valid_data))
                self.data_std = float(np.std(valid_data))
                print(f"数据统计: mean={self.data_mean:.3f}, std={self.data_std:.3f}")
            else:
                self.data_mean = 0.0
                self.data_std = 1.0
        else:
            self.data_mean = 0.0
            self.data_std = 1.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, timestep = self.samples[idx]
        file_path = self.files[file_idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                current_data = f['data'][timestep]
                next_data = f['data'][min(timestep + 1, f['data'].shape[0] - 1)]
                
                # 转换为tensor
                current_tensor = torch.from_numpy(current_data).float()
                next_tensor = torch.from_numpy(next_data).float()
                
                # Resize
                if current_tensor.shape[1:] != self.target_size:
                    current_tensor = F.interpolate(
                        current_tensor.unsqueeze(0), 
                        size=self.target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    next_tensor = F.interpolate(
                        next_tensor.unsqueeze(0), 
                        size=self.target_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                # 分离特征和目标
                env_features = current_tensor[:22]
                initial_fire = current_tensor[22:23]
                target_fire = next_tensor[22:23]
                
                # 数据清洗
                env_features = self._clean_features(env_features)
                initial_fire = self._clean_fire_state(initial_fire)
                target_fire = self._clean_fire_state(target_fire)
                
                # 数据增强
                if self.training and torch.rand(1) < 0.3:
                    env_features, initial_fire, target_fire = self._augment_data(
                        env_features, initial_fire, target_fire
                    )
                
                return env_features, initial_fire, target_fire
                
        except Exception as e:
            return (torch.zeros(22, *self.target_size),
                   torch.zeros(1, *self.target_size),
                   torch.zeros(1, *self.target_size))
    
    def _clean_features(self, features):
        """清洁特征数据"""
        # 处理NaN和Inf
        features = torch.where(torch.isfinite(features), features, torch.tensor(0.0))
        
        # 标准化
        if hasattr(self, 'data_mean'):
            features = (features - self.data_mean) / (self.data_std + 1e-8)
        
        # 裁剪极值
        features = torch.clamp(features, -5, 5)
        
        return features
    
    def _clean_fire_state(self, fire_state):
        """清洁火灾状态"""
        fire_state = torch.where(torch.isfinite(fire_state), fire_state, torch.tensor(0.0))
        fire_state = torch.clamp(fire_state, 0, 1)
        return fire_state
    
    def _augment_data(self, env_features, initial_fire, target_fire):
        """数据增强"""
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            env_features = torch.flip(env_features, [2])
            initial_fire = torch.flip(initial_fire, [2])
            target_fire = torch.flip(target_fire, [2])
        
        # 随机垂直翻转
        if torch.rand(1) < 0.5:
            env_features = torch.flip(env_features, [1])
            initial_fire = torch.flip(initial_fire, [1])
            target_fire = torch.flip(target_fire, [1])
        
        # 添加轻微噪声
        noise_scale = 0.01
        env_features += torch.randn_like(env_features) * noise_scale
        
        return env_features, initial_fire, target_fire

class OptimizedTrainer:
    """优化的训练器"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 改进的损失函数
        self.criterion = FocalLoss(alpha=0.8, gamma=2.0)
        
        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-3,  # 稍大的学习率
            weight_decay=1e-5  # 较小的权重衰减
        )
        
        # 动态学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.best_val_auprc = 0.0  # 改为监控AUPRC
        
        print(f"🚀 优化训练器初始化完成")
        print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (env_features, initial_fire, target_fire) in enumerate(self.train_loader):
            env_features = env_features.to(self.device)
            initial_fire = initial_fire.to(self.device)
            target_fire = target_fire.to(self.device)
            
            # 跳过无效数据
            if (torch.isnan(env_features).any() or torch.isnan(initial_fire).any() or 
                torch.isnan(target_fire).any()):
                continue
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predicted_fire = self.model(env_features, initial_fire, num_steps=3)
            
            # 计算损失
            loss = self.criterion(predicted_fire, target_fire)
            
            # 添加正则化项
            reg_loss = 0
            for param in self.model.parameters():
                reg_loss += torch.norm(param, 2)
            loss += 1e-6 * reg_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
        
        return total_loss / max(valid_batches, 1)
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        valid_batches = 0
        
        with torch.no_grad():
            for env_features, initial_fire, target_fire in self.val_loader:
                env_features = env_features.to(self.device)
                initial_fire = initial_fire.to(self.device)
                target_fire = target_fire.to(self.device)
                
                if (torch.isnan(env_features).any() or torch.isnan(initial_fire).any() or 
                    torch.isnan(target_fire).any()):
                    continue
                
                predicted_fire = self.model(env_features, initial_fire, num_steps=3)
                loss = self.criterion(predicted_fire, target_fire)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    all_preds.append(predicted_fire.cpu().numpy().flatten())
                    all_targets.append(target_fire.cpu().numpy().flatten())
        
        # 计算AUPRC
        if all_preds and all_targets:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            targets_binary = (targets > 0.5).astype(int)
            
            try:
                auprc = average_precision_score(targets_binary, preds)
            except:
                auprc = 0.0
        else:
            auprc = 0.0
        
        return total_loss / max(valid_batches, 1), auprc
    
    def train(self, num_epochs=30):
        """完整训练流程"""
        print(f"🚀 开始优化CA训练 {num_epochs} epochs")
        
        save_dir = Path('optimized_ca_outputs')
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
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
            
            # 保存最佳模型（基于AUPRC）
            if val_auprc > self.best_val_auprc:
                self.best_val_auprc = val_auprc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc
                }, save_dir / 'best_optimized_ca_model.pth')
                print("💾 保存最佳优化CA模型")
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        print(f"\n✅ 优化CA训练完成!")
        print(f"🏆 最佳验证AUPRC: {self.best_val_auprc:.4f}")
        
        return self.best_val_auprc
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0].plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0].set_title('优化CA损失曲线')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUPRC曲线
        axes[1].plot(epochs, self.val_auprcs, 'g-', label='验证AUPRC', linewidth=2)
        axes[1].set_title('优化CA AUPRC曲线')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUPRC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 改进统计
        improvement_text = f"""优化CA模型改进:

🎯 最佳AUPRC: {self.best_val_auprc:.4f}
📉 最低验证损失: {min(self.val_losses):.4f}
📈 AUPRC提升: {(max(self.val_auprcs) / (0.05 if max(self.val_auprcs) > 0.05 else 1) - 1) * 100:.1f}%

优化策略:
✅ 简化架构
✅ 改进损失函数  
✅ 数据增强
✅ 动态学习率
✅ 梯度裁剪"""
        
        axes[2].text(0.1, 0.5, improvement_text, transform=axes[2].transAxes, 
                    fontsize=10, verticalalignment='center')
        axes[2].set_title('优化效果')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'optimized_ca_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🚀 优化版CA野火模型")
    print("=" * 60)
    print("🎯 针对性能问题的优化版本")
    print("=" * 60)
    
    # 优化配置
    config = {
        'data_dir': 'data/processed',
        'batch_size': 8,  # 增大batch size
        'num_epochs': 30,
        'max_files': 150,  # 增加数据量
        'target_size': (128, 128),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n优化配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查数据
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建增强数据集
    print(f"\n1. 创建增强数据集...")
    train_dataset = EnhancedDataset(
        config['data_dir'],
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size']
    )
    
    val_dataset = EnhancedDataset(
        config['data_dir'],
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size']
    )
    
    # 复制统计量
    if hasattr(train_dataset, 'data_mean'):
        val_dataset.data_mean = train_dataset.data_mean
        val_dataset.data_std = train_dataset.data_std
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    if len(train_dataset) == 0:
        print("❌ 训练集为空")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 创建优化模型
    print(f"\n2. 创建优化CA模型...")
    model = OptimizedCAModel(
        input_channels=22,
        hidden_dim=32
    )
    
    # 创建优化训练器
    print(f"\n3. 创建优化训练器...")
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device']
    )
    
    # 开始训练
    print(f"\n4. 开始优化训练...")
    best_auprc = trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\n🎉 优化CA训练完成!")
    print(f"🏆 最佳AUPRC: {best_auprc:.4f}")
    print(f"📁 结果: optimized_ca_outputs/")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc() 