#!/usr/bin/env python3
"""
Cellular Automata (CA) 野火传播模型
基于物理规则的元胞自动机模拟野火传播动力学
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import json
import time
import h5py
from sklearn.metrics import average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class FireState:
    """火灾状态枚举"""
    UNBURNED = 0      # 未燃烧
    BURNING = 1       # 正在燃烧
    BURNED = 2        # 已燃烧完毕
    WATER = 3         # 水体/不可燃
    
class CellularAutomataCore:
    """元胞自动机核心引擎"""
    
    def __init__(self, grid_size=(128, 128)):
        self.grid_size = grid_size
        self.H, self.W = grid_size
        
        # 邻域定义（8邻域 + 扩展邻域）
        self.neighbors_8 = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        
        # 风向影响权重（8个方向）
        self.wind_directions = {
            'N':  (-1, 0),  'NE': (-1, 1),  'E':  (0, 1),  'SE': (1, 1),
            'S':  (1, 0),   'SW': (1, -1),  'W':  (0, -1), 'NW': (-1, -1)
        }
        
        print(f"🔥 CA引擎初始化完成，网格大小: {grid_size}")
    
    def get_wind_direction_name(self, wind_angle):
        """将风向角度转换为方向名称"""
        # 标准化角度到 0-360
        wind_angle = wind_angle % 360
        
        # 将角度转换为8个基本方向
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((wind_angle + 22.5) // 45) % 8
        return directions[idx]
    
    def calculate_fire_probability(self, i, j, fire_state, fuel_load, moisture, 
                                 wind_speed, wind_direction, temperature, slope):
        """计算单个格子的着火概率"""
        
        # 基础概率
        base_prob = 0.01
        
        # 如果已经是水体，概率为0
        if fire_state[i, j] == FireState.WATER:
            return 0.0
        
        # 如果已经燃烧过，概率为0
        if fire_state[i, j] == FireState.BURNED:
            return 0.0
        
        # 燃料负荷影响 (0-1标准化)
        fuel_factor = np.clip(fuel_load[i, j] / 100.0, 0, 1)
        
        # 湿度影响 (湿度越高，着火概率越低)
        moisture_factor = np.clip(1 - moisture[i, j] / 100.0, 0.1, 1)
        
        # 温度影响 (温度越高，着火概率越高)
        temp_factor = np.clip((temperature[i, j] + 20) / 60.0, 0.1, 2)
        
        # 坡度影响 (坡度越大，火势传播越快)
        slope_factor = 1 + np.clip(slope[i, j] / 45.0, 0, 1) * 0.5
        
        # 邻域火源影响
        neighbor_fire_factor = self._calculate_neighbor_influence(
            i, j, fire_state, wind_speed, wind_direction
        )
        
        # 综合概率计算
        total_prob = (base_prob * 
                     fuel_factor * 
                     moisture_factor * 
                     temp_factor * 
                     slope_factor * 
                     neighbor_fire_factor)
        
        return np.clip(total_prob, 0, 1)
    
    def _calculate_neighbor_influence(self, i, j, fire_state, wind_speed, wind_direction):
        """计算邻域火源影响"""
        influence = 1.0
        
        # 获取风向
        wind_dir_name = self.get_wind_direction_name(wind_direction)
        wind_vector = self.wind_directions[wind_dir_name]
        wind_strength = np.clip(wind_speed / 20.0, 0, 3)  # 风速影响因子
        
        for di, dj in self.neighbors_8:
            ni, nj = i + di, j + dj
            
            # 边界检查
            if 0 <= ni < self.H and 0 <= nj < self.W:
                if fire_state[ni, nj] == FireState.BURNING:
                    # 基础邻域影响
                    base_influence = 2.0
                    
                    # 风向影响：顺风方向影响更大
                    wind_factor = 1.0
                    if wind_strength > 0.1:
                        # 计算从邻居到当前位置的方向向量
                        direction_vector = (-di, -dj)  # 注意方向
                        
                        # 计算与风向的夹角影响
                        dot_product = (direction_vector[0] * wind_vector[0] + 
                                     direction_vector[1] * wind_vector[1])
                        
                        # 顺风时影响增强，逆风时影响减弱
                        wind_factor = 1 + wind_strength * dot_product * 0.5
                        wind_factor = max(0.5, wind_factor)  # 最小保持50%影响
                    
                    # 距离影响（邻域距离）
                    distance = np.sqrt(di*di + dj*dj)
                    distance_factor = 1.0 / distance if distance > 0 else 1.0
                    
                    influence += base_influence * wind_factor * distance_factor
        
        return influence
    
    def step_evolution(self, fire_state, fuel_load, moisture, wind_speed, 
                      wind_direction, temperature, slope, ignition_prob=None):
        """执行一步演化"""
        new_fire_state = fire_state.copy()
        
        # 遍历所有格子
        for i in range(self.H):
            for j in range(self.W):
                current_state = fire_state[i, j]
                
                if current_state == FireState.UNBURNED:
                    # 计算着火概率
                    fire_prob = self.calculate_fire_probability(
                        i, j, fire_state, fuel_load, moisture,
                        wind_speed[i, j], wind_direction[i, j], 
                        temperature[i, j], slope[i, j]
                    )
                    
                    # 外部点火概率
                    if ignition_prob is not None:
                        fire_prob = max(fire_prob, ignition_prob[i, j])
                    
                    # 随机点火
                    if np.random.random() < fire_prob:
                        new_fire_state[i, j] = FireState.BURNING
                
                elif current_state == FireState.BURNING:
                    # 燃烧一段时间后变为已燃烧
                    # 简化模型：燃烧一步后就变为已燃烧
                    burnout_prob = 0.3 + 0.4 * (1 - fuel_load[i, j] / 100.0)
                    if np.random.random() < burnout_prob:
                        new_fire_state[i, j] = FireState.BURNED
        
        return new_fire_state

class LearnableCA(nn.Module):
    """可学习的元胞自动机"""
    
    def __init__(self, input_channels=22, hidden_dim=64, grid_size=(128, 128)):
        super(LearnableCA, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        
        # 环境特征编码器
        self.env_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # CA规则网络
        self.ca_rule = nn.Sequential(
            nn.Conv2d(hidden_dim + 4, hidden_dim, 3, padding=1),  # +4 for fire states
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 4, 1),  # 4个状态的概率
        )
        
        # 风向影响网络
        self.wind_influence = self._create_wind_networks()
        
        # 状态转移概率网络
        self.transition_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def _create_wind_networks(self):
        """创建8个方向的风向影响网络"""
        wind_nets = nn.ModuleDict()
        
        # 8个主要风向
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        for direction in directions:
            # 创建可学习的卷积层（而不是固定权重）
            conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            # 初始化为合理的风向权重
            with torch.no_grad():
                if direction in ['N', 'S']:
                    conv.weight.data = torch.tensor([[[
                        [0.1, 0.2, 0.1],
                        [0.2, 1.0, 0.2],
                        [0.1, 0.2, 0.1]
                    ]]], dtype=torch.float32)
                elif direction in ['E', 'W']:
                    conv.weight.data = torch.tensor([[[
                        [0.1, 0.2, 0.1],
                        [0.2, 1.0, 0.2],
                        [0.1, 0.2, 0.1]
                    ]]], dtype=torch.float32)
                else:  # 对角方向
                    conv.weight.data = torch.tensor([[[
                        [0.1, 0.1, 0.1],
                        [0.1, 1.0, 0.3],
                        [0.1, 0.1, 0.1]
                    ]]], dtype=torch.float32)
            
            wind_nets[direction] = conv
        
        return wind_nets
    
    def apply_wind_influence(self, fire_state, wind_speed, wind_direction):
        """应用风向影响"""
        batch_size = fire_state.size(0)
        influenced_fire = torch.zeros_like(fire_state)
        
        # 将风向转换为8个方向的权重
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        for b in range(batch_size):
            for direction in directions:
                # 计算该方向的权重
                dir_angle = directions.index(direction) * 45
                wind_angle = wind_direction[b, 0].mean().item()  # 简化：使用平均风向
                
                # 计算方向匹配度
                angle_diff = abs(wind_angle - dir_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)
                direction_weight = max(0, 1 - angle_diff / 90.0)  # 90度内有效
                
                # 风速影响
                wind_strength = wind_speed[b, 0].mean().item() / 20.0  # 标准化到0-1
                wind_strength = np.clip(wind_strength, 0, 1)
                
                # 应用风向卷积
                if direction_weight > 0.1 and wind_strength > 0.1:
                    wind_effect = self.wind_influence[direction](fire_state[b:b+1, 0:1])
                    influenced_fire[b:b+1, 0:1] += (wind_effect * 
                                                   direction_weight * 
                                                   wind_strength)
        
        return influenced_fire
    
    def forward(self, env_features, initial_fire_state, num_steps=5):
        """前向传播：执行CA演化"""
        batch_size = env_features.size(0)
        
        # 编码环境特征
        env_encoded = self.env_encoder(env_features)
        
        # 提取关键环境变量
        wind_speed = env_features[:, 8:9]      # 风速
        wind_direction = env_features[:, 7:8]  # 风向
        
        # 初始化火灾状态
        fire_state = initial_fire_state.clone()
        fire_evolution = [fire_state.clone()]
        
        # CA演化循环
        for step in range(num_steps):
            # 准备CA输入：环境特征 + 当前火灾状态
            fire_state_encoded = self._encode_fire_state(fire_state)
            ca_input = torch.cat([env_encoded, fire_state_encoded], dim=1)
            
            # 计算状态转移概率
            transition_logits = self.ca_rule(ca_input)
            transition_probs = F.softmax(transition_logits, dim=1)
            
            # 应用风向影响
            wind_influence = self.apply_wind_influence(fire_state, wind_speed, wind_direction)
            
            # 更新火灾状态
            fire_state = self._update_fire_state(
                fire_state, transition_probs, wind_influence
            )
            
            fire_evolution.append(fire_state.clone())
        
        return fire_state, fire_evolution
    
    def _encode_fire_state(self, fire_state):
        """将火灾状态编码为one-hot形式"""
        batch_size, _, H, W = fire_state.shape
        
        # 将连续值转换为离散状态
        fire_discrete = torch.zeros_like(fire_state).long()
        fire_discrete[fire_state < 0.25] = FireState.UNBURNED
        fire_discrete[(fire_state >= 0.25) & (fire_state < 0.75)] = FireState.BURNING
        fire_discrete[fire_state >= 0.75] = FireState.BURNED
        
        # One-hot编码
        fire_one_hot = F.one_hot(fire_discrete.squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()
        
        return fire_one_hot
    
    def _update_fire_state(self, current_state, transition_probs, wind_influence):
        """更新火灾状态 - 使用可微分的软更新"""
        batch_size, _, H, W = current_state.shape
        
        # 应用风向影响
        wind_boost = torch.sigmoid(wind_influence) * 0.3
        
        # 使用软更新而不是硬阈值，保持梯度连续性
        # 提取各状态的概率
        prob_unburned = transition_probs[:, 0:1]
        prob_burning = transition_probs[:, 1:2] + wind_boost
        prob_burned = transition_probs[:, 2:3]
        
        # 软状态转移 - 使用加权平均而不是硬分类
        # 当前状态权重
        current_weight = 0.7
        
        # 计算新状态值
        # 0.0 = 未燃烧, 0.5 = 燃烧, 1.0 = 已燃烧
        target_state = (prob_unburned * 0.0 + 
                       prob_burning * 0.5 + 
                       prob_burned * 1.0)
        
        # 软更新：保留部分当前状态，混合目标状态
        new_state = current_weight * current_state + (1 - current_weight) * target_state
        
        # 确保状态在合理范围内
        new_state = torch.clamp(new_state, 0.0, 1.0)
        
        return new_state

class CAWildfireDataset(Dataset):
    """CA模型专用数据集"""
    
    def __init__(self, data_dir, mode='train', max_files=100, target_size=(128, 128)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.target_size = target_size
        
        print(f"🔥 初始化CA数据集 ({mode}模式)")
        
        # 收集文件
        self._collect_files(max_files)
        
        # 构建样本
        self.samples = []
        self._build_samples()
        
        print(f"✅ CA数据集初始化完成: {len(self)} 个样本")
    
    def _collect_files(self, max_files):
        """收集HDF5文件"""
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
        """构建时序样本"""
        for file_idx, file_path in enumerate(self.files):
            try:
                with h5py.File(file_path, 'r') as f:
                    n_timesteps = f['data'].shape[0]
                    
                    # 创建时序序列（CA需要时序演化）
                    seq_length = 5  # CA演化步数
                    for t in range(0, n_timesteps - seq_length + 1, 2):
                        self.samples.append((file_idx, t, seq_length))
                        
            except Exception as e:
                print(f"跳过文件 {file_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, start_t, seq_length = self.samples[idx]
        file_path = self.files[file_idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 读取时序数据
                sequence_data = f['data'][start_t:start_t+seq_length]  # (T, C, H, W)
                
                # 转换为tensor
                sequence_tensor = torch.from_numpy(sequence_data).float()
                
                # Resize
                if sequence_tensor.shape[2:] != self.target_size:
                    T, C = sequence_tensor.shape[:2]
                    sequence_resized = F.interpolate(
                        sequence_tensor.view(T*C, 1, *sequence_tensor.shape[2:]),
                        size=self.target_size,
                        mode='bilinear',
                        align_corners=False
                    ).view(T, C, *self.target_size)
                else:
                    sequence_resized = sequence_tensor
                
                # 分离环境特征和火灾状态
                env_features = sequence_resized[0, :22]  # 初始环境特征
                initial_fire = sequence_resized[0, 22:23]  # 初始火灾状态
                target_fire = sequence_resized[-1, 22:23]  # 最终火灾状态
                
                # 数据清洗
                env_features = torch.where(torch.isfinite(env_features), 
                                         env_features, torch.tensor(0.0))
                initial_fire = torch.where(torch.isfinite(initial_fire), 
                                         initial_fire, torch.tensor(0.0))
                target_fire = torch.where(torch.isfinite(target_fire), 
                                        target_fire, torch.tensor(0.0))
                
                # 标准化火灾状态到[0,1]
                initial_fire = torch.sigmoid(initial_fire * 0.1)  # 减小sigmoid输入
                target_fire = torch.sigmoid(target_fire * 0.1)
                
                return env_features, initial_fire, target_fire
                
        except Exception as e:
            # 返回零张量
            return (torch.zeros(22, *self.target_size),
                   torch.zeros(1, *self.target_size),
                   torch.zeros(1, *self.target_size))

class CATrainer:
    """CA模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数：组合损失
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_auprcs = []
        self.best_val_loss = float('inf')
        
        print(f"🔥 CA训练器初始化完成")
        print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    def ca_loss(self, predicted_fire, target_fire, evolution_sequence=None):
        """CA专用损失函数"""
        # 基础重建损失
        recon_loss = self.mse_loss(predicted_fire, target_fire)
        
        # 二值化损失（鼓励清晰的状态）
        binary_target = (target_fire > 0.5).float()
        binary_loss = self.bce_loss(predicted_fire, binary_target)
        
        # 平滑性损失（相邻像素状态相似）
        smooth_loss = 0
        if predicted_fire.size(0) > 0:
            # 水平平滑
            h_smooth = torch.mean((predicted_fire[:, :, :, 1:] - predicted_fire[:, :, :, :-1])**2)
            # 垂直平滑
            v_smooth = torch.mean((predicted_fire[:, :, 1:, :] - predicted_fire[:, :, :-1, :])**2)
            smooth_loss = (h_smooth + v_smooth) * 0.1
        
        # 演化一致性损失
        evolution_loss = 0
        if evolution_sequence is not None and len(evolution_sequence) > 1:
            for i in range(1, len(evolution_sequence)):
                # 状态转移应该合理（燃烧区域不能突然消失）
                prev_state = evolution_sequence[i-1]
                curr_state = evolution_sequence[i]
                
                # 已燃烧区域应该保持已燃烧
                burned_mask = (prev_state > 0.75)
                if burned_mask.any():
                    evolution_loss += torch.mean((curr_state[burned_mask] - 1.0)**2) * 0.1
        
        total_loss = recon_loss + binary_loss + smooth_loss + evolution_loss
        
        return total_loss, {
            'recon': recon_loss.item(),
            'binary': binary_loss.item(),
            'smooth': smooth_loss if isinstance(smooth_loss, (int, float)) else smooth_loss.item(),
            'evolution': evolution_loss if isinstance(evolution_loss, (int, float)) else evolution_loss.item()
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        print_interval = max(1, len(self.train_loader) // 10)
        
        for batch_idx, (env_features, initial_fire, target_fire) in enumerate(self.train_loader):
            env_features = env_features.to(self.device)
            initial_fire = initial_fire.to(self.device)
            target_fire = target_fire.to(self.device)
            
            # 数据验证
            if (torch.isnan(env_features).any() or torch.isnan(initial_fire).any() or 
                torch.isnan(target_fire).any()):
                continue
            
            self.optimizer.zero_grad()
            
            # CA演化
            predicted_fire, evolution_sequence = self.model(
                env_features, initial_fire, num_steps=5
            )
            
            # 计算损失
            loss, loss_components = self.ca_loss(predicted_fire, target_fire, evolution_sequence)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # 打印进度
            if batch_idx % print_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f} "
                      f"[Recon: {loss_components['recon']:.4f}, "
                      f"Binary: {loss_components['binary']:.4f}]")
        
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
                
                predicted_fire, _ = self.model(env_features, initial_fire, num_steps=5)
                loss, _ = self.ca_loss(predicted_fire, target_fire)
                
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
    
    def train(self, num_epochs=25):
        """完整训练流程"""
        print(f"🚀 开始CA模型训练 {num_epochs} epochs")
        
        save_dir = Path('ca_wildfire_outputs')
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
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auprc': val_auprc
                }, save_dir / 'best_ca_model.pth')
                print("💾 保存最佳CA模型")
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        print(f"\n✅ CA训练完成!")
        print(f"📁 结果保存在: {save_dir}")
        
        return self.best_val_loss
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        axes[0].plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0].set_title('CA模型损失曲线')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUPRC曲线
        axes[1].plot(epochs, self.val_auprcs, 'g-', label='验证AUPRC', linewidth=2)
        axes[1].set_title('CA模型AUPRC曲线')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUPRC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 统计信息
        best_auprc = max(self.val_auprcs) if self.val_auprcs else 0
        stats_text = f"""CA模型训练统计:

最佳验证损失: {self.best_val_loss:.6f}
最佳AUPRC: {best_auprc:.4f}
总训练轮数: {len(self.train_losses)}

模型特点:
- 基于物理规则
- 考虑风向影响
- 状态转移学习
- 时序演化建模"""
        
        axes[2].text(0.1, 0.5, stats_text, transform=axes[2].transAxes, fontsize=11)
        axes[2].set_title('CA模型统计')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'ca_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🔥 Cellular Automata 野火模型")
    print("=" * 60)
    print("🧮 基于物理规则的元胞自动机野火传播模拟")
    print("=" * 60)
    
    # 配置
    config = {
        'data_dir': 'data/processed',
        'batch_size': 4,
        'num_epochs': 25,
        'max_files': 100,
        'target_size': (128, 128),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'ca_steps': 5
    }
    
    print(f"\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 检查数据目录
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保野火数据在 data/processed 目录下")
        return
    
    # 训练可学习CA模型
    print(f"\n🚀 开始训练可学习CA模型...")
    
    # 创建数据集
    print(f"\n1. 创建CA数据集...")
    train_dataset = CAWildfireDataset(
        config['data_dir'],
        mode='train',
        max_files=config['max_files'],
        target_size=config['target_size']
    )
    
    val_dataset = CAWildfireDataset(
        config['data_dir'],
        mode='val',
        max_files=config['max_files']//4,
        target_size=config['target_size']
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 验证集: {len(val_dataset)} 样本")
    
    if len(train_dataset) == 0:
        print("❌ 训练集为空，请检查数据目录和文件格式")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Windows兼容
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 创建可学习CA模型
    print(f"\n2. 创建可学习CA模型...")
    ca_model = LearnableCA(
        input_channels=22,
        hidden_dim=64,
        grid_size=config['target_size']
    )
    
    # 创建训练器
    print(f"\n3. 创建CA训练器...")
    ca_trainer = CATrainer(
        model=ca_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device']
    )
    
    # 开始训练
    print(f"\n4. 开始训练...")
    best_loss = ca_trainer.train(num_epochs=config['num_epochs'])
    
    print(f"\n✅ 可学习CA模型训练完成!")
    print(f"🏆 最佳验证损失: {best_loss:.6f}")
    print(f"📁 结果保存在: ca_wildfire_outputs/")
    print(f"💾 最佳模型: best_ca_model.pth")
    print(f"📊 训练曲线: ca_training_curves.png")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行主程序
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc() 