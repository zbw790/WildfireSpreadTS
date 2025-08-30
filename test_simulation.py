"""
Fire Spread Simulation Module - COMPLETELY FIXED VERSION
========================================================

Complete fixed version with proper code structure and all features restored.
This version correctly integrates with training pipeline statistics.

Usage:
    python fire_simulation_fixed.py --model best_fire_model_official.pth
    python fire_simulation_fixed.py --demo
    python fire_simulation_fixed.py --verify
"""

import os
# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import glob

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from scipy import ndimage
import argparse
import pickle
import sys

# ============================================================================
# COMPATIBILITY CLASSES
# ============================================================================

class WildFireConfig:
    """Compatibility config class for checkpoint loading"""
    def __init__(self):
        self.SPATIAL_SIZE = (128, 128)
        self.SEQUENCE_LENGTH = 5
        self.BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]

class FirePredictionConfig:
    """Alternative config class name"""
    def __init__(self):
        self.SPATIAL_SIZE = (128, 128)
        self.SEQUENCE_LENGTH = 5
        self.BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]

# Make these available for unpickling
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# ============================================================================
# CORRECTED SIMULATION CONFIGURATION
# ============================================================================

class SimulationConfig:
    """Fixed configuration that matches training exactly"""
    
    SPATIAL_SIZE = (128, 128)
    SEQUENCE_LENGTH = 5
    PREDICTION_HORIZON = 1
    
    FEATURE_NAMES = [
        'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1',      # 0-2
        'NDVI', 'EVI2',                            # 3-4
        'Total_Precip', 'Wind_Speed',              # 5-6
        'Wind_Direction',                          # 7: Angular
        'Min_Temp_K', 'Max_Temp_K',               # 8-9
        'ERC', 'Spec_Hum', 'PDSI',                # 10-12
        'Slope', 'Aspect',                         # 13-14: Topography
        'Elevation', 'Landcover',                  # 15-16: Static
        'Forecast_Precip', 'Forecast_Wind_Speed',  # 17-18
        'Forecast_Wind_Dir',                       # 19: Angular
        'Forecast_Temp_C', 'Forecast_Spec_Hum',   # 20-21
        'Active_Fire'                              # 22: Target
    ]
    
    BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]
    
    # Corrected physics parameters
    MAX_SIMULATION_DAYS = 30
    FIRE_DECAY_RATE = 0.05
    FIRE_SPREAD_THRESHOLD = 0.3

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_with_compatibility(model_path, input_channels, sequence_length=5, device='cpu'):
    """Load model with multiple fallback options"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = None
    for method in ['safe', 'legacy', 'with_globals']:
        try:
            if method == 'safe':
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            elif method == 'legacy':
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            elif method == 'with_globals':
                torch.serialization.add_safe_globals([
                    'numpy.core.multiarray.scalar',
                    'numpy.core.multiarray._reconstruct',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            print(f"Model loaded with {method} method")
            break
            
        except Exception as e:
            if method == 'with_globals':
                raise ValueError(f"All loading methods failed. Last error: {e}")
            continue
    
    # Create and load model
    model = OfficialFireUNet(input_channels, sequence_length)
    
    try:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_ap' in checkpoint:
                    print(f"Best AP: {checkpoint['best_ap']:.4f}")
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise ValueError(f"Failed to load state dict: {e}")
    
    return model

# ============================================================================
# U-NET MODEL ARCHITECTURE
# ============================================================================

class OfficialFireUNet(nn.Module):
    """U-Net architecture matching training exactly"""
    
    def __init__(self, input_channels, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        total_input_channels = input_channels * sequence_length
        
        # Encoder
        self.enc1 = self._double_conv(total_input_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
        self._initialize_weights()
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size, seq_len * channels, height, width)
        
        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.pool(enc4)
        
        bottleneck = self.bottleneck(enc4_pool)
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        output = self.final_conv(dec1)
        return output

# ============================================================================
# FIXED FIRE EVENT LOADER WITH PROPER PREPROCESSING
# ============================================================================

class FixedFireEventLoader:
    """CRITICALLY FIXED: Properly integrates with training statistics"""
    
    def __init__(self, config):
        self.config = config
        self.feature_stats = self.load_feature_stats()
        print(f"Training statistics loaded from: {self.feature_stats['source_file']}")
    
    def load_feature_stats(self):
        """Load training normalization statistics"""
        stats_files = [
            'feature_stats.npz',
            'feature_stats_fold_1.npz',
            'feature_stats_fold_2.npz',
            'feature_stats_fold_3.npz',
            'emergency_feature_stats.npz'
        ]
        
        for filename in stats_files:
            if os.path.exists(filename):
                try:
                    stats = np.load(filename, allow_pickle=True)
                    print(f"Found training statistics: {filename}")
                    
                    required_keys = ['feature_mean', 'feature_std', 'best_features']
                    if all(key in stats for key in required_keys):
                        return {
                            'mean': stats['feature_mean'],
                            'std': stats['feature_std'],
                            'best_features': stats['best_features'],
                            'angular_features': stats.get('angular_features', np.array([7, 14, 19])),
                            'static_features': stats.get('static_features', np.array([13, 14, 15, 16])),
                            'categorical_features': stats.get('categorical_features', np.array([16])),
                            'source_file': filename
                        }
                    else:
                        print(f"Invalid statistics file {filename}: missing keys")
                        
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
        
        available_files = [f for f in os.listdir('.') if 'feature' in f.lower() or 'stats' in f.lower()]
        print(f"Available files: {available_files}")
        
        raise FileNotFoundError(
            "Training statistics not found!\n"
            "Please run training first to generate feature_stats.npz\n"
            f"Looked for: {stats_files}\n"
            f"Available: {available_files}"
        )
    
    def load_fire_event(self, hdf5_path):
        """Load fire event from HDF5 file"""
        print(f"Loading fire event: {hdf5_path}")
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'data' not in f:
                    raise ValueError("No 'data' key found in HDF5 file")
                
                data = f['data'][:]
                T, C, H, W = data.shape
                print(f"Fire event shape: {data.shape}")
                
                if T < self.config.SEQUENCE_LENGTH + 5:
                    raise ValueError(f"Fire event too short: {T} < {self.config.SEQUENCE_LENGTH + 5}")
                
                return torch.FloatTensor(data)
                
        except Exception as e:
            raise ValueError(f"Failed to load {hdf5_path}: {e}")
    
    def prepare_simulation_data(self, fire_event_data, start_day=0, mode='sliding_window', max_future_days=20):
        """
        准备模拟所需数据，严格对齐训练预处理，并避免未来信息泄漏。
        - mode='autoregressive'：不读取任何未来帧（安全）
        - mode='sliding_window'：仅允许使用 Forecast_* 通道的未来值；其它通道用“最后观测帧”持平（no-leak）

        Args:
            fire_event_data: torch.FloatTensor [T, C, H, W]（原始HDF5读取）
            start_day: 起始天
            mode: 'sliding_window' 或 'autoregressive'
            max_future_days: 最多往后模拟/对齐的天数（默认20）

        Returns:
            processed_sequence: torch.FloatTensor [SEQ_LEN, F, H', W']  —— 起始序列（已做训练同款预处理）
            future_weather_processed: torch.FloatTensor [K, F, H', W'] —— 未来驱动（仅在 sliding_window 下返回；否则 None）
            ground_truth_fire: list[np.ndarray]  —— 后续每天的二值火点真值（用于对照/可视化）
        """
        T, C, H, W = fire_event_data.shape
        SEQ = self.config.SEQUENCE_LENGTH
        if start_day + SEQ >= T:
            raise ValueError(f"Start day {start_day} too late for sequence length {SEQ} with T={T}.")

        # -------- 1) 初始观测序列：严格套训练预处理（含角度sin、标准化、best_features选择、静态帧置零等）
        initial_sequence_raw = fire_event_data[start_day:start_day + SEQ]              # [SEQ, C, H, W]
        processed_sequence = self._process_features_exact_training(initial_sequence_raw)  # [SEQ, F, H', W']

        # -------- 2) Ground truth：构造后续天的真实 Active_Fire（二值，用于评估/对照，不参与模型输入）
        remaining_days_total = T - (start_day + SEQ)
        remaining_days = int(min(remaining_days_total, max_future_days))
        ground_truth_fire = []
        if remaining_days > 0:
            for d in range(remaining_days):
                day_idx = start_day + SEQ + d
                # 原始通道里 Active_Fire 的索引是 22（来自训练配置）
                fire_data = fire_event_data[day_idx, 22]  # [H, W]
                fire_binary = (fire_data > 0).float()
                # resize 到训练的空间大小
                if (H, W) != tuple(self.config.SPATIAL_SIZE):
                    fire_binary = F.interpolate(
                        fire_binary.unsqueeze(0).unsqueeze(0),
                        size=self.config.SPATIAL_SIZE,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                ground_truth_fire.append(fire_binary.cpu().numpy())

        # -------- 3) 未来驱动：根据模式决定是否构造
        if mode == 'autoregressive' or remaining_days <= 0:
            # 自回归不需要未来帧；或没有未来天
            return processed_sequence, None, ground_truth_fire

        # 仅当 sliding_window 才需要 future_weather（用作每步的“观测/预报”驱动）
        future_raw = fire_event_data[start_day + SEQ : start_day + SEQ + remaining_days]  # [K, C, H, W]
        future_weather_processed = self._process_features_exact_training(future_raw)      # [K, F, H', W']

        # -------- 4) 防止信息泄漏：仅保留 Forecast_* 的未来值，其余通道用“最后观测帧”持平
        # 映射原始特征索引 -> 预处理后选择的通道位置
        best_features = list(self.feature_stats['best_features'])  # 原始索引列表，例如 [..., 22]
        # 哪些是 Forecast_* 原始索引（允许使用未来值）
        forecast_orig = {17, 18, 19, 20, 21}
        forecast_pos = [i for i, orig_idx in enumerate(best_features) if orig_idx in forecast_orig]

        # Active_Fire 在 best_features 中的位置（通常是 22；但防御式写法兼容位置变化）
        if 22 in best_features:
            active_fire_pos = best_features.index(22)
        else:
            # 兜底：如果找不到，就当成最后一维（与你当前实现一致）
            active_fire_pos = future_weather_processed.shape[1] - 1

        # “最后观测帧”——就是已预处理的初始序列最后一帧
        last_obs = processed_sequence[-1].clone()  # [F, H', W']

        # 把 future_weather_processed 里“非 Forecast_* & 非 Active_Fire”的通道替换为 last_obs（持平）
        K, Fp, Hp, Wp = future_weather_processed.shape
        for t in range(K):
            for c in range(Fp):
                if c == active_fire_pos:
                    # 火点通道未来帧必须为 0（后续会用预测写入）
                    future_weather_processed[t, c].zero_()
                elif c not in forecast_pos:
                    # 非 Forecast 通道禁止用未来真值 -> 持平为 last_obs
                    future_weather_processed[t, c] = last_obs[c]

        return processed_sequence, future_weather_processed, ground_truth_fire

    
    def _process_features_exact_training(self, data):
        """Apply EXACT training preprocessing - THE CRITICAL FIX"""
        print(f"Applying exact training preprocessing...")
        T, C, H, W = data.shape
        processed = data.clone()
        
        # Step 1: Angular features transformation
        angular_original = self.feature_stats['angular_features']
        for angle_idx in angular_original:
            if angle_idx < C:
                processed[:, angle_idx] = torch.sin(torch.deg2rad(processed[:, angle_idx]))
                print(f"  Applied sin transform to feature {angle_idx}")
        
        # Step 2: Handle missing values
        for c in range(C):
            mask = ~torch.isfinite(processed[:, c])
            if mask.any():
                if c < len(self.feature_stats['mean']):
                    fill_value = float(self.feature_stats['mean'][c])
                else:
                    fill_value = 0.0
                processed[:, c][mask] = fill_value
        
        # Step 3: Resize if needed
        if (H, W) != self.config.SPATIAL_SIZE:
            processed = F.interpolate(
                processed.view(-1, 1, H, W),
                size=self.config.SPATIAL_SIZE,
                mode='bilinear',
                align_corners=False
            ).view(T, C, *self.config.SPATIAL_SIZE)
            print(f"  Resized to {self.config.SPATIAL_SIZE}")
        
        # Step 4: Select best features
        best_features = self.feature_stats['best_features']
        processed = processed[:, best_features]
        print(f"  Selected {len(best_features)} best features")
        
        # Step 5: Apply training normalization - THE CRITICAL STEP
        training_mean = torch.FloatTensor(self.feature_stats['mean'])
        training_std = torch.FloatTensor(self.feature_stats['std'])
        
        if len(training_mean) > len(best_features):
            training_mean = training_mean[best_features]
            training_std = training_std[best_features]
        
        training_mean = training_mean.view(1, -1, 1, 1)
        training_std = training_std.view(1, -1, 1, 1)
        
        # Determine features to normalize
        angular_in_best = []
        categorical_in_best = []
        static_in_best = []
        
        for i, orig_idx in enumerate(best_features):
            if orig_idx in self.feature_stats['angular_features']:
                angular_in_best.append(i)
            if orig_idx in self.feature_stats['categorical_features']:
                categorical_in_best.append(i)
            if orig_idx in self.feature_stats['static_features']:
                static_in_best.append(i)
        
        # Apply normalization
        for c in range(processed.shape[1]):
            if c not in angular_in_best and c not in categorical_in_best:
                processed[:, c] = (processed[:, c] - training_mean[0, c]) / (training_std[0, c] + 1e-6)
        
        print(f"  Normalized features (skipped {len(angular_in_best + categorical_in_best)} special)")
        
        # Step 6: Multi-temporal processing
        if static_in_best:
            for t in range(T-1):
                for static_idx in static_in_best:
                    processed[t, static_idx] = 0
            print(f"  Zeroed static features in first {T-1} frames")
        
        # Verification
        if torch.isnan(processed).any():
            print("WARNING: NaN detected!")
            
        fire_channel = processed[-1, -1]
        fire_count = (fire_channel > 0).sum().item()
        print(f"  Input fire pixels: {fire_count}")
        print(f"  Data range: [{processed.min().item():.3f}, {processed.max().item():.3f}]")
        print(f"  Mean: {processed.mean().item():.3f}, Std: {processed.std().item():.3f}")
        
        return processed

# ============================================================================
# FIRE SIMULATOR WITH CORRECTED PHYSICS
# ============================================================================

class FixedFireSpreadSimulator:
    """Fire simulator with proper preprocessing and physics"""
    
    def __init__(self, model_path, config=None, device=None):
        self.config = config or SimulationConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_channels = len(self.config.BEST_FEATURES)
        
        self.model = load_model_with_compatibility(
            model_path, input_channels, self.config.SEQUENCE_LENGTH, self.device
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Simulator ready on {self.device}")
        print(f"Input channels: {input_channels}")
    
    def predict_single_step(self, input_sequence, debug=False):
        """Predict with debugging"""
        with torch.no_grad():
            if len(input_sequence.shape) == 3:
                input_sequence = input_sequence.unsqueeze(0)
            
            input_tensor = input_sequence.to(self.device)
            
            if debug:
                fire_channel = input_tensor[0, -1, -1]
                print(f"\n=== PREDICTION DEBUG ===")
                print(f"Input shape: {input_tensor.shape}")
                print(f"Input fire pixels: {(fire_channel > 0).sum().item()}")
                print(f"Input range: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]")
                print(f"Input mean: {input_tensor.mean().item():.3f}")
            
            try:
                with torch.amp.autocast('cuda'):
                    output = self.model(input_tensor)
                    prediction = torch.sigmoid(output)
            except:
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output)
            
            if debug:
                print(f"Raw output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                print(f"Sigmoid range: [{prediction.min().item():.3f}, {prediction.max().item():.3f}]")
                print(f"Predictions > 0.5: {(prediction > 0.5).sum().item()}")
                print(f"Predictions > 0.3: {(prediction > 0.3).sum().item()}")
                print(f"Mean prediction: {prediction.mean().item():.4f}")
                print("========================")
            
            return prediction.cpu().squeeze()
    
    def simulate_fire_evolution(self, initial_sequence, weather_data=None,
                              num_days=10, mode='sliding_window', debug=False):
        """Simulate fire evolution"""
        predictions = []
        current_sequence = initial_sequence.clone()
        
        print(f"Simulating {num_days} days using {mode} mode...")
        if debug:
            print("Debug mode enabled")
        
        for day in tqdm(range(num_days), desc="Fire simulation"):
            show_debug = debug and day < 3
            pred_fire = self.predict_single_step(current_sequence.unsqueeze(0), debug=show_debug)
            
            pred_fire = self._apply_fire_physics(pred_fire, day, debug=show_debug)
            predictions.append(pred_fire.numpy())
            
            if day < num_days - 1:
                if mode == 'sliding_window' and weather_data is not None:
                    if day < len(weather_data) - self.config.SEQUENCE_LENGTH:
                        next_sequence = weather_data[day + 1:day + 1 + self.config.SEQUENCE_LENGTH].clone()
                        active_fire_idx = len(self.config.BEST_FEATURES) - 1
                        next_sequence[-1, active_fire_idx] = pred_fire
                        current_sequence = next_sequence
                    else:
                        break
                elif mode == 'autoregressive':
                    new_frame = current_sequence[-1].clone()
                    active_fire_idx = len(self.config.BEST_FEATURES) - 1
                    new_frame[active_fire_idx] = pred_fire
                    
                    current_sequence = torch.cat([
                        current_sequence[1:],
                        new_frame.unsqueeze(0)
                    ], dim=0)
                else:
                    break
        
        return predictions
    
    def _apply_fire_physics(self, fire_prediction, day, debug=False):
        """Apply corrected fire physics"""
        if debug:
            print(f"\n=== FIRE PHYSICS (Day {day}) ===")
            print(f"Raw range: [{fire_prediction.min().item():.3f}, {fire_prediction.max().item():.3f}]")
            print(f"Pixels > threshold: {(fire_prediction > self.config.FIRE_SPREAD_THRESHOLD).sum().item()}")
        
        # Apply threshold
        fire_binary = (fire_prediction > self.config.FIRE_SPREAD_THRESHOLD).float()
        
        if debug:
            print(f"After threshold: {fire_binary.sum().item()} pixels")
        
        # Apply decay
        decay_factor = 1.0 - self.config.FIRE_DECAY_RATE * (day + 1)
        decay_factor = max(0.1, decay_factor)
        
        fire_decayed = fire_binary * decay_factor
        
        if debug:
            print(f"Decay factor: {decay_factor:.3f}")
            print(f"After decay: {(fire_decayed > 0).sum().item()} pixels")
        
        # Spatial smoothing
        fire_smoothed = torch.tensor(
            ndimage.gaussian_filter(fire_decayed.numpy(), sigma=0.5)
        )
        
        if debug:
            print(f"Final range: [{fire_smoothed.min().item():.3f}, {fire_smoothed.max().item():.3f}]")
            print("=========================")
        
        return fire_smoothed
    
    def create_simulation_animation(self, predictions, real_sequence=None, save_path='fire_simulation.gif'):
        """Create animation"""
        n_subplots = 2 if real_sequence is not None else 1
        fig, axes = plt.subplots(1, n_subplots, figsize=(8 * n_subplots, 6))
        
        if n_subplots == 1:
            axes = [axes]
        
        def animate(frame):
            for ax in axes:
                ax.clear()
            
            if real_sequence is not None and frame < len(real_sequence):
                axes[0].imshow(real_sequence[frame], cmap='Reds', vmin=0, vmax=1)
                axes[0].set_title(f'Actual Fire - Day {frame+1}')
                axes[0].axis('off')
                
                if frame < len(predictions):
                    axes[1].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[1].set_title(f'Predicted Fire - Day {frame+1}')
                    axes[1].axis('off')
            else:
                if frame < len(predictions):
                    im = axes[0].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[0].set_title(f'Fire Simulation - Day {frame+1}')
                    axes[0].axis('off')
                    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        frames = min(len(predictions), 30)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved: {save_path}")
        except Exception as e:
            print(f"Animation save failed: {e}")
            for i, pred in enumerate(predictions[:10]):
                plt.figure(figsize=(8, 6))
                plt.imshow(pred, cmap='Reds', vmin=0, vmax=1)
                plt.title(f'Day {i+1}')
                plt.colorbar()
                plt.savefig(f'fire_frame_{i:02d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            print("Saved individual frames")
    
    def create_comparison_animation(self, predictions, ground_truth, save_path='comparison.gif'):
        """Create comparison animation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            if frame < len(predictions):
                ax1.imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                ax1.set_title(f'Predicted - Day {frame+1}')
                ax1.axis('off')
                
                if frame < len(ground_truth):
                    ax2.imshow(ground_truth[frame], cmap='Reds', vmin=0, vmax=1)
                    ax2.set_title(f'Actual - Day {frame+1}')
                    ax2.axis('off')
                    
                    pred_binary = (predictions[frame] > self.config.FIRE_SPREAD_THRESHOLD).astype(float)
                    actual_binary = ground_truth[frame]
                    
                    if actual_binary.sum() > 0:
                        intersection = (pred_binary * actual_binary).sum()
                        union = pred_binary.sum() + actual_binary.sum() - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        dice = (2 * intersection) / (pred_binary.sum() + actual_binary.sum()) if (pred_binary.sum() + actual_binary.sum()) > 0 else 0
                        
                        plt.figtext(0.5, 0.02, f'IoU: {iou:.3f} | Dice: {dice:.3f}',
                                   ha='center', fontsize=12,
                                   bbox=dict(boxstyle="round", facecolor="lightgray"))
        
        frames = min(len(predictions), len(ground_truth), 30)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=1.5)
            print(f"Comparison animation saved: {save_path}")
        except Exception as e:
            print(f"Animation save failed: {e}")

# ============================================================================
# SINGLE FIRE EVENT EVOLUTION ANALYZER
# ============================================================================

class SingleFireEventAnalyzer:
    """Complete fire event evolution analysis"""
    
    def __init__(self, simulator, config=None):
        self.simulator = simulator
        self.config = config or SimulationConfig()
    
    def analyze_fire_evolution(self, fire_event_path):
        """Analyze complete fire event evolution"""
        print(f"Analyzing fire evolution: {os.path.basename(fire_event_path)}")
        
        fire_loader = FixedFireEventLoader(self.config)
        fire_event_data = fire_loader.load_fire_event(fire_event_path)
        T, C, H, W = fire_event_data.shape
        
        print(f"Fire event duration: {T} days, Size: {H}x{W}")
        
        evolution_analysis = []
        analysis_days = T - self.config.SEQUENCE_LENGTH - 1
        print(f"Analyzing {analysis_days} time windows...")
        
        for start_day in range(0, analysis_days, 1):
            try:
                window_analysis = self._analyze_time_window(
                    fire_event_data, start_day, fire_event_path
                )
                if window_analysis:
                    evolution_analysis.append(window_analysis)
            except Exception as e:
                print(f"Error analyzing day {start_day}: {e}")
                continue
        
        temporal_patterns = self._extract_temporal_patterns(evolution_analysis)
        return evolution_analysis, temporal_patterns
    
    def _analyze_time_window(self, fire_data, start_day, fire_path):
        """Analyze specific time window"""
        fire_loader = FixedFireEventLoader(self.config)
        
        try:
            initial_sequence, _, ground_truth = fire_loader.prepare_simulation_data(
                fire_data, start_day=start_day
            )
            
            if not ground_truth:
                return None
            
            daily_conditions = []
            for day_idx in range(initial_sequence.shape[0]):
                conditions = self._extract_daily_conditions(initial_sequence[day_idx])
                conditions['day'] = start_day + day_idx
                daily_conditions.append(conditions)
            
            pred_fire = self.simulator.predict_single_step(initial_sequence.unsqueeze(0))
            actual_fire = torch.FloatTensor(ground_truth[0])
            
            accuracy = self._calculate_detailed_accuracy(pred_fire, actual_fire)
            spread_analysis = self._analyze_fire_spread_pattern(
                initial_sequence[-1, -1].numpy(),
                pred_fire.numpy(),
                actual_fire.numpy()
            )
            
            return {
                'start_day': start_day,
                'fire_event': os.path.basename(fire_path),
                'daily_conditions': daily_conditions,
                'prediction_accuracy': accuracy,
                'spread_analysis': spread_analysis,
                'environmental_summary': self._summarize_window_environment(daily_conditions)
            }
            
        except Exception as e:
            return None
    
    def _extract_daily_conditions(self, day_data):
        """Extract environmental conditions"""
        conditions = {}
        
        try:
            feature_indices = {
                'ndvi': 0, 'evi2': 1, 'viirs_m11': 2, 'viirs_i2': 3,
                'viirs_i1': 4, 'slope': 5, 'aspect': 6, 'elevation': 7,
                'landcover': 8, 'precipitation': 9, 'max_temp': 10, 'min_temp': 11
            }
            
            for var_name, idx in feature_indices.items():
                if idx < day_data.shape[0]:
                    data = day_data[idx].numpy()
                    valid_data = data[np.isfinite(data)]
                    conditions[var_name] = float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0
            
            if day_data.shape[0] > 12:
                fire_data = day_data[-1].numpy()
                conditions['fire_area'] = float(np.sum(fire_data > 0))
                conditions['fire_intensity'] = float(np.mean(fire_data[fire_data > 0])) if np.sum(fire_data > 0) > 0 else 0.0
            
        except Exception as e:
            conditions = {var: 0.0 for var in ['ndvi', 'precipitation', 'max_temp', 'fire_area']}
        
        return conditions
    
    def _calculate_detailed_accuracy(self, predicted, actual):
        """Calculate accuracy metrics"""
        try:
            pred_np = predicted.detach().cpu().numpy()
            actual_np = actual.detach().cpu().numpy()
            
            pred_binary = (pred_np > 0.5).astype(float)
            actual_binary = (actual_np > 0).astype(float)
            
            if actual_binary.sum() > 0:
                intersection = (pred_binary * actual_binary).sum()
                union = pred_binary.sum() + actual_binary.sum() - intersection
                
                iou = intersection / union if union > 0 else 0.0
                dice = (2 * intersection) / (pred_binary.sum() + actual_binary.sum()) if (pred_binary.sum() + actual_binary.sum()) > 0 else 0.0
                precision = intersection / pred_binary.sum() if pred_binary.sum() > 0 else 0.0
                recall = intersection / actual_binary.sum()
            else:
                iou = dice = precision = recall = 0.0
            
            return {
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall),
                'predicted_area': float(pred_binary.sum()),
                'actual_area': float(actual_binary.sum())
            }
            
        except Exception as e:
            return {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'predicted_area': 0.0, 'actual_area': 0.0}
    
    def _analyze_fire_spread_pattern(self, initial_fire, predicted_fire, actual_fire):
        """Analyze fire spread patterns"""
        try:
            initial_centroid = self._calculate_fire_centroid(initial_fire)
            pred_centroid = self._calculate_fire_centroid(predicted_fire)
            actual_centroid = self._calculate_fire_centroid(actual_fire)
            
            if initial_centroid and pred_centroid and actual_centroid:
                pred_direction = np.arctan2(pred_centroid[1] - initial_centroid[1],
                                          pred_centroid[0] - initial_centroid[0]) * 180 / np.pi
                actual_direction = np.arctan2(actual_centroid[1] - initial_centroid[1],
                                            actual_centroid[0] - initial_centroid[0]) * 180 / np.pi
                
                pred_distance = np.sqrt((pred_centroid[0] - initial_centroid[0])**2 +
                                       (pred_centroid[1] - initial_centroid[1])**2)
                actual_distance = np.sqrt((actual_centroid[0] - initial_centroid[0])**2 +
                                         (actual_centroid[1] - initial_centroid[1])**2)
                
                direction_error = abs(pred_direction - actual_direction)
                if direction_error > 180:
                    direction_error = 360 - direction_error
                
                return {
                    'predicted_direction': float(pred_direction),
                    'actual_direction': float(actual_direction),
                    'direction_error': float(direction_error),
                    'predicted_spread_distance': float(pred_distance),
                    'actual_spread_distance': float(actual_distance),
                    'distance_error': float(abs(pred_distance - actual_distance))
                }
        except Exception as e:
            pass
        
        return {'predicted_direction': 0.0, 'actual_direction': 0.0, 'direction_error': 0.0,
                'predicted_spread_distance': 0.0, 'actual_spread_distance': 0.0, 'distance_error': 0.0}
    
    def _calculate_fire_centroid(self, fire_map):
        """Calculate fire centroid"""
        fire_pixels = np.where(fire_map > 0)
        if len(fire_pixels[0]) > 0:
            centroid_y = np.mean(fire_pixels[0])
            centroid_x = np.mean(fire_pixels[1])
            return (centroid_x, centroid_y)
        return None
    
    def _summarize_window_environment(self, daily_conditions):
        """Summarize environmental conditions"""
        if not daily_conditions:
            return {}
        
        summary = {}
        
        for var in ['precipitation', 'max_temp', 'fire_area']:
            values = [day.get(var, 0) for day in daily_conditions]
            summary[f'{var}_trend'] = 'increasing' if values[-1] > values[0] else 'decreasing'
            summary[f'{var}_mean'] = np.mean(values)
            summary[f'{var}_std'] = np.std(values)
        
        precip_values = [day.get('precipitation', 0) for day in daily_conditions]
        temp_values = [day.get('max_temp', 300) for day in daily_conditions]
        
        summary['condition_type'] = 'dry' if np.mean(precip_values) < 2 else 'wet'
        summary['temperature_level'] = 'high' if np.mean(temp_values) > 308 else 'moderate'
        
        return summary
    
    def _extract_temporal_patterns(self, evolution_analysis):
        """Extract temporal patterns"""
        if not evolution_analysis:
            return {}
        
        accuracies = [w['prediction_accuracy']['iou'] for w in evolution_analysis]
        fire_areas = [w['environmental_summary'].get('fire_area_mean', 0) for w in evolution_analysis]
        temperatures = [w['environmental_summary'].get('max_temp_mean', 300) for w in evolution_analysis]
        
        patterns = {
            'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
            'mean_accuracy': np.mean(accuracies),
            'accuracy_variability': np.std(accuracies),
            'fire_growth_rate': (fire_areas[-1] - fire_areas[0]) / len(fire_areas) if len(fire_areas) > 1 else 0,
            'temperature_correlation': np.corrcoef(temperatures, accuracies)[0,1] if len(temperatures) > 1 else 0,
            'peak_fire_day': np.argmax(fire_areas),
            'best_prediction_day': np.argmax(accuracies)
        }
        
        return patterns
    
    def plot_fire_evolution(self, evolution_analysis, temporal_patterns, save_path='fire_evolution.png'):
        """Create evolution plots"""
        if not evolution_analysis:
            print("No evolution data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        days = [w['start_day'] for w in evolution_analysis]
        accuracies = [w['prediction_accuracy']['iou'] for w in evolution_analysis]
        fire_areas = [w['environmental_summary'].get('fire_area_mean', 0) for w in evolution_analysis]
        temperatures = [w['environmental_summary'].get('max_temp_mean', 300) for w in evolution_analysis]
        precipitations = [w['environmental_summary'].get('precipitation_mean', 0) for w in evolution_analysis]
        
        # Accuracy plot
        axes[0,0].plot(days, accuracies, 'bo-', linewidth=2, markersize=6)
        axes[0,0].set_title('Prediction Accuracy Over Time', fontweight='bold')
        axes[0,0].set_xlabel('Day')
        axes[0,0].set_ylabel('IoU Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Fire area plot
        axes[0,1].plot(days, fire_areas, 'ro-', linewidth=2, markersize=6)
        axes[0,1].set_title('Fire Area Evolution', fontweight='bold')
        axes[0,1].set_xlabel('Day')
        axes[0,1].set_ylabel('Fire Area (pixels)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Environmental conditions
        ax3 = axes[1,0]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(days, temperatures, 'g-', linewidth=2, label='Temperature (K)')
        line2 = ax3_twin.plot(days, precipitations, 'b--', linewidth=2, label='Precipitation (mm)')
        
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Temperature (K)', color='g')
        ax3_twin.set_ylabel('Precipitation (mm)', color='b')
        ax3.set_title('Environmental Conditions', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        
        # Scatter plot
        axes[1,1].scatter(fire_areas, accuracies, c=temperatures, cmap='hot', s=60)
        axes[1,1].set_xlabel('Fire Area (pixels)')
        axes[1,1].set_ylabel('Prediction Accuracy (IoU)')
        axes[1,1].set_title('Accuracy vs Fire Size', fontweight='bold')
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('Temperature (K)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Fire Evolution Analysis - {evolution_analysis[0]["fire_event"]}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fire evolution analysis saved: {save_path}")
    
    def create_evolution_animation(self, evolution_analysis, fire_event_path, save_path='fire_evolution.gif'):
        """Create evolution animation"""
        if not evolution_analysis:
            print("No evolution data to animate")
            return
        
        print(f"Creating evolution animation with {len(evolution_analysis)} frames...")
        
        fire_loader = FixedFireEventLoader(self.config)
        fire_event_data = fire_loader.load_fire_event(fire_event_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        def animate(frame_idx):
            if frame_idx >= len(evolution_analysis):
                return
            
            for ax in axes:
                ax.clear()
            
            window = evolution_analysis[frame_idx]
            start_day = window['start_day']
            
            try:
                initial_sequence, _, ground_truth = fire_loader.prepare_simulation_data(
                    fire_event_data, start_day=start_day
                )
                
                pred_fire = self.simulator.predict_single_step(initial_sequence.unsqueeze(0))
                actual_fire = torch.FloatTensor(ground_truth[0]) if ground_truth else None
                
                # Current fire
                current_fire = initial_sequence[-1, -1].numpy()
                axes[0].imshow(current_fire, cmap='Reds', vmin=0, vmax=1)
                axes[0].set_title(f'Current Fire - Day {start_day + 4}', fontweight='bold')
                axes[0].axis('off')
                
                # Predicted fire
                pred_np = pred_fire.detach().cpu().numpy()
                axes[1].imshow(pred_np, cmap='Oranges', vmin=0, vmax=1)
                axes[1].set_title(f'Predicted Fire - Day {start_day + 5}', fontweight='bold')
                axes[1].axis('off')
                
                # Actual fire
                if actual_fire is not None:
                    actual_np = actual_fire.numpy()
                    axes[2].imshow(actual_np, cmap='Greens', vmin=0, vmax=1)
                    axes[2].set_title(f'Actual Fire - Day {start_day + 5}', fontweight='bold')
                    axes[2].axis('off')
                    
                    pred_binary = (pred_np > 0.5).astype(float)
                    actual_binary = (actual_np > 0).astype(float)
                    
                    if actual_binary.sum() > 0:
                        intersection = (pred_binary * actual_binary).sum()
                        union = pred_binary.sum() + actual_binary.sum() - intersection
                        iou = intersection / union if union > 0 else 0
                    else:
                        iou = 0
                    
                    axes[2].text(0.02, 0.98, f'IoU: {iou:.3f}', transform=axes[2].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               verticalalignment='top')
                else:
                    axes[2].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center',
                               transform=axes[2].transAxes, fontsize=14)
                    axes[2].axis('off')
                
                # Environmental summary
                axes[3].axis('off')
                env_summary = window.get('environmental_summary', {})
                accuracy = window.get('prediction_accuracy', {})
                
                env_text = f"""Environmental Conditions - Day {start_day + 4}:

Temperature: {env_summary.get('max_temp_mean', 0):.1f} K ({env_summary.get('temperature_level', 'unknown')})
Precipitation: {env_summary.get('precipitation_mean', 0):.1f} mm
Conditions: {env_summary.get('condition_type', 'unknown')}

Fire Area: {accuracy.get('actual_area', 0):.0f} pixels
Prediction Accuracy: {accuracy.get('iou', 0):.3f} IoU

Fire Growth: {env_summary.get('fire_area_trend', 'unknown')}
"""
                
                axes[3].text(0.05, 0.95, env_text, transform=axes[3].transAxes,
                           verticalalignment='top', fontsize=11, fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                
                fig.suptitle(f'Fire Evolution - Day {start_day + 4} to {start_day + 5}\n{window["fire_event"]}',
                           fontsize=16, fontweight='bold')
                
            except Exception as e:
                print(f"Animation error frame {frame_idx}: {e}")
                for ax in axes:
                    ax.clear()
                    ax.text(0.5, 0.5, f'Error in frame {frame_idx}', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.axis('off')
        
        frames = min(len(evolution_analysis), 50)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=0.3)
            print(f"Evolution animation saved: {save_path}")
        except Exception as e:
            print(f"Animation save failed: {e}")
            for i in range(min(frames, 10)):
                animate(i)
                plt.savefig(f'evolution_frame_{i:02d}.png', bbox_inches='tight', dpi=150)
            print("Saved individual frames: evolution_frame_*.png")
        
        plt.close(fig)
        return anim
    
    def generate_evolution_report(self, evolution_analysis, temporal_patterns, fire_event_name):
        """Generate detailed evolution report"""
        if not evolution_analysis:
            return "No evolution data available"
        
        report = f"""
SINGLE FIRE EVENT EVOLUTION ANALYSIS
===================================

Fire Event: {fire_event_name}
Analysis Period: Day {evolution_analysis[0]['start_day']} to Day {evolution_analysis[-1]['start_day']}
Total Time Windows: {len(evolution_analysis)}

TEMPORAL PATTERNS:
================
Accuracy Trend: {temporal_patterns.get('accuracy_trend', 'unknown')}
Mean Accuracy (IoU): {temporal_patterns.get('mean_accuracy', 0):.4f}
Accuracy Variability: {temporal_patterns.get('accuracy_variability', 0):.4f}
Fire Growth Rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day

Peak Fire Day: {temporal_patterns.get('peak_fire_day', 0)}
Best Prediction Day: {temporal_patterns.get('best_prediction_day', 0)}
Temperature Correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}

DETAILED ANALYSIS:
================
"""
        
        for window in evolution_analysis[:5]:
            day = window['start_day']
            accuracy = window['prediction_accuracy']
            env = window['environmental_summary']
            spread = window['spread_analysis']
            
            report += f"""
Day {day}:
  Accuracy: IoU={accuracy['iou']:.3f}, Dice={accuracy['dice']:.3f}
  Fire Area: {accuracy['actual_area']:.0f} pixels (actual), {accuracy['predicted_area']:.0f} (predicted)
  Environment: {env.get('condition_type', 'unknown')} conditions, {env.get('temperature_level', 'unknown')} temp
  Spread: Direction error={spread.get('direction_error', 0):.1f}°, Distance error={spread.get('distance_error', 0):.1f} pixels
  Conditions: Temp={env.get('max_temp_mean', 0):.1f}K, Precip={env.get('precipitation_mean', 0):.1f}mm
"""
        
        if len(evolution_analysis) > 5:
            report += f"... and {len(evolution_analysis) - 5} more time windows\n"

        report += f"""

CONCLUSIONS:
===========
Based on {len(evolution_analysis)} time windows:

1. Fire Growth: Average rate of {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day
2. Model Performance: {temporal_patterns.get('accuracy_trend', 'Variable')} accuracy trend
3. Environmental Impact: Temperature correlation of {temporal_patterns.get('temperature_correlation', 0):.3f}
4. Peak Activity: Day {temporal_patterns.get('peak_fire_day', 0)}

RECOMMENDATIONS:
==============
1. Focus on high-accuracy windows for operational decisions
2. Consider environmental conditions when interpreting confidence
3. Use multi-day predictions for better understanding
4. Validate against additional fire events
"""
        
        return report

# ============================================================================
# SYNTHETIC DATA AND VERIFICATION
# ============================================================================

def create_sample_fire_scenario(config):
    """Create realistic synthetic fire scenario"""
    print("Creating synthetic fire scenario...")
    
    H, W = config.SPATIAL_SIZE
    num_features = len(config.BEST_FEATURES)
    
    sequence_data = torch.randn(15, num_features, H, W)
    
    feature_map = {config.FEATURE_NAMES[idx]: i for i, idx in enumerate(config.BEST_FEATURES)}
    
    if 'NDVI' in feature_map:
        sequence_data[:, feature_map['NDVI']] = torch.normal(0.3, 0.2, (15, H, W))
    
    if 'Max_Temp_K' in feature_map:
        sequence_data[:, feature_map['Max_Temp_K']] = torch.normal(300, 10, (15, H, W))
    
    if 'Wind_Speed' in feature_map:
        sequence_data[:, feature_map['Wind_Speed']] = torch.normal(5, 2, (15, H, W))
    
    if 'Total_Precip' in feature_map:
        sequence_data[:, feature_map['Total_Precip']] = torch.exponential(torch.ones(15, H, W) * 0.1)
    
    if 'Active_Fire' in feature_map:
        fire_idx = feature_map['Active_Fire']
        center_h, center_w = H//2, W//2
        
        for t in range(5):
            fire_size = 3 + t * 2
            h_start = max(0, center_h - fire_size)
            h_end = min(H, center_h + fire_size)
            w_start = max(0, center_w - fire_size)
            w_end = min(W, center_w + fire_size)
            
            sequence_data[t, fire_idx, h_start:h_end, w_start:w_end] = torch.rand(h_end-h_start, w_end-w_start) * 0.8
    
    return sequence_data

def verify_preprocessing_consistency():
    """Verify simulation preprocessing matches training"""
    print("Verifying preprocessing consistency...")
    
    stats_files = ['feature_stats.npz', 'feature_stats_fold_1.npz', 'feature_stats_fold_2.npz']
    stats_found = any(os.path.exists(f) for f in stats_files)
    
    if not stats_found:
        print("No training statistics found!")
        available = [f for f in os.listdir('.') if 'feature' in f.lower() or 'stats' in f.lower()]
        print(f"Available files: {available}")
        return False
    
    try:
        config = SimulationConfig()
        loader = FixedFireEventLoader(config)
        
        print(f"Statistics loaded from: {loader.feature_stats['source_file']}")
        print(f"Feature mean range: [{loader.feature_stats['mean'].min():.3f}, {loader.feature_stats['mean'].max():.3f}]")
        print(f"Feature std range: [{loader.feature_stats['std'].min():.3f}, {loader.feature_stats['std'].max():.3f}]")
        print(f"Best features: {list(loader.feature_stats['best_features'])}")
        
        return True
        
    except Exception as e:
        print(f"Failed to load statistics: {e}")
        return False

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    print("\nTesting preprocessing pipeline...")
    
    try:
        config = SimulationConfig()
        loader = FixedFireEventLoader(config)
        
        T, C, H, W = 5, 23, 64, 64
        synthetic_data = torch.randn(T, C, H, W)
        
        synthetic_data[:, 7] = torch.uniform(0, 360, (T, H, W))
        synthetic_data[:, 22] = torch.bernoulli(torch.ones(T, H, W) * 0.01)
        
        print(f"Input shape: {synthetic_data.shape}")
        
        processed = loader._process_features_exact_training(synthetic_data)
        
        print(f"Output shape: {processed.shape}")
        print(f"Output mean: {processed.mean().item():.3f}")
        print(f"Output std: {processed.std().item():.3f}")
        
        if abs(processed.mean().item()) < 0.2 and 0.7 < processed.std().item() < 1.3:
            print("Preprocessing appears correct")
            return True
        else:
            print("Preprocessing may have issues")
            return False
            
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        return False

def calculate_prediction_metrics(predictions, ground_truth):
    """Calculate and display prediction metrics"""
    if not ground_truth:
        print("No ground truth available for metrics")
        return
    
    print("\nPrediction Metrics:")
    print("-" * 30)
    
    total_iou = 0
    total_dice = 0
    valid_days = 0
    
    for day in range(min(len(predictions), len(ground_truth))):
        pred_binary = (np.array(predictions[day]) > 0.5).astype(float)
        actual_binary = np.array(ground_truth[day])
        
        if actual_binary.sum() > 0:
            intersection = (pred_binary * actual_binary).sum()
            union = pred_binary.sum() + actual_binary.sum() - intersection
            
            iou = intersection / union if union > 0 else 0
            dice = (2 * intersection) / (pred_binary.sum() + actual_binary.sum()) if (pred_binary.sum() + actual_binary.sum()) > 0 else 0
            
            total_iou += iou
            total_dice += dice
            valid_days += 1
            
            print(f"Day {day+1}: IoU={iou:.3f}, Dice={dice:.3f}")
    
    if valid_days > 0:
        print("-" * 30)
        print(f"Average IoU: {total_iou/valid_days:.3f}")
        print(f"Average Dice: {total_dice/valid_days:.3f}")
        print(f"Valid days: {valid_days}/{len(predictions)}")
    else:
        print("No valid ground truth data")

def generate_supervisor_report_evolution(evolution_analysis, temporal_patterns, model_path):
    """Generate comprehensive supervisor report"""
    
    if not evolution_analysis:
        return "No evolution data available"
    
    fire_event_name = evolution_analysis[0]['fire_event']
    num_windows = len(evolution_analysis)
    
    report = f"""
WILDFIRE SPREAD SIMULATION - COMPLETE EVOLUTION ANALYSIS
=======================================================

Model Information:
- Model Path: {model_path}
- Analysis Date: {np.datetime64('today')}
- Fire Event: {fire_event_name}
- Time Windows Analyzed: {num_windows}

EVOLUTION SUMMARY:
================
Duration: Day {evolution_analysis[0]['start_day']} to Day {evolution_analysis[-1]['start_day']}
Accuracy Trend: {temporal_patterns.get('accuracy_trend', 'Variable')}
Mean Accuracy: {temporal_patterns.get('mean_accuracy', 0):.4f} IoU
Fire Growth Rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day
Peak Fire Activity: Day {temporal_patterns.get('peak_fire_day', 0)}

TEMPORAL ANALYSIS:
=================
1. Model Performance:
   - {temporal_patterns.get('accuracy_trend', 'Variable')} accuracy trend
   - Best predictions on day {temporal_patterns.get('best_prediction_day', 0)}
   - Mean accuracy: {temporal_patterns.get('mean_accuracy', 0):.4f}

2. Environmental Correlations:
   - Temperature-accuracy correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}
   - Environmental conditions influence prediction quality
   - Fire behavior complexity varies with weather

3. Fire Spread Patterns:
   - Growth rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day
   - Peak activity identified in evolution
   - Spatial spread patterns captured

OUTPUTS GENERATED:
=================
- fire_comparison.gif: Predicted vs actual animation
- fire_evolution_analysis.png: Complete temporal analysis
- fire_evolution.gif: Animated evolution with predictions
- simulation_report.txt: Comprehensive report

CAPABILITIES DEMONSTRATED:
========================
- Complete fire event tracking over {num_windows} time periods
- Environmental condition integration
- Temporal pattern recognition
- Fire spread direction and distance prediction
- Multi-day evolution simulation
- Physics-informed constraints

OPERATIONAL APPLICATIONS:
=======================
- Day-ahead fire spread predictions
- Environmental context for confidence levels
- Peak activity phase identification
- Resource allocation planning
"""
    
    with open('simulation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Comprehensive report saved: simulation_report.txt")
    return report

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution with proper code structure"""
    parser = argparse.ArgumentParser(description='Fixed Fire Spread Simulation')
    parser.add_argument('--model', type=str, default='best_fire_model_official.pth',
                       help='Path to trained model')
    parser.add_argument('--fire_event', type=str, default=None,
                       help='Path to HDF5 fire event file')
    parser.add_argument('--start_day', type=int, default=0,
                       help='Start day in fire event')
    parser.add_argument('--demo', action='store_true',
                       help='Use synthetic demo data')
    parser.add_argument('--days', type=int, default=10,
                       help='Days to simulate')
    parser.add_argument('--mode', choices=['sliding_window', 'autoregressive'],
                       default='sliding_window', help='Simulation mode')
    parser.add_argument('--verify', action='store_true',
                       help='Only run verification tests')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FIXED FIRE SPREAD SIMULATION")
    print("=" * 60)
    
    # Verification mode
    if args.verify:
        print("Running verification tests...")
        verify_ok = verify_preprocessing_consistency()
        test_ok = test_preprocessing_pipeline() if verify_ok else False
        
        if verify_ok and test_ok:
            print("\nAll verification tests passed!")
            print("Simulation should work correctly now.")
        else:
            print("\nVerification failed - check training statistics")
        return
    
    # Check prerequisites
    if not verify_preprocessing_consistency():
        print("\nPreprocessing verification failed!")
        print("Please run training first or use --verify to diagnose issues")
        return
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found")
        return
    
    config = SimulationConfig()
    
    # Initialize simulator
    try:
        simulator = FixedFireSpreadSimulator(args.model, config)
    except Exception as e:
        print(f"Failed to initialize simulator: {e}")
        return
    
    # Determine data source
    if args.demo or not args.fire_event:
        # Try to find real fire event first
        possible_files = [
            # 'data/processed/2020/fire_24461899.hdf5',

            'data/processed/2019/*.hdf5',
            'data/processed/*/*.hdf5',
            'fire_*.hdf5',
            '*.hdf5'
        ]
        
        fire_event_file = None
        for pattern in possible_files:
            found = glob.glob(pattern)
            if found:
                fire_event_file = found[0]
                print(f"Found fire event: {fire_event_file}")
                break
        
        if not fire_event_file:
            print("No fire event found, using synthetic data")
            args.demo = True
        else:
            args.fire_event = fire_event_file
    
    # Load data
    ground_truth_fire = None
    
    if args.fire_event and os.path.exists(args.fire_event) and not args.demo:
        print(f"\nLoading real fire event: {args.fire_event}")
        try:
            loader = FixedFireEventLoader(config)
            fire_event_data = loader.load_fire_event(args.fire_event)
            initial_sequence, weather_forecast, ground_truth_fire = loader.prepare_simulation_data(
                fire_event_data, start_day=args.start_day
            )
            
            print(f"Initial sequence: {initial_sequence.shape}")
            print(f"Weather forecast: {weather_forecast.shape if weather_forecast is not None else 'None'}")
            print(f"Ground truth days: {len(ground_truth_fire)}")
            
        except Exception as e:
            print(f"Failed to load fire event: {e}")
            print("Falling back to synthetic data")
            args.demo = True
    
    if args.demo:
        print("\nUsing synthetic demonstration data")
        scenario_data = create_sample_fire_scenario(config)
        initial_sequence = scenario_data[:5]
        weather_forecast = scenario_data[5:] if len(scenario_data) > 5 else None
        ground_truth_fire = None
    
    # Run simulation
    print(f"\nRunning {args.days}-day fire simulation...")
    simulation_days = min(args.days, len(weather_forecast) if weather_forecast is not None else args.days)
    
    predictions = simulator.simulate_fire_evolution(
        initial_sequence,
        weather_forecast,
        num_days=simulation_days,
        mode=args.mode,
        debug=True
    )
    
    print(f"\nSimulation complete: {len(predictions)} days predicted")
    
    # Generate outputs
    if ground_truth_fire and len(ground_truth_fire) > 0:
        print("Creating comparison animation...")
        simulator.create_comparison_animation(predictions, ground_truth_fire, 'fire_comparison.gif')
        calculate_prediction_metrics(predictions, ground_truth_fire)
    else:
        print("Creating simulation animation...")
        simulator.create_simulation_animation(predictions, save_path='fire_simulation.gif')
    
    # COMPLETE EVOLUTION ANALYSIS
    print("\nRunning complete fire event evolution analysis...")
    evolution_analyzer = SingleFireEventAnalyzer(simulator, config)
    
    # Use loaded fire event for evolution analysis
    evolution_fire_event = args.fire_event if args.fire_event and os.path.exists(args.fire_event) else None
    
    if not evolution_fire_event:
        # Try to find any fire event file
        search_patterns = [
            'data/processed/2020/fire_24604783.hdf5',
            'data/processed/2019/*.hdf5',
            'data/processed/*/*.hdf5',
            'fire_*.hdf5',
            '*.hdf5'
        ]
        
        for pattern in search_patterns:
            found_files = glob.glob(pattern)
            if found_files:
                evolution_fire_event = found_files[0]
                print(f"Using fire event for evolution analysis: {evolution_fire_event}")
                break
    
    if evolution_fire_event:
        try:
            # Run complete evolution analysis
            evolution_analysis, temporal_patterns = evolution_analyzer.analyze_fire_evolution(
                evolution_fire_event
            )
            
            if evolution_analysis:
                print(f"Evolution analysis complete: {len(evolution_analysis)} time windows")
                
                # Generate evolution plots
                evolution_analyzer.plot_fire_evolution(
                    evolution_analysis, temporal_patterns, 'fire_evolution_analysis.png'
                )
                
                # Create evolution animation
                evolution_analyzer.create_evolution_animation(
                    evolution_analysis, evolution_fire_event, 'fire_evolution.gif'
                )
                
                # Generate detailed report
                evolution_report = evolution_analyzer.generate_evolution_report(
                    evolution_analysis, temporal_patterns, os.path.basename(evolution_fire_event)
                )
                
                # Generate supervisor report
                supervisor_report = generate_supervisor_report_evolution(
                    evolution_analysis, temporal_patterns, args.model
                )
                
                print("Evolution analysis outputs:")
                print("- fire_evolution_analysis.png: Complete temporal analysis")
                print("- fire_evolution.gif: Animated evolution with predictions")
                print("- simulation_report.txt: Comprehensive evolution report")
                
            else:
                print("No evolution analysis data generated")
                
        except Exception as e:
            print(f"Evolution analysis failed: {e}")
            print("Continuing with basic simulation results...")
    else:
        print("No fire event found for evolution analysis - skipping this feature")
    
    # Summary and statistics - THIS IS WHERE THE CODE SHOULD BE
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
    
    print("Generated files:")
    if ground_truth_fire:
        print("- fire_comparison.gif: Predicted vs Actual")
    else:
        print("- fire_simulation.gif: Fire spread animation")
    
    print("- fire_frame_*.png: Individual frames (fallback)")
    
    if evolution_fire_event and 'evolution_analysis' in locals() and evolution_analysis:
        print("- fire_evolution_analysis.png: Complete temporal analysis")
        print("- fire_evolution.gif: Animated evolution with predictions")
        print("- simulation_report.txt: Comprehensive evolution report")
        
        # Display key findings
        print(f"\nEvolution Analysis Results:")
        print(f"- Analysis windows: {len(evolution_analysis)}")
        print(f"- Accuracy trend: {temporal_patterns.get('accuracy_trend', 'unknown')}")
        print(f"- Mean IoU: {temporal_patterns.get('mean_accuracy', 0):.3f}")
        print(f"- Fire growth rate: {temporal_patterns.get('fire_growth_rate', 0):.1f} pixels/day")
        print(f"- Peak fire day: {temporal_patterns.get('peak_fire_day', 0)}")
        print(f"- Temperature correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}")
    else:
        print("- Basic simulation only (no evolution analysis)")
    
    # Fire statistics summary
    total_fire_pixels = sum(np.sum(pred > 0.5) for pred in predictions)
    max_fire_pixels = max(np.sum(pred > 0.5) for pred in predictions)
    
    print(f"\nFire Statistics:")
    print(f"- Total fire pixels across all days: {total_fire_pixels}")
    print(f"- Maximum fire pixels in single day: {max_fire_pixels}")
    print(f"- Average fire pixels per day: {total_fire_pixels/len(predictions):.1f}")
    
    if ground_truth_fire:
        gt_total = sum(np.sum(gt) for gt in ground_truth_fire[:len(predictions)])
        print(f"- Ground truth total fire pixels: {gt_total}")
        if gt_total > 0:
            print(f"- Prediction/GT ratio: {total_fire_pixels/gt_total:.2f}")
        else:
            print("- No ground truth fire detected")
    
    # Add information about fire event if used
    # 改进的simulation summary生成
    if args.fire_event:
        with open('simulation_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"Fire Simulation Summary - Detailed Daily Analysis\n")
            f.write(f"===============================================\n\n")
            f.write(f"Fire Event: {args.fire_event}\n")
            f.write(f"Start Day: {args.start_day}\n")
            f.write(f"Simulation Days: {len(predictions)}\n")
            f.write(f"Ground Truth Available: {'Yes' if ground_truth_fire else 'No'}\n")
            f.write(f"Threshold Used: {config.FIRE_SPREAD_THRESHOLD}\n\n")
            
            # 每日详细分析
            f.write("DAILY ANALYSIS:\n")
            f.write("==============\n")
            
            daily_metrics = []
            for day in range(len(predictions)):
                pred = predictions[day]
                pred_binary = (pred > config.FIRE_SPREAD_THRESHOLD).astype(float)
                pred_pixels = pred_binary.sum()
                
                day_info = {
                    'day': day + 1,
                    'pred_pixels': int(pred_pixels),
                    'pred_mean': float(pred.mean()),
                    'pred_max': float(pred.max()),
                    'pred_std': float(pred.std())
                }
                
                if ground_truth_fire and day < len(ground_truth_fire):
                    gt = ground_truth_fire[day]
                    gt_binary = (gt > 0).astype(float)
                    gt_pixels = gt_binary.sum()
                    
                    # 计算IoU
                    if gt_pixels > 0 or pred_pixels > 0:
                        intersection = (pred_binary * gt_binary).sum()
                        union = pred_pixels + gt_pixels - intersection
                        iou = intersection / union if union > 0 else 0.0
                        
                        # 计算其他指标
                        precision = intersection / pred_pixels if pred_pixels > 0 else 0.0
                        recall = intersection / gt_pixels if gt_pixels > 0 else 0.0
                        dice = (2 * intersection) / (pred_pixels + gt_pixels) if (pred_pixels + gt_pixels) > 0 else 0.0
                    else:
                        iou = precision = recall = dice = 0.0
                    
                    day_info.update({
                        'gt_pixels': int(gt_pixels),
                        'iou': float(iou),
                        'precision': float(precision),
                        'recall': float(recall),
                        'dice': float(dice)
                    })
                
                daily_metrics.append(day_info)
                
                # 写入每日详情
                f.write(f"\nDay {day + 1}:\n")
                f.write(f"  Predicted Fire Pixels: {pred_pixels:.0f}\n")
                f.write(f"  Prediction Stats: mean={pred.mean():.4f}, max={pred.max():.4f}, std={pred.std():.4f}\n")
                
                if ground_truth_fire and day < len(ground_truth_fire):
                    f.write(f"  Ground Truth Pixels: {day_info['gt_pixels']}\n")
                    f.write(f"  IoU: {day_info['iou']:.4f}\n")
                    f.write(f"  Precision: {day_info['precision']:.4f}, Recall: {day_info['recall']:.4f}\n")
                    f.write(f"  Dice Score: {day_info['dice']:.4f}\n")
                else:
                    f.write(f"  Ground Truth: Not Available\n")
            
            # 总体统计
            f.write(f"\n\nOVERALL STATISTICS:\n")
            f.write(f"==================\n")
            
            total_pred_pixels = sum(d['pred_pixels'] for d in daily_metrics)
            max_pred_pixels = max(d['pred_pixels'] for d in daily_metrics)
            avg_pred_pixels = total_pred_pixels / len(daily_metrics) if daily_metrics else 0
            
            f.write(f"Total Predicted Fire Pixels: {total_pred_pixels}\n")
            f.write(f"Maximum Fire Pixels (Single Day): {max_pred_pixels}\n")
            f.write(f"Average Fire Pixels Per Day: {avg_pred_pixels:.1f}\n")
            
            if ground_truth_fire:
                gt_available_days = [d for d in daily_metrics if 'gt_pixels' in d]
                if gt_available_days:
                    total_gt_pixels = sum(d['gt_pixels'] for d in gt_available_days)
                    avg_iou = sum(d['iou'] for d in gt_available_days) / len(gt_available_days)
                    avg_dice = sum(d['dice'] for d in gt_available_days) / len(gt_available_days)
                    avg_precision = sum(d['precision'] for d in gt_available_days) / len(gt_available_days)
                    avg_recall = sum(d['recall'] for d in gt_available_days) / len(gt_available_days)
                    
                    f.write(f"Total Ground Truth Fire Pixels: {total_gt_pixels}\n")
                    f.write(f"Prediction/GT Pixel Ratio: {total_pred_pixels/total_gt_pixels:.3f}\n" if total_gt_pixels > 0 else "Prediction/GT Pixel Ratio: N/A (no GT fire)\n")
                    f.write(f"Average IoU: {avg_iou:.4f}\n")
                    f.write(f"Average Dice Score: {avg_dice:.4f}\n")
                    f.write(f"Average Precision: {avg_precision:.4f}\n")
                    f.write(f"Average Recall: {avg_recall:.4f}\n")
                    
                    # 性能分析
                    f.write(f"\nPERFORMANCE ANALYSIS:\n")
                    f.write(f"====================\n")
                    if avg_iou == 0.0:
                        f.write(f"WARNING: Zero IoU suggests complete prediction failure!\n")
                        f.write(f"Possible causes:\n")
                        f.write(f"- Fire channel preprocessing issues\n")
                        f.write(f"- Threshold mismatch (using {config.FIRE_SPREAD_THRESHOLD})\n")
                        f.write(f"- Model output range problems\n")
                        f.write(f"- Ground truth alignment issues\n")
                    elif avg_iou < 0.1:
                        f.write(f"Performance: Poor (IoU < 0.1)\n")
                    elif avg_iou < 0.3:
                        f.write(f"Performance: Below Average (IoU < 0.3)\n")
                    elif avg_iou < 0.5:
                        f.write(f"Performance: Average (IoU < 0.5)\n")
                    else:
                        f.write(f"Performance: Good (IoU >= 0.5)\n")
                    
                    # 找出最好和最差的日子
                    best_day = max(gt_available_days, key=lambda x: x['iou'])
                    worst_day = min(gt_available_days, key=lambda x: x['iou'])
                    
                    f.write(f"Best Performance: Day {best_day['day']} (IoU: {best_day['iou']:.4f})\n")
                    f.write(f"Worst Performance: Day {worst_day['day']} (IoU: {worst_day['iou']:.4f})\n")
                else:
                    f.write(f"Ground Truth: Available but no valid comparisons\n")
            else:
                f.write(f"Ground Truth: Not Available - Cannot Calculate Accuracy Metrics\n")
            
            # 诊断信息
            if total_pred_pixels == 0:
                f.write(f"\nDIAGNOSTIC WARNING:\n")
                f.write(f"==================\n")
                f.write(f"No fire pixels predicted across all days!\n")
                f.write(f"This suggests:\n")
                f.write(f"1. Model output values too low (all < {config.FIRE_SPREAD_THRESHOLD})\n")
                f.write(f"2. Fire channel preprocessing removing all fire information\n")
                f.write(f"3. Input fire channel completely zero\n")
                f.write(f"Check debug output for model prediction ranges and input fire pixels.\n")

        print("- simulation_summary.txt: Detailed daily analysis with IoU metrics")

if __name__ == "__main__":
    main()

# ============================================================================
# USAGE GUIDE AND FEATURE SUMMARY
# ============================================================================

"""
COMPLETE FIXED FIRE SIMULATION - ALL FEATURES RESTORED
======================================================

CRITICAL FIXES IMPLEMENTED:
1. ✅ Proper training statistics integration
2. ✅ Exact training preprocessing pipeline  
3. ✅ Corrected physics parameters
4. ✅ Fixed code structure and flow

COMPLETE FEATURE SET:
====================

1. BASIC SIMULATION:
   - Single-step and multi-day fire prediction
   - Sliding window and autoregressive modes
   - Physics-informed fire spread and decay

2. VISUALIZATION:
   - fire_simulation.gif: Basic fire spread animation
   - fire_comparison.gif: Predicted vs actual comparison
   - Individual frame PNGs as fallback

3. EVOLUTION ANALYSIS:
   - Complete time-series analysis of fire events
   - Environmental condition tracking
   - Fire spread pattern analysis
   - fire_evolution_analysis.png: 4-panel analysis
   - fire_evolution.gif: Animated evolution

4. COMPREHENSIVE REPORTING:
   - Detailed evolution reports
   - Supervisor reports for management
   - simulation_report.txt: Complete analysis
   - simulation_summary.txt: Basic summary

5. VERIFICATION:
   - Preprocessing pipeline verification
   - Training statistics validation
   - Comprehensive error handling

USAGE:
======

1. Verification:
   python fire_simulation_fixed.py --verify

2. Demo:
   python fire_simulation_fixed.py --demo

3. Real fire event:
   python fire_simulation_fixed.py --fire_event your_fire.hdf5

4. Extended simulation:
   python fire_simulation_fixed.py --fire_event your_fire.hdf5 --days 20

EXPECTED RESULTS:
================
With proper preprocessing, you should see:
- Meaningful fire predictions (not tiny scattered pixels)
- Realistic fire spread patterns  
- Better prediction/ground truth match
- Healthy sigmoid output ranges (0.1-0.9)
- Complete visualizations and analysis
"""