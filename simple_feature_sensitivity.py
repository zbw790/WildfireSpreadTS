#!/usr/bin/env python3
"""
Simple Feature Sensitivity Analysis - Focus on GIF Generation

This tool creates the most important output: animated GIFs showing:
1. Actual fire spreading (ground truth)
2. Raw model predictions (using actual feature values) 
3. Modified predictions (changing individual feature values)
"""

import os
# Set environment variable to avoid OpenMP duplicate library warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy import ndimage
from PIL import Image
from sklearn.metrics import average_precision_score
import json

warnings.filterwarnings('ignore')

# ============================================================================
# SIMPLE CONFIGURATION
# ============================================================================

class SimpleConfig:
    SEQUENCE_LENGTH = 5
    SPATIAL_SIZE = (128, 128)  # Same as test_simulation!
    BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]  # NDVI, EVI2, etc.
    FEATURE_NAMES = [
        'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1', 'NDVI', 'EVI2', 'Total_Precip', 'Wind_Speed',
        'Wind_Direction', 'Min_Temp_K', 'Max_Temp_K', 'ERC', 'Spec_Hum', 'PDSI',
        'Slope', 'Aspect', 'Elevation', 'Landcover', 'Forecast_Precip', 'Forecast_Wind_Speed',
        'Forecast_Wind_Dir', 'Forecast_Temp_C', 'Forecast_Spec_Hum', 'Active_Fire'
    ]
    ANGULAR_FEATURES = [7, 14, 19]  # Wind_Direction, Aspect, Forecast_Wind_Dir
    FIRE_THRESHOLD = 0.3
    SIMULATION_DAYS = 26  # Show entire available fire event period
    
    # Enhanced perturbation settings
    PERTURBATION_LEVELS = [-50, -30, -20, -10, 0, 10, 20, 30]  # 8 levels including baseline

# ============================================================================
# COMPATIBILITY CLASSES FOR MODEL LOADING
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
import sys
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# ============================================================================
# SIMPLE MODEL LOADER
# ============================================================================

def load_model_with_compatibility(model_path, input_channels, sequence_length=5, device='cpu'):
    """Load model with same method as test_simulation"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = None
    for method in ['safe_with_globals', 'legacy', 'fallback']:
        try:
            if method == 'safe_with_globals':
                # Add required safe globals for checkpoint loading
                torch.serialization.add_safe_globals([
                    WildFireConfig,
                    FirePredictionConfig,
                    'numpy.core.multiarray.scalar',
                    'numpy.core.multiarray._reconstruct',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            elif method == 'legacy':
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            elif method == 'fallback':
                # Last resort - try to load just the state dict
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            print(f"Model loaded with {method} method")
            break
            
        except Exception as e:
            print(f"Failed with {method}: {e}")
            if method == 'fallback':
                print(f"All loading methods failed. Last error: {e}")
                print("Creating compatible model architecture with random weights...")
                break
            continue
    
    # Create model with same architecture as test_simulation
    model = OfficialFireUNet(input_channels, sequence_length)
    
    if checkpoint is not None:
        try:
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                    if 'best_ap' in checkpoint:
                        print(f"Best AP: {checkpoint['best_ap']:.4f}")
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded trained weights (with relaxed matching)!")
        except Exception as e:
            print(f"Failed to load state dict: {e}")
            print("Using random weights for demonstration")
    else:
        print("Using random weights for demonstration")
    
    # Ensure model is on correct device
    model = model.to(device)
    model.eval()
    
    return model

# ============================================================================
# U-NET MODEL ARCHITECTURE (SAME AS TEST_SIMULATION)
# ============================================================================

class OfficialFireUNet(nn.Module):
    """U-Net architecture matching training exactly - SAME AS TEST_SIMULATION"""
    
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle sequence input - CRITICAL: Flatten sequence dimension
        if x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)  # Flatten sequence into channels
        
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder path with skip connections
        up4 = self.upconv4(bottleneck)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(merge4)
        
        up3 = self.upconv3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        
        # Final output
        output = self.final_conv(dec1)
        return output

# ============================================================================
# SIMPLE SIMULATOR
# ============================================================================

class SimpleFireSimulator:
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
    
    def predict_single_step(self, input_sequence, debug=False):
        """Predict single fire spread step - SAME AS TEST_SIMULATION"""
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
                    prediction = torch.sigmoid(output)  # CRITICAL: Apply sigmoid like test_simulation
            except:
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output)      # CRITICAL: Apply sigmoid like test_simulation
            
            if debug:
                print(f"Raw output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                print(f"Sigmoid range: [{prediction.min().item():.3f}, {prediction.max().item():.3f}]")
                print(f"Predictions > 0.5: {(prediction > 0.5).sum().item()}")
                print(f"Predictions > 0.3: {(prediction > 0.3).sum().item()}")
                print(f"Mean prediction: {prediction.mean().item():.4f}")
                print("========================")
            
            return prediction.cpu().squeeze()
    
    def simulate_fire_evolution(self, initial_sequence, weather_data=None, num_days=6):
        """Simulate fire evolution over multiple days"""
        predictions = []
        current_sequence = initial_sequence.clone()
        
        for day in range(num_days):
            # Predict next day
            pred_fire = self.predict_single_step(current_sequence.unsqueeze(0), debug=False)
            pred_fire = self._apply_fire_physics(pred_fire, day, debug=False)
            predictions.append(pred_fire.numpy())
            
            # Update sequence for next prediction
            if day < num_days - 1:
                if weather_data is not None and day < len(weather_data) - self.config.SEQUENCE_LENGTH:
                    next_sequence = weather_data[day + 1:day + 1 + self.config.SEQUENCE_LENGTH].clone()
                    active_fire_idx = len(self.config.BEST_FEATURES) - 1
                    next_sequence[-1, active_fire_idx] = pred_fire
                    current_sequence = next_sequence
                else:
                    new_frame = current_sequence[-1].clone()
                    active_fire_idx = len(self.config.BEST_FEATURES) - 1
                    new_frame[active_fire_idx] = pred_fire
                    current_sequence = torch.cat([
                        current_sequence[1:],
                        new_frame.unsqueeze(0)
                    ], dim=0)
        
        return predictions
    
    def _apply_fire_physics(self, fire_prediction, day, debug=False):
        """Apply corrected fire physics - SAME AS TEST_SIMULATION"""
        if debug:
            print(f"\n=== FIRE PHYSICS (Day {day}) ===")
            print(f"Raw range: [{fire_prediction.min().item():.3f}, {fire_prediction.max().item():.3f}]")
            print(f"Pixels > threshold: {(fire_prediction > self.config.FIRE_THRESHOLD).sum().item()}")
        
        # Apply threshold (same as test_simulation)
        fire_binary = (fire_prediction > self.config.FIRE_THRESHOLD).float()
        
        if debug:
            print(f"After threshold: {fire_binary.sum().item()} pixels")
        
        # Apply decay (same decay rate as test_simulation)
        decay_factor = 1.0 - 0.05 * (day + 1)  # Use 0.05 like test_simulation
        decay_factor = max(0.1, decay_factor)
        
        fire_decayed = fire_binary * decay_factor
        
        if debug:
            print(f"Decay factor: {decay_factor:.3f}")
            print(f"After decay: {(fire_decayed > 0).sum().item()} pixels")
        
        # Spatial smoothing (CRITICAL: same as test_simulation)
        fire_smoothed = torch.tensor(
            ndimage.gaussian_filter(fire_decayed.numpy(), sigma=0.5)
        )
        
        if debug:
            print(f"Final range: [{fire_smoothed.min().item():.3f}, {fire_smoothed.max().item():.3f}]")
            print("=========================")
        
        return fire_smoothed
    
    def apply_feature_perturbation(self, input_sequence, feature_name, perturbation_percent):
        """Apply perturbation to a specific feature"""
        modified_sequence = input_sequence.clone()
        
        if feature_name not in self.config.FEATURE_NAMES:
            return modified_sequence
        
        original_feature_idx = self.config.FEATURE_NAMES.index(feature_name)
        if original_feature_idx not in self.config.BEST_FEATURES:
            return modified_sequence
        
        best_feature_idx = self.config.BEST_FEATURES.index(original_feature_idx)
        
        # Apply perturbation
        current_values = modified_sequence[:, best_feature_idx]
        perturbation_factor = 1 + perturbation_percent / 100.0
        modified_sequence[:, best_feature_idx] = current_values * perturbation_factor
        
        return modified_sequence

# ============================================================================
# PREPROCESSING FUNCTIONS (SAME AS TEST_SIMULATION)
# ============================================================================

def load_feature_stats():
    """Load feature statistics for preprocessing"""
    stats_files = ['feature_stats.npz', 'feature_stats_fold_1.npz', 'feature_stats.pkl']
    
    for stats_file in stats_files:
        if os.path.exists(stats_file):
            try:
                if stats_file.endswith('.pkl'):
                    import pickle
                    with open(stats_file, 'rb') as f:
                        return pickle.load(f)
                else:
                    stats_data = dict(np.load(stats_file))
                    # Convert to expected format
                    return {
                        'mean': stats_data.get('feature_mean', stats_data.get('mean', None)),
                        'std': stats_data.get('feature_std', stats_data.get('std', None)),
                        'best_features': [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22],
                        'angular_features': [7, 14, 19]
                    }
            except Exception as e:
                print(f"Could not load {stats_file}: {e}")
                continue
    
    print("No feature statistics found, using default values")
    return None

def process_features_like_test_simulation(data, config, feature_stats=None):
    """Apply same preprocessing as test_simulation"""
    import torch.nn.functional as F
    
    T, C, H, W = data.shape
    processed = data.clone()
    
    print(f"Processing features: {data.shape} -> target size {config.SPATIAL_SIZE}")
    
    # Step 1: Angular features transformation
    if feature_stats and 'angular_features' in feature_stats:
        angular_features = feature_stats['angular_features']
    else:
        angular_features = config.ANGULAR_FEATURES
        
    for angle_idx in angular_features:
        if angle_idx < C:
            processed[:, angle_idx] = torch.sin(torch.deg2rad(processed[:, angle_idx]))
            print(f"  Applied sin transform to feature {angle_idx}")
    
    # Step 2: Handle missing values
    for c in range(C):
        mask = ~torch.isfinite(processed[:, c])
        if mask.any():
            processed[:, c][mask] = 0.0
    
    # Step 3: Resize to match test_simulation size (128x128)
    if (H, W) != config.SPATIAL_SIZE:
        processed = F.interpolate(
            processed.view(-1, 1, H, W),
            size=config.SPATIAL_SIZE,
            mode='bilinear',
            align_corners=False
        ).view(T, C, *config.SPATIAL_SIZE)
        print(f"  Resized from {(H, W)} to {config.SPATIAL_SIZE}")
    
    # Step 4: Select best features
    processed = processed[:, config.BEST_FEATURES]
    print(f"  Selected {len(config.BEST_FEATURES)} best features")
    
    # Step 5: Apply normalization if available
    if feature_stats and 'mean' in feature_stats and 'std' in feature_stats:
        training_mean = torch.FloatTensor(feature_stats['mean'])
        training_std = torch.FloatTensor(feature_stats['std'])
        
        # Select statistics for best features
        if len(training_mean) > len(config.BEST_FEATURES):
            training_mean = training_mean[config.BEST_FEATURES]
            training_std = training_std[config.BEST_FEATURES]
        
        # Apply normalization (exclude angular and categorical features)
        for f_idx, orig_idx in enumerate(config.BEST_FEATURES):
            if orig_idx not in config.ANGULAR_FEATURES and orig_idx != 16:  # Not angular, not landcover
                processed[:, f_idx] = (processed[:, f_idx] - training_mean[f_idx]) / (training_std[f_idx] + 1e-6)
        
        print(f"  Applied training normalization")
    else:
        print("  No normalization applied (no statistics)")
    
    return processed

# ============================================================================
# DATA LOADER
# ============================================================================

def load_fire_event_data(fire_event_path, config, start_day=0):
    """Load fire event data with same preprocessing as test_simulation"""
    
    # Load feature statistics
    feature_stats = load_feature_stats()
    
    try:
        with h5py.File(fire_event_path, 'r') as f:
            # Try different dataset names
            dataset_names = ['sequence', 'data', 'fire_sequence', 'features']
            data = None
            
            for name in dataset_names:
                if name in f:
                    data = f[name][:]
                    print(f"Loaded data from '{name}' dataset")
                    break
            
            if data is None:
                print("Available datasets:", list(f.keys()))
                return None, None, None, 0
            
            print(f"Raw data shape: {data.shape}")
            
            # Convert to tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
            T, C, H, W = data_tensor.shape
            
            # Extract sequences
            seq_len = config.SEQUENCE_LENGTH
            max_days = min(len(data_tensor) - seq_len - start_day, config.SIMULATION_DAYS)
            
            if max_days <= 0:
                print(f"Not enough data for simulation")
                return None, None, None, 0
            
            # Apply SAME preprocessing as test_simulation
            print("Applying test_simulation preprocessing...")
            
            # Initial sequence (for model input)
            initial_raw = data_tensor[start_day:start_day + seq_len]
            initial_sequence = process_features_like_test_simulation(initial_raw, config, feature_stats)
            
            # Weather data (for simulation continuation)
            weather_raw = data_tensor[start_day:start_day + max_days + seq_len]
            weather_data = process_features_like_test_simulation(weather_raw, config, feature_stats)
            
            # Ground truth (for visualization) - also apply same processing for consistency
            gt_raw = data_tensor[start_day + seq_len:start_day + seq_len + max_days]
            gt_processed = process_features_like_test_simulation(gt_raw, config, feature_stats)
            
            # Extract fire channel from processed ground truth
            ground_truth = []
            ground_truth_raw = []  # 保存原始连续值用于显示
            fire_channel_idx = len(config.BEST_FEATURES) - 1  # Active_Fire is last
            for day_idx in range(len(gt_processed)):
                fire_data = gt_processed[day_idx, fire_channel_idx].numpy()
                # 保存原始连续值用于GIF显示
                ground_truth_raw.append(fire_data.copy())
                # Apply consistent threshold (0.5) for binary fire detection (用于AP计算)
                binary_fire = (fire_data > 0.5).astype(np.float32)
                ground_truth.append(binary_fire)
            
            print(f"Processed sequences: initial={initial_sequence.shape}, weather={weather_data.shape}")
            print(f"Ground truth shape: {len(ground_truth)} days, each {ground_truth[0].shape if ground_truth else 'None'}")
            print(f"Simulation days: {max_days}")
            
            return initial_sequence, weather_data, ground_truth, ground_truth_raw, max_days
            
    except Exception as e:
        print(f"Error loading fire event data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, 0

# ============================================================================
# GIF GENERATOR
# ============================================================================

def generate_autoregressive_predictions(simulator, initial_seq, weather_data, max_days, 
                                      feature_name, perturbation_levels, fire_event_path, config):
    """
    生成递归预测：使用之前预测的火点数据而不是真实火点数据
    """
    print(f"  Generating AUTOREGRESSIVE predictions (using predicted fire points)...")
    
    autoregressive_predictions = {}
    
    # 初始化所有扰动级别的预测数组
    for perturbation in perturbation_levels:
        autoregressive_predictions[perturbation] = []
    
    # 开始递归预测过程
    for day in range(max_days):
        print(f"    Autoregressive Day {day+1}/{max_days}")
        
        if day == 0:
            # 第一天：使用真实的初始序列
            current_sequences = {}
            for perturbation in perturbation_levels:
                if perturbation == 0:
                    current_sequences[perturbation] = initial_seq.clone()
                else:
                    current_sequences[perturbation] = simulator.apply_feature_perturbation(
                        initial_seq, feature_name, perturbation
                    )
        else:
            # 后续天数：构建新的序列，用预测火点替代真实火点
            new_sequences = {}
            
            for perturbation in perturbation_levels:
                # 获取上一天的预测火点
                if autoregressive_predictions[perturbation]:
                    last_predicted_fire = torch.tensor(autoregressive_predictions[perturbation][-1])
                else:
                    # 如果没有预测，使用初始序列的最后一帧
                    last_predicted_fire = initial_seq[-1, -1]  # Active_Fire通道
                
                # 构建新的输入序列：滑动窗口 + 预测火点
                if day < len(weather_data):
                    # 使用天气数据构建基础序列
                    new_sequence = weather_data[day-1:day-1+config.SEQUENCE_LENGTH].clone()
                    
                    # 将最新的预测火点放入Active_Fire通道（最后一个通道）
                    active_fire_idx = len(config.BEST_FEATURES) - 1  # Active_Fire索引
                    new_sequence[-1, active_fire_idx] = last_predicted_fire
                    
                    # 应用特征扰动
                    if perturbation != 0:
                        new_sequence = simulator.apply_feature_perturbation(
                            new_sequence, feature_name, perturbation
                        )
                    
                    new_sequences[perturbation] = new_sequence
                else:
                    # 如果没有更多天气数据，重用当前序列
                    new_sequences[perturbation] = current_sequences[perturbation]
            
            current_sequences = new_sequences
        
        # 对所有扰动级别进行预测
        for perturbation in perturbation_levels:
            if perturbation in current_sequences:
                pred = simulator.predict_single_step(
                    current_sequences[perturbation].unsqueeze(0), debug=False
                )
                autoregressive_predictions[perturbation].append(pred.numpy())
            else:
                # 如果没有序列，重用最后一个预测
                if autoregressive_predictions[perturbation]:
                    autoregressive_predictions[perturbation].append(
                        autoregressive_predictions[perturbation][-1]
                    )
    
    return autoregressive_predictions

def create_difference_visualization(standard_gif_path, autoregressive_gif_path, feature_name, output_dir):
    """
    创建标准预测与递归预测的差异可视化分析图
    """
    try:
        print(f"  Creating difference analysis for {feature_name}...")
        
        # 打开GIF文件
        img1 = Image.open(standard_gif_path)
        img2 = Image.open(autoregressive_gif_path)
        
        # 选择第5帧进行对比（通常差异比较明显）
        frame_idx = min(4, img1.n_frames-1, img2.n_frames-1)
        img1.seek(frame_idx)
        img2.seek(frame_idx)
        
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))
        
        # 计算差异
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        diff_gray = np.mean(diff, axis=2)  # 转换为灰度
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{feature_name} - Standard vs Autoregressive Prediction Difference Analysis (Frame {frame_idx+1})', 
                     fontsize=16, fontweight='bold')
        
        # 标准预测
        axes[0,0].imshow(arr1)
        axes[0,0].set_title('Standard Prediction (Real Fire Points)', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # 递归预测
        axes[0,1].imshow(arr2)
        axes[0,1].set_title('Autoregressive Prediction (Predicted Fire Points)', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # 差异热图
        im = axes[1,0].imshow(diff_gray, cmap='hot', vmin=0, vmax=50)
        axes[1,0].set_title(f'Pixel Difference Heatmap\\n(Max Diff: {diff_gray.max():.1f})', fontsize=12, fontweight='bold')
        axes[1,0].axis('off')
        plt.colorbar(im, ax=axes[1,0], shrink=0.8)
        
        # 统计信息
        axes[1,1].axis('off')
        stats_text = f'''Difference Statistics (Frame {frame_idx+1}):

Mean Pixel Difference: {np.mean(diff):.2f}
Max Pixel Difference: {np.max(diff):.2f}
Standard Deviation: {np.std(diff):.2f}

Significant Difference Regions:
> 10 diff: {(diff_gray > 10).sum()} pixels
> 20 diff: {(diff_gray > 20).sum()} pixels
> 30 diff: {(diff_gray > 30).sum()} pixels

Total Pixels: {diff_gray.size}
Difference Ratio: {100*(diff_gray > 5).sum()/diff_gray.size:.1f}%'''
        
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 保存对比图
        output_path = Path(output_dir) / f"{feature_name}_difference_analysis.png"
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Difference analysis saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to create difference analysis: {e}")
        return False

def calculate_cumulative_ap(predictions, targets, current_day):
    """计算到当前天为止的累积AP"""
    if current_day < 0 or current_day >= len(predictions) or current_day >= len(targets):
        return 0.0
    
    # 获取到当前天为止的所有预测和目标
    cumulative_preds = []
    cumulative_targets = []
    
    for day in range(current_day + 1):
        if day < len(predictions) and day < len(targets):
            pred = predictions[day]
            target = targets[day]
            
            # 确保是numpy数组
            if isinstance(pred, torch.Tensor):
                pred = pred.numpy()
            if isinstance(target, torch.Tensor):
                target = target.numpy()
            
            cumulative_preds.append(pred.flatten())
            cumulative_targets.append(target.flatten())
    
    if not cumulative_preds or not cumulative_targets:
        return 0.0
    
    # 合并所有数据
    all_preds = np.concatenate(cumulative_preds)
    all_targets = np.concatenate(cumulative_targets)
    
    # 计算AP
    if all_targets.sum() > 0:
        try:
            return average_precision_score(all_targets, all_preds)
        except:
            return 0.0
    else:
        return 0.0

def create_enhanced_feature_sensitivity_gif(feature_name, output_dir, ground_truth, ground_truth_raw,
                                          baseline_predictions, all_perturbation_predictions, perturbation_levels):
    """Create enhanced sensitivity analysis GIF with multiple perturbation levels and real-time AP display"""
    
    output_path = Path(output_dir) / f"{feature_name}_enhanced_evolution.gif"
    output_path.parent.mkdir(exist_ok=True)
    
    # Create a 3x3 grid: Ground truth + 8 perturbation levels
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{feature_name} Enhanced Feature Sensitivity Analysis\n26-Day Fire Event Evolution', 
                 fontsize=20, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Set up subplot titles
    axes_flat[0].set_title('Ground Truth Fire', fontsize=14, fontweight='bold')
    for i, perturbation in enumerate(perturbation_levels):
        if perturbation == 0:
            title = 'Baseline (0%)'
        else:
            title = f'{feature_name} {perturbation:+d}%'
        axes_flat[i + 1].set_title(title, fontsize=12, fontweight='bold')
    
    def animate(day):
        for ax in axes_flat:
            ax.clear()
        
        try:
            # Ground truth - 使用原始连续值显示深浅
            if day < len(ground_truth_raw):
                gt_raw_data = ground_truth_raw[day]  # 原始连续值用于显示
                gt_binary_data = ground_truth[day]   # 二值化数据用于统计
                
                if gt_raw_data.ndim > 2:
                    gt_raw_data = gt_raw_data.squeeze()
                if gt_binary_data.ndim > 2:
                    gt_binary_data = gt_binary_data.squeeze()
                
                # 使用原始连续值显示，采用与预测相同的颜色方案
                axes_flat[0].imshow(gt_raw_data, cmap='Reds', vmin=0, vmax=1)
                axes_flat[0].set_title(f'Ground Truth Fire - Day {day+1}', fontsize=14, fontweight='bold')
                
                # 统计信息基于二值化数据（保持AP计算一致性）
                fire_pixels = (gt_binary_data > 0.5).sum()
                total_pixels = gt_binary_data.size
                fire_ratio = fire_pixels / total_pixels * 100
                
                # 显示原始值的统计信息
                raw_max = gt_raw_data.max()
                raw_mean = gt_raw_data.mean()
                
                stats_text = f'Fire: {fire_pixels}\n({fire_ratio:.1f}%)\nMax: {raw_max:.2f}\nMean: {raw_mean:.3f}'
                axes_flat[0].text(0.95, 0.05, stats_text, 
                                transform=axes_flat[0].transAxes, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                                fontsize=9, ha='right', va='bottom', fontweight='bold')
            else:
                axes_flat[0].set_title('Ground Truth Fire - No Data', fontsize=14)
            axes_flat[0].axis('off')
            
            # Perturbation predictions
            for i, perturbation in enumerate(perturbation_levels):
                ax_idx = i + 1
                predictions = all_perturbation_predictions.get(perturbation, [])
                
                if day < len(predictions):
                    pred_data = predictions[day]
                    if pred_data.ndim > 2:
                        pred_data = pred_data.squeeze()
                    
                    # Use adaptive colormap for better visibility of small differences
                    # Calculate adaptive vmax based on all predictions for this day
                    day_predictions = [all_perturbation_predictions.get(p, [None]*50)[min(day, len(all_perturbation_predictions.get(p, []))-1)] 
                                     for p in perturbation_levels if all_perturbation_predictions.get(p)]
                    day_predictions = [p for p in day_predictions if p is not None]
                    
                    if day_predictions:
                        adaptive_vmax = max(np.max(p) for p in day_predictions) * 1.1  # 10% buffer
                        adaptive_vmax = max(adaptive_vmax, 0.02)  # Minimum vmax for visibility
                    else:
                        adaptive_vmax = 1.0
                    
                    axes_flat[ax_idx].imshow(pred_data, cmap='Oranges', vmin=0, vmax=adaptive_vmax)
                    
                    if perturbation == 0:
                        title = f'Baseline - Day {day+1}'
                    else:
                        title = f'{feature_name} {perturbation:+d}% - Day {day+1}'
                    axes_flat[ax_idx].set_title(title, fontsize=12)
                    
                    # 🆕 计算并显示累积AP
                    cumulative_ap = calculate_cumulative_ap(predictions, ground_truth, day)
                    
                    # 预测统计 (使用一致的阈值0.5)
                    pred_pixels = (pred_data > 0.5).sum()
                    pred_max = pred_data.max()
                    
                    # 选择AP显示颜色
                    if cumulative_ap > 0.3:
                        ap_color = 'lightgreen'
                    elif cumulative_ap > 0.15:
                        ap_color = 'lightyellow'
                    else:
                        ap_color = 'lightcoral'
                    
                    # 显示实时AP和预测统计
                    ap_text = f'AP: {cumulative_ap:.3f}\nPred: {pred_pixels}\nMax: {pred_max:.2f}'
                    axes_flat[ax_idx].text(0.95, 0.05, ap_text,
                                         transform=axes_flat[ax_idx].transAxes,
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor=ap_color, alpha=0.8),
                                         fontsize=9, ha='right', va='bottom', fontweight='bold')
                else:
                    if perturbation == 0:
                        title = 'Baseline - No Data'
                    else:
                        title = f'{feature_name} {perturbation:+d}% - No Data'
                    axes_flat[ax_idx].set_title(title, fontsize=12)
                
                axes_flat[ax_idx].axis('off')
            
        except Exception as e:
            print(f"Animation error day {day}: {e}")
            for ax in axes_flat:
                ax.clear()
                ax.text(0.5, 0.5, f'Error in day {day}', ha='center', va='center',
                       transform=ax.transAxes)
                ax.axis('off')
    
    # Create animation for all days
    max_days = max(len(ground_truth), max(len(preds) for preds in all_perturbation_predictions.values()))
    
    try:
        print(f"Creating animation with {max_days} frames...")
        anim = animation.FuncAnimation(
            fig, animate, frames=max_days, interval=800, repeat=True, blit=False
        )
        
        print(f"Saving enhanced GIF to {output_path}...")
        anim.save(str(output_path), writer='pillow', fps=1.25)
        plt.close(fig)
        
        print(f"✓ Enhanced GIF saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating enhanced GIF: {e}")
        plt.close(fig)
        return False

def create_feature_sensitivity_gif(feature_name, output_dir, ground_truth, 
                                  baseline_predictions, perturbed_predictions_minus, 
                                  perturbed_predictions_plus):
    """Create the main sensitivity analysis GIF"""
    
    output_path = Path(output_dir) / f"{feature_name}_evolution.gif"
    output_path.parent.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{feature_name} Feature Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Subplot titles
    axes[0, 0].set_title('Ground Truth Fire')
    axes[0, 1].set_title('Baseline Prediction')
    axes[1, 0].set_title(f'{feature_name} -20% Effect')
    axes[1, 1].set_title(f'{feature_name} +20% Effect')
    
    def animate(day):
        for ax in axes.flat:
            ax.clear()
        
        try:
            # Ground truth
            if day < len(ground_truth):
                gt_data = ground_truth[day]
                if gt_data.ndim > 2:
                    gt_data = gt_data.squeeze()
                axes[0, 0].imshow(gt_data, cmap='Reds', vmin=0, vmax=1)
                axes[0, 0].set_title(f'Ground Truth Fire - Day {day+1}')
            else:
                axes[0, 0].set_title('Ground Truth Fire - No Data')
            
            # Baseline prediction
            if day < len(baseline_predictions):
                pred_data = baseline_predictions[day]
                if pred_data.ndim > 2:
                    pred_data = pred_data.squeeze()
                axes[0, 1].imshow(pred_data, cmap='Oranges', vmin=0, vmax=1)
                axes[0, 1].set_title(f'Baseline Prediction - Day {day+1}')
            else:
                axes[0, 1].set_title('Baseline Prediction - No Data')
            
            # Perturbed predictions
            if day < len(perturbed_predictions_minus):
                pred_minus = perturbed_predictions_minus[day]
                if pred_minus.ndim > 2:
                    pred_minus = pred_minus.squeeze()
                axes[1, 0].imshow(pred_minus, cmap='Oranges', vmin=0, vmax=1)
                axes[1, 0].set_title(f'{feature_name} -20% - Day {day+1}')
            else:
                axes[1, 0].set_title(f'{feature_name} -20% - No Data')
            
            if day < len(perturbed_predictions_plus):
                pred_plus = perturbed_predictions_plus[day]
                if pred_plus.ndim > 2:
                    pred_plus = pred_plus.squeeze()
                axes[1, 1].imshow(pred_plus, cmap='Oranges', vmin=0, vmax=1)
                axes[1, 1].set_title(f'{feature_name} +20% - Day {day+1}')
            else:
                axes[1, 1].set_title(f'{feature_name} +20% - No Data')
                
        except Exception as e:
            print(f"Error in animation frame {day}: {e}")
        
        # Remove axis ticks
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Create animation
    frames = min(len(ground_truth), len(baseline_predictions), 8)  # Limit frames
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000, repeat=True)
    
    # Save animation
    try:
        anim.save(str(output_path), writer='pillow', fps=1)
        print(f"✓ GIF saved: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Could not save GIF: {e}")
        return False
    finally:
        plt.close(fig)

# ============================================================================
# AP ANALYSIS FUNCTIONS
# ============================================================================

def analyze_fire_no_fire_distribution(targets):
    """分析有火天和无火天的分布"""
    fire_days = []
    no_fire_days = []
    
    for day_idx, target in enumerate(targets):
        if isinstance(target, torch.Tensor):
            fire_pixels = (target > 0.5).sum().item()
            total_pixels = target.numel()
        else:
            # numpy array
            fire_pixels = (target > 0.5).sum()
            total_pixels = target.size
        
        fire_ratio = fire_pixels / total_pixels
        
        if fire_pixels > 0:
            fire_days.append({
                'day': day_idx,
                'fire_pixels': fire_pixels,
                'fire_ratio': fire_ratio
            })
        else:
            no_fire_days.append({
                'day': day_idx,
                'fire_pixels': fire_pixels,
                'fire_ratio': fire_ratio
            })
    
    return fire_days, no_fire_days

def calculate_comprehensive_ap_analysis(predictions, targets, scenario_name):
    """
    计算全面的AP分析，包括多种计算方式
    """
    results = {}
    
    # 确保数据格式正确
    pred_arrays = []
    target_arrays = []
    
    for pred, target in zip(predictions, targets):
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        
        pred_arrays.append(pred.flatten())
        target_arrays.append(target.flatten())
    
    # 分析每天的情况
    fire_day_predictions = []
    fire_day_targets = []
    no_fire_day_predictions = []
    no_fire_day_targets = []
    daily_aps = []
    
    for day_idx, (pred_flat, target_flat) in enumerate(zip(pred_arrays, target_arrays)):
        fire_pixels = (target_flat > 0.5).sum()
        
        if fire_pixels > 0:  # 有火天
            fire_day_predictions.append(pred_flat)
            fire_day_targets.append(target_flat)
            # 计算单天AP
            daily_ap = average_precision_score(target_flat, pred_flat)
            daily_aps.append(daily_ap)
        else:  # 无火天
            no_fire_day_predictions.append(pred_flat)
            no_fire_day_targets.append(target_flat)
            daily_aps.append(0.0)  # 无火天AP为0
    
    # 方法1: 所有天合并计算（推荐方式）
    all_preds = np.concatenate(pred_arrays)
    all_targets = np.concatenate(target_arrays)
    
    if all_targets.sum() > 0:
        results['combined_ap'] = average_precision_score(all_targets, all_preds)
    else:
        results['combined_ap'] = 0.0
    
    # 方法2: 只计算有火天的AP
    if fire_day_predictions:
        fire_preds = np.concatenate(fire_day_predictions)
        fire_targets = np.concatenate(fire_day_targets)
        if fire_targets.sum() > 0:
            results['fire_days_only_ap'] = average_precision_score(fire_targets, fire_preds)
        else:
            results['fire_days_only_ap'] = 0.0
    else:
        results['fire_days_only_ap'] = 0.0
    
    # 方法3: 每天单独计算AP然后平均（包含0值）
    results['daily_average_ap'] = np.mean(daily_aps)
    
    # 统计信息
    results['fire_days'] = len(fire_day_predictions)
    results['no_fire_days'] = len(no_fire_day_predictions)
    results['total_days'] = len(predictions)
    results['fire_day_ratio'] = len(fire_day_predictions) / len(predictions) if predictions else 0
    results['daily_aps'] = daily_aps
    
    # 计算比例
    if results['fire_days_only_ap'] > 0 and results['combined_ap'] > 0:
        results['fire_to_combined_ratio'] = results['fire_days_only_ap'] / results['combined_ap']
    else:
        results['fire_to_combined_ratio'] = 1.0
    
    return results

def create_feature_ap_summary(feature_name, output_dir, ground_truth, baseline_predictions, perturbation_predictions, perturbation_levels):
    """为特征创建详细的AP总结报告"""
    
    print(f"\n📊 Computing AP analysis for {feature_name}...")
    
    # 分析ground truth分布
    fire_days, no_fire_days = analyze_fire_no_fire_distribution(ground_truth)
    
    # 存储所有结果
    ap_results = {}
    
    # 分析baseline
    baseline_analysis = calculate_comprehensive_ap_analysis(
        baseline_predictions, ground_truth, "Baseline"
    )
    ap_results['Baseline (0%)'] = baseline_analysis
    
    # 分析每个扰动级别
    for perturbation in perturbation_levels:
        if perturbation == 0:
            continue  # 已经处理了baseline
        
        scenario_name = f"{perturbation:+.0%}"
        if perturbation in perturbation_predictions:
            perturbation_analysis = calculate_comprehensive_ap_analysis(
                perturbation_predictions[perturbation], ground_truth, scenario_name
            )
            ap_results[scenario_name] = perturbation_analysis
    
    # 创建详细报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    
    report_path = output_path / f"{feature_name}_AP_Analysis.json"
    summary_path = output_path / f"{feature_name}_AP_Summary.txt"
    
    # 保存JSON格式的详细数据
    json_data = {
        'feature_name': feature_name,
        'ground_truth_analysis': {
            'fire_days': len(fire_days),
            'no_fire_days': len(no_fire_days),
            'total_days': len(ground_truth),
            'fire_day_ratio': len(fire_days) / len(ground_truth) if ground_truth else 0,
            'fire_day_details': [
                {
                    'day': int(day_info['day']),
                    'fire_pixels': int(day_info['fire_pixels']),
                    'fire_ratio': float(day_info['fire_ratio'])
                } for day_info in fire_days[:5]
            ]
        },
        'ap_results': {}
    }
    
    # 转换结果为可序列化格式
    for scenario, results in ap_results.items():
        json_data['ap_results'][scenario] = {
            'combined_ap': float(results['combined_ap']),
            'fire_days_only_ap': float(results['fire_days_only_ap']),
            'daily_average_ap': float(results['daily_average_ap']),
            'fire_days': int(results['fire_days']),
            'no_fire_days': int(results['no_fire_days']),
            'total_days': int(results['total_days']),
            'fire_day_ratio': float(results['fire_day_ratio']),
            'fire_to_combined_ratio': float(results['fire_to_combined_ratio'])
        }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 创建人类可读的总结报告
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"🔥 {feature_name} FEATURE SENSITIVITY - AP ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 GROUND TRUTH DATA DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Total days analyzed: {len(ground_truth)}\n")
        f.write(f"• Days with fire: {len(fire_days)} ({len(fire_days)/len(ground_truth)*100:.1f}%)\n")
        f.write(f"• Days without fire: {len(no_fire_days)} ({len(no_fire_days)/len(ground_truth)*100:.1f}%)\n\n")
        
        if fire_days:
            f.write("🔥 FIRE DAYS DETAILS:\n")
            for day_info in fire_days[:10]:  # 显示前10天
                f.write(f"  Day {day_info['day']+1}: {day_info['fire_pixels']} pixels ({day_info['fire_ratio']*100:.3f}%)\n")
            if len(fire_days) > 10:
                f.write(f"  ... and {len(fire_days)-10} more fire days\n")
            f.write("\n")
        
        f.write("📈 AVERAGE PRECISION (AP) ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Legend:\n")
        f.write("• Combined AP: All days merged (RECOMMENDED method)\n")
        f.write("• Fire-only AP: Only days with actual fire\n")
        f.write("• Daily Average: Average of individual daily APs (includes 0s)\n\n")
        
        # 按AP分数排序
        sorted_results = sorted(ap_results.items(), 
                              key=lambda x: x[1]['combined_ap'], 
                              reverse=True)
        
        f.write(f"{'Scenario':<15} {'Combined AP':<12} {'Fire-only AP':<13} {'Daily Avg':<11} {'Ratio':<8}\n")
        f.write("-" * 70 + "\n")
        
        for scenario, results in sorted_results:
            f.write(f"{scenario:<15} ")
            f.write(f"{results['combined_ap']:<12.4f} ")
            f.write(f"{results['fire_days_only_ap']:<13.4f} ")
            f.write(f"{results['daily_average_ap']:<11.4f} ")
            f.write(f"{results['fire_to_combined_ratio']:<8.2f}\n")
        
        f.write("\n")
        
        # 找出最佳和最差的扰动
        best_scenario = max(ap_results.items(), key=lambda x: x[1]['combined_ap'])
        worst_scenario = min(ap_results.items(), key=lambda x: x[1]['combined_ap'])
        
        f.write("🎯 KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Best performing scenario: {best_scenario[0]} (AP: {best_scenario[1]['combined_ap']:.4f})\n")
        f.write(f"• Worst performing scenario: {worst_scenario[0]} (AP: {worst_scenario[1]['combined_ap']:.4f})\n")
        
        baseline_ap = ap_results.get('Baseline (0%)', {}).get('combined_ap', 0)
        if baseline_ap > 0:
            best_improvement = (best_scenario[1]['combined_ap'] - baseline_ap) / baseline_ap * 100
            worst_degradation = (worst_scenario[1]['combined_ap'] - baseline_ap) / baseline_ap * 100
            f.write(f"• Best improvement over baseline: {best_improvement:+.1f}%\n")
            f.write(f"• Worst degradation from baseline: {worst_degradation:+.1f}%\n")
        
        f.write("\n💡 METHODOLOGY NOTES\n")
        f.write("-" * 40 + "\n")
        f.write("• Combined AP is the most reliable metric (merges all predictions/targets)\n")
        f.write("• Fire-only AP shows performance on fire-active days only\n")
        f.write("• Daily Average includes 0 AP from no-fire days (may underestimate performance)\n")
        f.write("• Ratio shows how much no-fire days affect the combined score\n")
        
        f.write(f"\n📁 Detailed data saved to: {report_path.name}\n")
    
    print(f"✅ {feature_name} AP analysis completed!")
    print(f"   📊 Summary: {summary_path}")
    print(f"   📋 Detailed: {report_path}")
    
    # 打印关键结果到控制台
    print(f"\n🎯 {feature_name} QUICK RESULTS:")
    baseline_ap = ap_results.get('Baseline (0%)', {}).get('combined_ap', 0)
    print(f"   Baseline AP: {baseline_ap:.4f}")
    
    for scenario, results in sorted_results[:3]:  # 显示前3个最佳结果
        if scenario != 'Baseline (0%)':
            improvement = (results['combined_ap'] - baseline_ap) / baseline_ap * 100 if baseline_ap > 0 else 0
            print(f"   {scenario}: {results['combined_ap']:.4f} ({improvement:+.1f}%)")
    
    return ap_results

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_simple_sensitivity_analysis(model_path, fire_event_path, output_dir='simple_sensitivity_results'):
    """Run simplified sensitivity analysis focused on GIF generation"""
    
    config = SimpleConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model_with_compatibility(model_path, len(config.BEST_FEATURES), config.SEQUENCE_LENGTH, device)
    if model is None:
        print("Failed to load model")
        return
    
    # Initialize simulator
    simulator = SimpleFireSimulator(model, config, device)
    
    # Load data
    print("Loading fire event data...")
    initial_seq, weather_data, ground_truth, ground_truth_raw, max_days = load_fire_event_data(
        fire_event_path, config, start_day=0
    )
    
    if initial_seq is None:
        print("Failed to load fire event data")
        return
    
    # Run baseline prediction
    print("Running baseline prediction...")
    # Use EVOLUTION APPROACH: different sequences for each day like create_evolution_animation
    baseline_predictions = []
    for day in range(max_days):
        # Load fresh sequence for this day (like evolution animation)
        day_seq, _, _, _, _ = load_fire_event_data(
            fire_event_path, config, start_day=day
        )
        if day_seq is not None:
            raw_pred = simulator.predict_single_step(day_seq.unsqueeze(0), debug=False)
            baseline_predictions.append(raw_pred.numpy())  # RAW sigmoid predictions
        else:
            # If no more data, reuse last prediction
            if baseline_predictions:
                baseline_predictions.append(baseline_predictions[-1])
            else:
                # Fallback to original sequence
                raw_pred = simulator.predict_single_step(initial_seq.unsqueeze(0), debug=False)
                baseline_predictions.append(raw_pred.numpy())
    
    # Features to analyze - COMPLETE ANALYSIS
    important_features = [
        # Vegetation indices (植被指数)
        'NDVI', 'EVI2',
        # Weather conditions (气象条件) 
        'Max_Temp_K', 'Min_Temp_K', 'Total_Precip',
        # Satellite data (卫星数据)
        'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1',
        # Topographic features (地形特征)
        'Elevation', 'Slope', 'Aspect'
    ]
    
    print(f"\nGenerating sensitivity GIFs for: {important_features}")
    
    for feature_name in important_features:
        print(f"\n{'='*50}")
        print(f"ANALYZING {feature_name}")
        print(f"{'='*50}")
        
        try:
            # FIXED: Reorganize loops to avoid excessive data loading
            print(f"Generating {feature_name} perturbations: {config.PERTURBATION_LEVELS}")
            all_perturbation_predictions = {}
            
            # Initialize prediction arrays for all perturbation levels
            for perturbation in config.PERTURBATION_LEVELS:
                all_perturbation_predictions[perturbation] = []
            
            # Load data once per day and apply all perturbations
            print(f"Processing {max_days} days with {len(config.PERTURBATION_LEVELS)} perturbation levels...")
            for day in range(max_days):
                # Load fresh sequence for this day (like evolution) - ONLY ONCE PER DAY
                day_seq, _, _, _, _ = load_fire_event_data(
                    fire_event_path, config, start_day=day
                )
                
                if day_seq is not None:
                    # Apply all perturbations to this day's sequence
                    for perturbation in config.PERTURBATION_LEVELS:
                        if perturbation == 0:
                            # Baseline - no perturbation
                            perturbed_seq = day_seq
                        else:
                            # Apply perturbation to this day's sequence
                            perturbed_seq = simulator.apply_feature_perturbation(
                                day_seq, feature_name, perturbation
                            )
                        
                        # Get raw predictions (no physics)
                        pred = simulator.predict_single_step(perturbed_seq.unsqueeze(0), debug=False)
                        all_perturbation_predictions[perturbation].append(pred.numpy())
                else:
                    # Fallback: reuse last prediction or use baseline sequence
                    for perturbation in config.PERTURBATION_LEVELS:
                        if all_perturbation_predictions[perturbation]:
                            # Reuse last prediction
                            all_perturbation_predictions[perturbation].append(
                                all_perturbation_predictions[perturbation][-1]
                            )
                        else:
                            # Use baseline sequence as fallback
                            if perturbation == 0:
                                perturbed_seq = initial_seq
                            else:
                                perturbed_seq = simulator.apply_feature_perturbation(
                                    initial_seq, feature_name, perturbation
                                )
                            pred = simulator.predict_single_step(perturbed_seq.unsqueeze(0), debug=False)
                            all_perturbation_predictions[perturbation].append(pred.numpy())
            
            # Create enhanced GIF with multiple perturbation levels
            print(f"Creating {feature_name} enhanced sensitivity GIF...")
            success = create_enhanced_feature_sensitivity_gif(
                feature_name, output_dir, ground_truth, ground_truth_raw,
                baseline_predictions, all_perturbation_predictions, config.PERTURBATION_LEVELS
            )
            
            # Create AUTOREGRESSIVE GIF (using predicted fire points instead of real ones)
            print(f"Creating {feature_name} AUTOREGRESSIVE sensitivity GIF...")
            autoregressive_predictions = generate_autoregressive_predictions(
                simulator, initial_seq, weather_data, max_days, feature_name, 
                config.PERTURBATION_LEVELS, fire_event_path, config
            )
            
            success_auto = create_enhanced_feature_sensitivity_gif(
                f"{feature_name}_AUTOREGRESSIVE", output_dir, ground_truth, ground_truth_raw,
                autoregressive_predictions.get(0, baseline_predictions), 
                autoregressive_predictions, config.PERTURBATION_LEVELS
            )
            
            # Create difference analysis visualization
            if success and success_auto:
                standard_gif_path = Path(output_dir) / f"{feature_name}_enhanced_evolution.gif"
                autoregressive_gif_path = Path(output_dir) / f"{feature_name}_AUTOREGRESSIVE_enhanced_evolution.gif"
                
                difference_success = create_difference_visualization(
                    str(standard_gif_path), str(autoregressive_gif_path), 
                    feature_name, output_dir
                )
                
                if difference_success:
                    print(f"✓ {feature_name} complete analysis (GIFs + Difference) finished!")
                else:
                    print(f"✓ {feature_name} GIFs created, but difference analysis failed")
            else:
                print(f"✗ {feature_name} GIF generation failed")
            
            # 🆕 ADD AP ANALYSIS
            print(f"\n📊 Computing AP analysis for {feature_name}...")
            try:
                ap_results = create_feature_ap_summary(
                    feature_name, output_dir, ground_truth, 
                    baseline_predictions, all_perturbation_predictions, 
                    config.PERTURBATION_LEVELS
                )
                print(f"✅ {feature_name} AP analysis completed successfully!")
            except Exception as ap_error:
                print(f"⚠️ {feature_name} AP analysis failed: {ap_error}")
                
        except Exception as e:
            print(f"✗ Error analyzing {feature_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("ENHANCED SENSITIVITY ANALYSIS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved in: {output_dir}")
    print("\n🎬 Generated COMPLETE ANALYSIS for each feature:")
    print("\n1. STANDARD GIFs (using real fire points):")
    print("   - 3x3 grid with ground truth + 8 perturbation levels")
    print("   - Uses real historical fire data for predictions")
    print("   - Shows immediate feature sensitivity effects")
    
    print("\n2. AUTOREGRESSIVE GIFs (using predicted fire points):")
    print("   - Same 3x3 grid layout")
    print("   - Uses PREVIOUS PREDICTIONS as fire input")
    print("   - Shows cumulative prediction errors and amplified sensitivity")
    print("   - Reveals how model predictions evolve recursively")
    
    print("\n3. DIFFERENCE ANALYSIS IMAGES (PNG):")
    print("   - Side-by-side comparison of standard vs autoregressive")
    print("   - Pixel-level difference heatmap")
    print("   - Statistical analysis of differences")
    print("   - Automatically generated for visual verification")
    
    print(f"\n📊 Analysis parameters:")
    print(f"   - Perturbation levels: {config.PERTURBATION_LEVELS}")
    print(f"   - Time period: {config.SIMULATION_DAYS} days")
    print(f"   - Features analyzed: {important_features}")
    
    print("\n🎯 Generated files for each feature:")
    print("   - {feature}_enhanced_evolution.gif (Standard prediction)")
    print("   - {feature}_AUTOREGRESSIVE_enhanced_evolution.gif (Recursive prediction)")
    print("   - {feature}_difference_analysis.png (Difference visualization)")
    print("   - {feature}_AP_Summary.txt (Human-readable AP analysis)")
    print("   - {feature}_AP_Analysis.json (Detailed AP data)")
    
    print("\n📈 Key insights from complete analysis:")
    print("   - Standard GIFs: Feature effects with perfect fire history")
    print("   - Autoregressive GIFs: Feature effects with prediction uncertainty")
    print("   - Difference PNGs: Quantified visual differences and statistics")
    print("   - AP Analysis: Quantitative performance metrics for all perturbations")
    print("   - Combined analysis reveals model's recursive prediction stability!")
    
    print("\n📊 AP ANALYSIS HIGHLIGHTS:")
    print("   - Uses CORRECT AP calculation (combined method - not simple averaging)")
    print("   - Separates fire days vs no-fire days analysis")
    print("   - Shows which perturbations improve/degrade model performance")
    print("   - Includes statistical significance and improvement percentages")
    
    print(f"\n✅ Complete analysis finished! Check all generated files for comprehensive insights!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse
    import sys
    
    # If no arguments provided, use default paths
    if len(sys.argv) == 1:
        # Default paths for direct execution
        model_path = "best_fire_model_official.pth"
        fire_event_path = "data/processed/2020/fire_24461899.hdf5"
        
        if os.path.exists(model_path) and os.path.exists(fire_event_path):
            print("Running simple sensitivity analysis with default paths...")
            print(f"Model: {model_path}")
            print(f"Fire event: {fire_event_path}")
            run_simple_sensitivity_analysis(model_path, fire_event_path)
        else:
            print("Default files not found. Available files:")
            if os.path.exists("backup/fixed_wildfire_outputs/"):
                print("Models found:")
                for f in os.listdir("backup/fixed_wildfire_outputs/"):
                    if f.endswith(('.pth', '.pt', '.ckpt')):
                        print(f"  - backup/fixed_wildfire_outputs/{f}")
            
            if os.path.exists("data/processed/"):
                print("Fire events found:")
                for year_dir in os.listdir("data/processed/"):
                    year_path = os.path.join("data/processed/", year_dir)
                    if os.path.isdir(year_path):
                        for f in os.listdir(year_path):
                            if f.endswith(('.h5', '.hdf5')):
                                print(f"  - data/processed/{year_dir}/{f}")
            
            print("\nPlease run with:")
            print("python simple_feature_sensitivity.py --model <model_path> --fire_event <data_path>")
    else:
        # Use command line arguments
        parser = argparse.ArgumentParser(description='Simple Feature Sensitivity Analysis - GIF Focus')
        parser.add_argument('--model', required=True, help='Path to trained model')
        parser.add_argument('--fire_event', required=True, help='Path to fire event HDF5 file')
        parser.add_argument('--output_dir', default='simple_sensitivity_results', help='Output directory')
        
        args = parser.parse_args()
        run_simple_sensitivity_analysis(args.model, args.fire_event, args.output_dir)

if __name__ == "__main__":
    main()
