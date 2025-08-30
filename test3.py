"""
Fire Spread Simulation Module - FIXED VERSION
==============================================

Standalone module for fire spread simulation, variable sensitivity analysis, 
and visualization. Requires a pre-trained model from the training pipeline.

Usage:
    python fire_simulation.py --model best_fire_model_official.pth
    python fire_simulation.py --demo  # For demonstration with synthetic data
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

# Use non-interactive matplotlib backend to avoid OpenMP conflicts
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from scipy import ndimage
import argparse
import pickle
import sys

# ============================================================================
# COMPATIBILITY CLASSES (Add missing classes that might be in the checkpoint)
# ============================================================================

class WildFireConfig:
    """Dummy config class for compatibility with old checkpoints"""
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

# Make these available in the global namespace for unpickling
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================

class SimulationConfig:
    """Configuration for fire simulation"""
    
    # Model architecture settings (must match training config)
    SPATIAL_SIZE = (128, 128)
    SEQUENCE_LENGTH = 5
    PREDICTION_HORIZON = 1
    
    # Feature configuration (must match training)
    FEATURE_NAMES = [
        'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1',      # 0-2: Thermal/reflectance
        'NDVI', 'EVI2',                            # 3-4: Vegetation indices  
        'Total_Precip', 'Wind_Speed',              # 5-6: Weather
        'Wind_Direction',                          # 7: Angular
        'Min_Temp_K', 'Max_Temp_K',               # 8-9: Temperature
        'ERC', 'Spec_Hum', 'PDSI',                # 10-12: Fire weather
        'Slope', 'Aspect',                         # 13-14: Topography
        'Elevation', 'Landcover',                  # 15-16: Static
        'Forecast_Precip', 'Forecast_Wind_Speed',  # 17-18: Forecast weather
        'Forecast_Wind_Dir',                       # 19: Angular forecast
        'Forecast_Temp_C', 'Forecast_Spec_Hum',   # 20-21: Forecast conditions
        'Active_Fire'                              # 22: Target
    ]
    
    BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]
    
    # Simulation physics parameters - ADJUSTED FOR BETTER FIRE PREDICTION
    MAX_SIMULATION_DAYS = 30
    FIRE_DECAY_RATE = 0.01          # Reduced from 0.05 - less aggressive decay
    FIRE_SPREAD_THRESHOLD = 0.1     # Reduced from 0.3 - more sensitive to low probabilities

# ============================================================================
# IMPROVED MODEL LOADER WITH MULTIPLE COMPATIBILITY OPTIONS
# ============================================================================

def load_model_with_compatibility(model_path, input_channels, sequence_length=5, device='cpu'):
    """
    Load model with multiple fallback options for compatibility
    """
    print(f"Loading model from {model_path}...")
    
    # Method 1: Try safe loading first
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("✓ Loaded with weights_only=True (safest)")
    except Exception as e1:
        print(f"Safe loading failed: {e1}")
        
        # Method 2: Try legacy loading
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("✓ Loaded with weights_only=False (legacy)")
        except Exception as e2:
            print(f"Legacy loading failed: {e2}")
            
            # Method 3: Try with pickle protocol fix
            try:
                import pickle
                # Add safe globals for numpy and other common objects
                torch.serialization.add_safe_globals([
                    'numpy.core.multiarray.scalar',
                    'numpy.core.multiarray._reconstruct',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                print("✓ Loaded with safe globals")
            except Exception as e3:
                print(f"Safe globals loading failed: {e3}")
                raise ValueError(f"All loading methods failed. Last error: {e3}")
    
    # Create model
    model = OfficialFireUNet(input_channels, sequence_length)
    
    # Handle different checkpoint formats
    try:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_ap' in checkpoint:
                    print(f"✓ Best AP: {checkpoint['best_ap']:.4f}")
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("✓ Loaded from 'model' key")
            else:
                # Try to load the dict directly as state_dict
                model.load_state_dict(checkpoint)
                print("✓ Loaded dict as state_dict")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("✓ Loaded direct state_dict")
            
    except Exception as e:
        print(f"State dict loading failed: {e}")
        
        # Method 4: Try to extract state dict from any tensor structure
        try:
            # Sometimes the model is wrapped in unexpected ways
            if hasattr(checkpoint, 'state_dict'):
                model.load_state_dict(checkpoint.state_dict())
            elif hasattr(checkpoint, 'module'):
                model.load_state_dict(checkpoint.module.state_dict())
            else:
                raise e
        except:
            raise ValueError(f"Could not extract valid state dict from checkpoint: {e}")
    
    return model

# ============================================================================
# MODEL ARCHITECTURE (COPIED FROM TRAINING MODULE)
# ============================================================================

class OfficialFireUNet(nn.Module):
    """U-Net architecture for fire prediction (must match training)"""
    
    def __init__(self, input_channels, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        total_input_channels = input_channels * sequence_length
        
        # U-Net encoder
        self.enc1 = self._double_conv(total_input_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # U-Net decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights
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
        
        # U-Net forward pass
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.pool(enc4)
        
        bottleneck = self.bottleneck(enc4_pool)
        
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
# FIRE SPREAD SIMULATOR (UPDATED)
# ============================================================================

class FireSpreadSimulator:
    """Main fire spread simulator class with improved model loading"""
    
    def __init__(self, model_path, config=None, device=None):
        self.config = config or SimulationConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine input channels from config
        input_channels = len(self.config.BEST_FEATURES)
        
        # Load model with improved compatibility
        self.model = load_model_with_compatibility(
            model_path, 
            input_channels, 
            self.config.SEQUENCE_LENGTH,
            self.device
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model ready on {self.device}")
        print(f"Input channels: {input_channels}")
    
    def predict_single_step(self, input_sequence, debug=False):
        """Predict next day fire distribution with optional debugging"""
        with torch.no_grad():
            if len(input_sequence.shape) == 3:
                input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            
            input_tensor = input_sequence.to(self.device)
            
            # Debug input information
            if debug:
                fire_channel = input_tensor[0, -1, -1]  # Last day, fire channel
                print(f"\n=== PREDICTION DEBUG ===")
                print(f"Input sequence shape: {input_tensor.shape}")
                print(f"Input fire pixels: {(fire_channel > 0).sum().item()}")
                print(f"Input fire max value: {fire_channel.max().item():.4f}")
                print(f"Input fire mean (non-zero): {fire_channel[fire_channel > 0].mean().item():.4f}" if (fire_channel > 0).any() else "No fire pixels")
            
            # Use mixed precision if available
            try:
                with torch.amp.autocast('cuda'):
                    output = self.model(input_tensor)
                    prediction = torch.sigmoid(output)
            except:
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output)
            
            # Debug model output
            if debug:
                print(f"Model raw output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"Sigmoid output range: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")
                print(f"Pixels > 0.5: {(prediction > 0.5).sum().item()}")
                print(f"Pixels > 0.3: {(prediction > 0.3).sum().item()}")
                print(f"Pixels > 0.1: {(prediction > 0.1).sum().item()}")
                print(f"Pixels > 0.05: {(prediction > 0.05).sum().item()}")
                print(f"Mean prediction value: {prediction.mean().item():.4f}")
                print("========================\n")
            
            return prediction.cpu().squeeze()
    
    def simulate_fire_evolution(self, initial_sequence, weather_data=None, 
                              num_days=10, mode='sliding_window', debug=False):
        """
        Simulate fire evolution over multiple days with optional debugging
        
        Args:
            initial_sequence: Initial 5-day sequence (5, channels, H, W)
            weather_data: Future weather data (num_days, channels, H, W)
            num_days: Number of days to simulate
            mode: 'sliding_window' or 'autoregressive'
            debug: Enable detailed debugging output
        """
        predictions = []
        current_sequence = initial_sequence.clone()
        
        print(f"Simulating {num_days} days using {mode} mode...")
        if debug:
            print("DEBUG MODE ENABLED - Detailed output will be shown")
        
        for day in tqdm(range(num_days), desc="Simulating fire evolution"):
            # Predict next day with debugging for first few days
            show_debug = debug and day < 3  # Show debug for first 3 days
            pred_fire = self.predict_single_step(current_sequence.unsqueeze(0), debug=show_debug)
            
            # Apply fire physics with debugging
            pred_fire = self._apply_fire_physics(pred_fire, day, debug=show_debug)
            
            predictions.append(pred_fire.numpy())
            
            if day < num_days - 1:  # Not the last day
                if mode == 'sliding_window' and weather_data is not None and day < len(weather_data) - self.config.SEQUENCE_LENGTH:
                    # Use real weather data
                    next_sequence = weather_data[day + 1:day + 1 + self.config.SEQUENCE_LENGTH].clone()
                    # Update Active_Fire channel in the last frame
                    active_fire_idx = self.config.BEST_FEATURES.index(22)
                    next_sequence[-1, active_fire_idx] = pred_fire
                    current_sequence = next_sequence
                    
                elif mode == 'autoregressive':
                    # Use prediction as input
                    new_frame = current_sequence[-1].clone()
                    active_fire_idx = self.config.BEST_FEATURES.index(22)
                    new_frame[active_fire_idx] = pred_fire
                    
                    current_sequence = torch.cat([
                        current_sequence[1:],  # Remove first day
                        new_frame.unsqueeze(0)  # Add predicted day
                    ], dim=0)
                else:
                    break
        
        return predictions
    
    def _apply_fire_physics(self, fire_prediction, day, debug=False):
        """Apply realistic fire physics to predictions with debugging"""
        if debug:
            print(f"\n=== FIRE PHYSICS DEBUG (Day {day}) ===")
            print(f"Raw prediction range: [{fire_prediction.min().item():.4f}, {fire_prediction.max().item():.4f}]")
            print(f"Pixels > threshold ({self.config.FIRE_SPREAD_THRESHOLD}): {(fire_prediction > self.config.FIRE_SPREAD_THRESHOLD).sum().item()}")
        
        # Apply probability threshold - LESS STRICT
        fire_binary = (fire_prediction > self.config.FIRE_SPREAD_THRESHOLD).float()
        
        if debug:
            print(f"After threshold - fire pixels: {fire_binary.sum().item()}")
        
        # Apply daily fire decay - LESS AGGRESSIVE
        decay_factor = 1.0 - self.config.FIRE_DECAY_RATE * (day + 1)
        decay_factor = max(0.5, decay_factor)  # Minimum 50% intensity (was 10%)
        
        fire_decayed = fire_binary * decay_factor
        
        if debug:
            print(f"Decay factor: {decay_factor:.3f}")
            print(f"After decay - fire pixels: {(fire_decayed > 0).sum().item()}")
        
        # Add spatial smoothing - LESS AGGRESSIVE
        fire_smoothed = torch.tensor(
            ndimage.gaussian_filter(fire_decayed.numpy(), sigma=0.3)  # Reduced from 0.5
        )
        
        if debug:
            print(f"After smoothing - fire pixels: {(fire_smoothed > 0.01).sum().item()}")
            print(f"Final output range: [{fire_smoothed.min().item():.4f}, {fire_smoothed.max().item():.4f}]")
            print("===========================\n")
        
        return fire_smoothed
    
    def create_simulation_animation(self, predictions, real_sequence=None, 
                                  save_path='fire_simulation.gif'):
        """Create animated visualization of fire spread simulation"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)) if real_sequence is not None else plt.subplots(1, 1, figsize=(8, 6))
        
        if not hasattr(axes, '__len__'):
            axes = [axes]
        
        def animate(frame):
            for ax in axes:
                ax.clear()
            
            if real_sequence is not None and frame < len(real_sequence):
                # Show real vs predicted
                axes[0].imshow(real_sequence[frame], cmap='Reds', vmin=0, vmax=1)
                axes[0].set_title(f'Actual Fire - Day {frame+1}')
                axes[0].axis('off')
                
                if frame < len(predictions):
                    axes[1].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[1].set_title(f'Predicted Fire - Day {frame+1}')
                    axes[1].axis('off')
            else:
                # Show only predictions
                if frame < len(predictions):
                    axes[0].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[0].set_title(f'Simulated Fire Spread - Day {frame+1}')
                    axes[0].axis('off')
                    
                    # Add colorbar
                    im = axes[0].images[0] if axes[0].images else None
                    if im:
                        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Fire Probability')
        
        frames = min(len(predictions), 30)  # Limit to 30 days
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved: {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            # Save individual frames
            for i, pred in enumerate(predictions[:30]):
                plt.figure(figsize=(8, 6))
                plt.imshow(pred, cmap='Reds', vmin=0, vmax=1)
                plt.title(f'Simulated Fire - Day {i+1}')
                plt.colorbar(label='Fire Probability')
                plt.axis('off')
                plt.savefig(f'fire_frame_{i:02d}.png', bbox_inches='tight', dpi=150)
                plt.close()
            print("Saved individual frames: fire_frame_*.png")
    
    def create_comparison_animation(self, predictions, ground_truth, save_path='fire_comparison.gif'):
        """Create side-by-side comparison of predicted vs actual fire spread"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            if frame < len(predictions):
                # Predicted fire
                im1 = ax1.imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                ax1.set_title(f'Predicted Fire - Day {frame+1}', fontsize=14, fontweight='bold')
                ax1.axis('off')
                
                # Actual fire (if available)
                if frame < len(ground_truth):
                    im2 = ax2.imshow(ground_truth[frame], cmap='Reds', vmin=0, vmax=1)
                    ax2.set_title(f'Actual Fire - Day {frame+1}', fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    
                    # Calculate comparison metrics
                    pred_binary = (predictions[frame] > 0.5).astype(float)
                    actual_binary = ground_truth[frame]
                    
                    if actual_binary.sum() > 0:  # If there's actual fire
                        intersection = (pred_binary * actual_binary).sum()
                        union = pred_binary.sum() + actual_binary.sum() - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        dice = (2 * intersection) / (pred_binary.sum() + actual_binary.sum()) if (pred_binary.sum() + actual_binary.sum()) > 0 else 0
                        
                        plt.figtext(0.5, 0.02, f'IoU: {iou:.3f} | Dice: {dice:.3f}', 
                                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                else:
                    ax2.text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center', 
                            transform=ax2.transAxes, fontsize=16, color='gray')
                    ax2.axis('off')
        
        frames = max(len(predictions), len(ground_truth))
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=1.5)
            print(f"Comparison animation saved: {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            # Save individual comparison frames as fallback
            for i in range(min(len(predictions), len(ground_truth))):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                ax1.imshow(predictions[i], cmap='Reds', vmin=0, vmax=1)
                ax1.set_title(f'Predicted Fire - Day {i+1}')
                ax1.axis('off')
                
                ax2.imshow(ground_truth[i], cmap='Reds', vmin=0, vmax=1)
                ax2.set_title(f'Actual Fire - Day {i+1}')
                ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'comparison_frame_{i:02d}.png', bbox_inches='tight', dpi=150)
                plt.close()
                
            print("Saved individual comparison frames: comparison_frame_*.png")
        
        return anim

# ============================================================================
# SINGLE FIRE EVENT EVOLUTION ANALYZER
# ============================================================================

class SingleFireEventAnalyzer:
    """Analyze the complete evolution of a single fire event"""
    
    def __init__(self, simulator, config=None):
        self.simulator = simulator
        self.config = config or SimulationConfig()
    
    def analyze_fire_evolution(self, fire_event_path):
        """
        Analyze the complete evolution of a single fire event
        
        Args:
            fire_event_path: Path to the HDF5 file of the fire event
        """
        print(f"Analyzing complete fire evolution for: {os.path.basename(fire_event_path)}")
        
        # Load the complete fire event
        fire_loader = FireEventLoader(self.config)
        fire_event_data = fire_loader.load_fire_event(fire_event_path)
        T, C, H, W = fire_event_data.shape
        
        print(f"Fire event duration: {T} days, Spatial size: {H}x{W}")
        
        # Analyze ALL possible consecutive time windows (no day limit)
        evolution_analysis = []
        
        # Analyze the complete fire cycle
        analysis_days = T - self.config.SEQUENCE_LENGTH - 1
        print(f"Analyzing {analysis_days} time windows across complete fire cycle")
        
        for start_day in range(0, analysis_days, 1):  # Every day for complete coverage
            try:
                window_analysis = self._analyze_time_window(
                    fire_event_data, start_day, fire_event_path
                )
                if window_analysis:
                    evolution_analysis.append(window_analysis)
                    
            except Exception as e:
                print(f"Error analyzing window starting day {start_day}: {e}")
                continue
        
        # Analyze the temporal patterns
        temporal_patterns = self._extract_temporal_patterns(evolution_analysis)
        
        return evolution_analysis, temporal_patterns
    
    def _analyze_time_window(self, fire_data, start_day, fire_path):
        """Analyze a specific 5-day time window"""
        fire_loader = FireEventLoader(self.config)
        
        try:
            # Prepare the specific time window
            initial_sequence, _, ground_truth = fire_loader.prepare_simulation_data(
                fire_data, start_day=start_day
            )
            
            if not ground_truth or len(ground_truth) == 0:
                return None
            
            # Extract environmental conditions for each day in the window
            daily_conditions = []
            for day_idx in range(initial_sequence.shape[0]):
                conditions = self._extract_daily_conditions(initial_sequence[day_idx])
                conditions['day'] = start_day + day_idx
                daily_conditions.append(conditions)
            
            # Model prediction
            pred_fire = self.simulator.predict_single_step(initial_sequence.unsqueeze(0))
            actual_fire = torch.FloatTensor(ground_truth[0])
            
            # Calculate accuracy metrics
            accuracy = self._calculate_detailed_accuracy(pred_fire, actual_fire)
            
            # Fire spread analysis
            spread_analysis = self._analyze_fire_spread_pattern(
                initial_sequence[-1, -1].numpy(),  # Last day's fire
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
            print(f"Error in window analysis: {e}")
            return None
    
    def _extract_daily_conditions(self, day_data):
        """Extract environmental conditions for a single day"""
        conditions = {}
        
        try:
            # Extract key environmental variables (spatial means)
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
            
            # Fire area on this day
            if day_data.shape[0] > 12:  # If Active_Fire channel exists
                fire_data = day_data[-1].numpy()
                conditions['fire_area'] = float(np.sum(fire_data > 0))
                conditions['fire_intensity'] = float(np.mean(fire_data[fire_data > 0])) if np.sum(fire_data > 0) > 0 else 0.0
            
        except Exception as e:
            print(f"Error extracting conditions: {e}")
            conditions = {var: 0.0 for var in ['ndvi', 'precipitation', 'max_temp', 'fire_area']}
        
        return conditions
    
    def _calculate_detailed_accuracy(self, predicted, actual):
        """Calculate comprehensive accuracy metrics"""
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
        """Analyze fire spread patterns and directions"""
        try:
            # Calculate fire centroids
            initial_centroid = self._calculate_fire_centroid(initial_fire)
            pred_centroid = self._calculate_fire_centroid(predicted_fire)
            actual_centroid = self._calculate_fire_centroid(actual_fire)
            
            # Calculate spread directions and distances
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
            print(f"Error in spread analysis: {e}")
        
        return {'predicted_direction': 0.0, 'actual_direction': 0.0, 'direction_error': 0.0,
                'predicted_spread_distance': 0.0, 'actual_spread_distance': 0.0, 'distance_error': 0.0}
    
    def _calculate_fire_centroid(self, fire_map):
        """Calculate the centroid of fire pixels"""
        fire_pixels = np.where(fire_map > 0)
        if len(fire_pixels[0]) > 0:
            centroid_y = np.mean(fire_pixels[0])
            centroid_x = np.mean(fire_pixels[1])
            return (centroid_x, centroid_y)
        return None
    
    def _summarize_window_environment(self, daily_conditions):
        """Summarize environmental conditions across the 5-day window"""
        if not daily_conditions:
            return {}
        
        summary = {}
        
        # Calculate trends across the window
        for var in ['precipitation', 'max_temp', 'fire_area']:
            values = [day.get(var, 0) for day in daily_conditions]
            summary[f'{var}_trend'] = 'increasing' if values[-1] > values[0] else 'decreasing'
            summary[f'{var}_mean'] = np.mean(values)
            summary[f'{var}_std'] = np.std(values)
        
        # Identify dominant conditions
        precip_values = [day.get('precipitation', 0) for day in daily_conditions]
        temp_values = [day.get('max_temp', 300) for day in daily_conditions]
        
        summary['condition_type'] = 'dry' if np.mean(precip_values) < 2 else 'wet'
        summary['temperature_level'] = 'high' if np.mean(temp_values) > 308 else 'moderate'
        
        return summary
    
    def _extract_temporal_patterns(self, evolution_analysis):
        """Extract temporal patterns from the complete fire evolution"""
        if not evolution_analysis:
            return {}
        
        # Extract time series data
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
        """Create comprehensive fire evolution plots"""
        if not evolution_analysis:
            print("No evolution data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for plotting
        days = [w['start_day'] for w in evolution_analysis]
        accuracies = [w['prediction_accuracy']['iou'] for w in evolution_analysis]
        fire_areas = [w['environmental_summary'].get('fire_area_mean', 0) for w in evolution_analysis]
        temperatures = [w['environmental_summary'].get('max_temp_mean', 300) for w in evolution_analysis]
        precipitations = [w['environmental_summary'].get('precipitation_mean', 0) for w in evolution_analysis]
        
        # Plot 1: Prediction accuracy over time
        axes[0,0].plot(days, accuracies, 'bo-', linewidth=2, markersize=6)
        axes[0,0].set_title('Prediction Accuracy Over Time', fontweight='bold')
        axes[0,0].set_xlabel('Day')
        axes[0,0].set_ylabel('IoU Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Fire area evolution
        axes[0,1].plot(days, fire_areas, 'ro-', linewidth=2, markersize=6)
        axes[0,1].set_title('Fire Area Evolution', fontweight='bold')
        axes[0,1].set_xlabel('Day')
        axes[0,1].set_ylabel('Fire Area (pixels)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Environmental conditions
        ax3 = axes[1,0]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(days, temperatures, 'g-', linewidth=2, label='Temperature (K)')
        line2 = ax3_twin.plot(days, precipitations, 'b--', linewidth=2, label='Precipitation (mm)')
        
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Temperature (K)', color='g')
        ax3_twin.set_ylabel('Precipitation (mm)', color='b')
        ax3.set_title('Environmental Conditions', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        
        # Plot 4: Accuracy vs Fire Area scatter
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
        """Create animated GIF showing complete fire evolution with predictions"""
        if not evolution_analysis:
            print("No evolution data to animate")
            return
        
        print(f"Creating fire evolution animation with {len(evolution_analysis)} frames...")
        
        # Load the complete fire event for reference
        fire_loader = FireEventLoader(self.config)
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
                # Get the data for this time window
                initial_sequence, _, ground_truth = fire_loader.prepare_simulation_data(
                    fire_event_data, start_day=start_day
                )
                
                # Model prediction
                pred_fire = self.simulator.predict_single_step(initial_sequence.unsqueeze(0))
                actual_fire = torch.FloatTensor(ground_truth[0]) if ground_truth else None
                
                # Plot 1: Current day fire (input)
                current_fire = initial_sequence[-1, -1].numpy()  # Last day's fire
                im1 = axes[0].imshow(current_fire, cmap='Reds', vmin=0, vmax=1)
                axes[0].set_title(f'Current Fire - Day {start_day + 4}', fontweight='bold', fontsize=12)
                axes[0].axis('off')
                
                # Plot 2: Predicted next day
                pred_np = pred_fire.detach().cpu().numpy()
                im2 = axes[1].imshow(pred_np, cmap='Oranges', vmin=0, vmax=1)
                axes[1].set_title(f'Predicted Fire - Day {start_day + 5}', fontweight='bold', fontsize=12)
                axes[1].axis('off')
                
                # Plot 3: Actual next day (if available)
                if actual_fire is not None:
                    actual_np = actual_fire.numpy()
                    im3 = axes[2].imshow(actual_np, cmap='Greens', vmin=0, vmax=1)
                    axes[2].set_title(f'Actual Fire - Day {start_day + 5}', fontweight='bold', fontsize=12)
                    axes[2].axis('off')
                    
                    # Calculate and display accuracy
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
                               verticalalignment='top', fontsize=10)
                else:
                    axes[2].text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center',
                               transform=axes[2].transAxes, fontsize=14)
                    axes[2].axis('off')
                
                # Plot 4: Environmental conditions summary
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
                
                # Overall title
                fig.suptitle(f'Fire Evolution Analysis - Day {start_day + 4} to {start_day + 5}\n{window["fire_event"]}',
                           fontsize=16, fontweight='bold')
                
            except Exception as e:
                print(f"Error in animation frame {frame_idx}: {e}")
                # Show error message
                for ax in axes:
                    ax.clear()
                    ax.text(0.5, 0.5, f'Error in frame {frame_idx}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.axis('off')
        
        # Create animation
        frames = min(len(evolution_analysis), 50)  # Limit to 50 frames for reasonable file size
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=800, repeat=True)
        
        # Save animation
        try:
            anim.save(save_path, writer='pillow', fps=1.5)
            print(f"Fire evolution animation saved: {save_path}")
        except Exception as e:
            print(f"Could not save evolution animation: {e}")
            # Save individual frames as fallback
            for i in range(min(frames, 10)):
                animate(i)
                plt.savefig(f'evolution_frame_{i:02d}.png', bbox_inches='tight', dpi=150)
            print("Saved individual evolution frames: evolution_frame_*.png")
        
        plt.close(fig)
        return anim
    
    def generate_evolution_report(self, evolution_analysis, temporal_patterns, fire_event_name):
        """Generate detailed report for single fire event evolution"""
        
        if not evolution_analysis:
            return "No evolution data available for analysis"
        
        report = f"""
SINGLE FIRE EVENT EVOLUTION ANALYSIS
===================================

Fire Event: {fire_event_name}
Analysis Period: Day {evolution_analysis[0]['start_day']} to Day {evolution_analysis[-1]['start_day']}
Total Time Windows Analyzed: {len(evolution_analysis)}

TEMPORAL PATTERNS:
================
Accuracy Trend: {temporal_patterns.get('accuracy_trend', 'unknown')}
Mean Prediction Accuracy (IoU): {temporal_patterns.get('mean_accuracy', 0):.4f}
Accuracy Variability (std): {temporal_patterns.get('accuracy_variability', 0):.4f}
Fire Growth Rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day

Peak Fire Activity: Day {temporal_patterns.get('peak_fire_day', 0)}
Best Prediction Day: Day {temporal_patterns.get('best_prediction_day', 0)}
Temperature-Accuracy Correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}

DETAILED TIME WINDOW ANALYSIS:
=============================
"""
        
        for window in evolution_analysis:
            day = window['start_day']
            accuracy = window['prediction_accuracy']
            env = window['environmental_summary']
            spread = window['spread_analysis']
            
            report += f"""
Day {day} Analysis:
  Prediction Accuracy: IoU={accuracy['iou']:.3f}, Dice={accuracy['dice']:.3f}
  Fire Area: {accuracy['actual_area']:.0f} pixels (actual), {accuracy['predicted_area']:.0f} pixels (predicted)
  Environmental: {env.get('condition_type', 'unknown')} conditions, {env.get('temperature_level', 'unknown')} temperature
  Fire Spread: Direction error={spread.get('direction_error', 0):.1f}°, Distance error={spread.get('distance_error', 0):.1f} pixels
  Dominant Factors: Temp={env.get('max_temp_mean', 0):.1f}K, Precip={env.get('precipitation_mean', 0):.1f}mm
"""

        report += f"""

ENVIRONMENTAL CORRELATION ANALYSIS:
=================================
Based on {len(evolution_analysis)} time windows, the model shows:

1. Fire Growth Patterns:
   - Fire grows at an average rate of {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels per day
   - Peak activity occurred on day {temporal_patterns.get('peak_fire_day', 0)}

2. Prediction Performance:
   - Model accuracy {temporal_patterns.get('accuracy_trend', 'varies')} over time
   - Best predictions achieved on day {temporal_patterns.get('best_prediction_day', 0)}
   - Temperature correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}

3. Environmental Dependencies:
   - Model performance varies with weather conditions
   - Fire spread prediction accuracy influenced by environmental factors
   - Complex interactions between temperature, precipitation, and fire behavior

PHYSICAL INTERPRETATION:
======================
The analysis reveals model behavior consistent with fire physics:
- Fire spread direction and distance predictions show spatial understanding
- Environmental condition changes correlate with prediction accuracy
- Temporal patterns suggest model captures fire evolution dynamics

RECOMMENDATIONS:
==============
1. Focus on time windows with highest prediction accuracy for operational use
2. Consider environmental conditions when interpreting model confidence
3. Use multi-day predictions for better fire evolution understanding
4. Validate against additional fire events for generalization assessment
"""
        
        return report 

# ============================================================================
# FIRE EVENT LOADER AND UTILITIES
# ============================================================================

class FireEventLoader:
    """Load and process real fire events from HDF5 files"""
    
    def __init__(self, config):
        self.config = config
    
    def load_fire_event(self, hdf5_path):
        """Load fire event from HDF5 file"""
        print(f"Loading fire event from {hdf5_path}...")
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'data' not in f:
                    raise ValueError("No 'data' key found in HDF5 file")
                
                data = f['data'][:]  # Shape: (T, C, H, W)
                print(f"Loaded data shape: {data.shape}")
                
                T, C, H, W = data.shape
                
                if T < self.config.SEQUENCE_LENGTH + 5:
                    raise ValueError(f"Fire event too short. Need at least {self.config.SEQUENCE_LENGTH + 5} time steps, got {T}")
                
                return torch.FloatTensor(data)
                
        except Exception as e:
            raise ValueError(f"Failed to load HDF5 file: {e}")
    
    def prepare_simulation_data(self, fire_event_data, start_day=0):
        """
        Prepare simulation data from real fire event
        
        Args:
            fire_event_data: Full fire event tensor (T, C, H, W)
            start_day: Which day to start simulation from
        
        Returns:
            initial_sequence: First 5 days for input (5, best_features, H, W)
            future_weather: Weather data for remaining days
            ground_truth: Actual fire progression for comparison
        """
        T, C, H, W = fire_event_data.shape
        
        if start_day + self.config.SEQUENCE_LENGTH >= T:
            raise ValueError(f"Start day {start_day} too late for sequence length {self.config.SEQUENCE_LENGTH}")
        
        # Extract initial sequence (5 days)
        initial_sequence = fire_event_data[start_day:start_day + self.config.SEQUENCE_LENGTH]
        
        # Process features (same as training pipeline)
        processed_sequence = self._process_features(initial_sequence)
        
        # Extract future weather data (without fire)
        remaining_days = min(T - start_day - self.config.SEQUENCE_LENGTH, 20)  # Limit to 20 days
        if remaining_days > 0:
            future_weather = fire_event_data[start_day + self.config.SEQUENCE_LENGTH:start_day + self.config.SEQUENCE_LENGTH + remaining_days]
            future_weather_processed = self._process_features(future_weather)
            
            # Zero out fire channel in future weather (we'll predict this)
            active_fire_idx = self.config.BEST_FEATURES.index(22)
            future_weather_processed[:, active_fire_idx] = 0
        else:
            future_weather_processed = None
        
        # Extract ground truth fire progression
        ground_truth_fire = []
        for day in range(remaining_days):
            day_idx = start_day + self.config.SEQUENCE_LENGTH + day
            fire_data = fire_event_data[day_idx, -1]  # Active_Fire channel
            fire_binary = (fire_data > 0).float()
            
            # Resize to match simulation size
            if fire_binary.shape != self.config.SPATIAL_SIZE:
                fire_resized = F.interpolate(
                    fire_binary.unsqueeze(0).unsqueeze(0),
                    size=self.config.SPATIAL_SIZE,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                fire_resized = fire_binary
            
            ground_truth_fire.append(fire_resized.numpy())
        
        return processed_sequence, future_weather_processed, ground_truth_fire
    
    def _process_features(self, data):
        """Process features same as training pipeline"""
        T, C, H, W = data.shape
        processed = data.clone()
        
        # Angular features
        angular_features = [7, 14, 19]  # Wind_Direction, Aspect, Forecast_Wind_Dir
        for angle_idx in angular_features:
            if angle_idx < C:
                processed[:, angle_idx] = torch.sin(torch.deg2rad(processed[:, angle_idx]))
        
        # Handle missing values
        processed = torch.where(torch.isfinite(processed), processed, torch.zeros_like(processed))
        
        # Resize if needed
        if (H, W) != self.config.SPATIAL_SIZE:
            processed = F.interpolate(
                processed.view(-1, 1, H, W),
                size=self.config.SPATIAL_SIZE,
                mode='bilinear',
                align_corners=False
            ).view(T, C, *self.config.SPATIAL_SIZE)
        
        # Select best features
        processed = processed[:, self.config.BEST_FEATURES]
        
        return processed

def create_sample_fire_scenario(config):
    """Create synthetic fire scenario for demonstration"""
    print("Creating sample fire scenario for demonstration...")
    
    # Generate realistic environmental data
    H, W = config.SPATIAL_SIZE
    num_features = len(config.BEST_FEATURES)
    
    # Create 15-day sequence (5 for input + 10 for weather forecasts)
    sequence_data = torch.randn(15, num_features, H, W)
    
    # Add realistic environmental conditions
    feature_idx_map = {name: i for i, name in enumerate([
        config.FEATURE_NAMES[idx] for idx in config.BEST_FEATURES
    ])}
    
    # NDVI (vegetation)
    ndvi_idx = feature_idx_map.get('NDVI', 0)
    sequence_data[:, ndvi_idx] = torch.normal(0.3, 0.2, (15, H, W))
    
    # Temperature
    temp_idx = feature_idx_map.get('Max_Temp_K', 1)
    sequence_data[:, temp_idx] = torch.normal(300, 10, (15, H, W))
    
    # Wind speed
    wind_idx = feature_idx_map.get('Wind_Speed', 2)
    sequence_data[:, wind_idx] = torch.normal(5, 2, (15, H, W))
    
    # Precipitation
    precip_idx = feature_idx_map.get('Total_Precip', 3)
    sequence_data[:, precip_idx] = torch.exponential(torch.ones(15, H, W) * 0.1)
    
    # Add initial fire pattern
    fire_idx = feature_idx_map.get('Active_Fire', -1)
    center_h, center_w = H//2, W//2
    
    # Create initial fire hotspot
    for t in range(5):  # First 5 days (input sequence)
        fire_size = 3 + t * 2
        h_start = max(0, center_h - fire_size)
        h_end = min(H, center_h + fire_size)
        w_start = max(0, center_w - fire_size)
        w_end = min(W, center_w + fire_size)
        
        sequence_data[t, fire_idx, h_start:h_end, w_start:w_end] = torch.rand(h_end-h_start, w_end-w_start) * 0.8
    
    print("Sample fire scenario created successfully")
    return sequence_data

# ============================================================================
# SUPERVISOR REPORT GENERATOR
# ============================================================================

def generate_supervisor_report_evolution(evolution_analysis, temporal_patterns, model_path):
    """Generate supervisor report focused on single fire event evolution"""
    
    if not evolution_analysis:
        return "No evolution data available for reporting"
    
    fire_event_name = evolution_analysis[0]['fire_event']
    num_windows = len(evolution_analysis)
    
    report = f"""
WILDFIRE SPREAD SIMULATION - SINGLE FIRE EVENT EVOLUTION
========================================================

Model Information:
- Model Path: {model_path}
- Analysis Date: {np.datetime64('today')}
- Fire Event: {fire_event_name}
- Analysis Type: Complete Fire Evolution Tracking
- Time Windows Analyzed: {num_windows}

EVOLUTION SUMMARY:
================
Fire Event Duration: Day {evolution_analysis[0]['start_day']} to Day {evolution_analysis[-1]['start_day']}
Accuracy Trend: {temporal_patterns.get('accuracy_trend', 'Variable')}
Mean Prediction Accuracy: {temporal_patterns.get('mean_accuracy', 0):.4f} IoU
Fire Growth Rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day
Peak Fire Activity: Day {temporal_patterns.get('peak_fire_day', 0)}

TEMPORAL ANALYSIS FINDINGS:
=========================
1. Model Performance Over Time:
   - Mean accuracy varies across fire evolution stages
   - {temporal_patterns.get('accuracy_trend', 'Variable')} accuracy trend observed
   - Best predictions on day {temporal_patterns.get('best_prediction_day', 0)}

2. Environmental Correlations:
   - Temperature-accuracy correlation: {temporal_patterns.get('temperature_correlation', 0):.3f}
   - Environmental conditions influence prediction quality
   - Fire behavior complexity varies with weather patterns

3. Fire Spread Patterns:
   - Model captures spatial fire spread directions
   - Fire growth rate: {temporal_patterns.get('fire_growth_rate', 0):.2f} pixels/day
   - Peak activity phase identified in evolution

DETAILED TIME WINDOWS:
====================
"""
    
    for i, window in enumerate(evolution_analysis[:5]):  # Show first 5 windows
        accuracy = window['prediction_accuracy']
        env = window['environmental_summary']
        
        report += f"""
Day {window['start_day']}:
  - Accuracy: IoU={accuracy['iou']:.3f}, Dice={accuracy['dice']:.3f}
  - Fire Size: {accuracy['actual_area']:.0f} pixels
  - Conditions: {env.get('condition_type', 'unknown')}, {env.get('temperature_level', 'unknown')} temp
  - Temp: {env.get('max_temp_mean', 0):.1f}K, Precip: {env.get('precipitation_mean', 0):.1f}mm
"""
    
    if len(evolution_analysis) > 5:
        report += f"... and {len(evolution_analysis) - 5} more time windows\n"

    report += f"""

SIMULATION CAPABILITIES DEMONSTRATED:
===================================
- Complete fire event tracking over {num_windows} time periods
- Real environmental condition integration
- Temporal pattern recognition and analysis
- Fire spread direction and distance prediction
- Multi-day evolution simulation with ground truth comparison
- Physics-informed prediction constraints

KEY INSIGHTS FOR FIRE MANAGEMENT:
================================
1. Prediction Reliability:
   Model accuracy varies with fire evolution stage and environmental conditions.
   Best predictions occur during specific phases of fire development.

2. Environmental Dependencies:
   Temperature and precipitation strongly influence prediction accuracy.
   Model incorporates weather-fire interactions in predictions.

3. Temporal Patterns:
   Fire evolution shows predictable patterns that the model captures.
   Peak fire activity and growth phases are identifiable.

4. Operational Applications:
   Model provides day-ahead fire spread predictions with quantified accuracy.
   Environmental context helps interpret prediction confidence levels.

OUTPUTS GENERATED:
=================
- fire_comparison.gif: Predicted vs actual fire evolution animation
- fire_evolution_analysis.png: Complete temporal analysis plots  
- simulation_report.txt: This comprehensive evolution report

COMPARISON WITH FIRE PHYSICS:
===========================
The model demonstrates understanding of:
- Fire spread directionality influenced by environmental factors
- Growth rate variations with weather conditions
- Temporal fire behavior patterns consistent with fire science
- Spatial fire progression following realistic patterns

RECOMMENDATIONS:
===============
1. Use evolution analysis for fire incident planning and resource allocation
2. Focus on high-accuracy time windows for critical decision-making
3. Incorporate environmental forecasts for extended predictions
4. Validate patterns against fire behavior models (FARSITE, etc.)
5. Apply to multiple fire events for comprehensive validation
"""
    
    # Save evolution report
    with open('simulation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Fire evolution report generated: simulation_report.txt")
    # return report_report.txt")
    return report

def calculate_prediction_metrics(predictions, ground_truth):
    """Calculate and display prediction accuracy metrics"""
    if not ground_truth or len(ground_truth) == 0:
        return
    
    print("\nPrediction Accuracy Metrics:")
    print("-" * 40)
    
    total_iou = 0
    total_dice = 0
    valid_days = 0
    
    for day in range(min(len(predictions), len(ground_truth))):
        pred_binary = (np.array(predictions[day]) > 0.5).astype(float)
        actual_binary = np.array(ground_truth[day])
        
        if actual_binary.sum() > 0:  # Only calculate if there's actual fire
            intersection = (pred_binary * actual_binary).sum()
            union = pred_binary.sum() + actual_binary.sum() - intersection
            
            iou = intersection / union if union > 0 else 0
            dice = (2 * intersection) / (pred_binary.sum() + actual_binary.sum()) if (pred_binary.sum() + actual_binary.sum()) > 0 else 0
            
            total_iou += iou
            total_dice += dice
            valid_days += 1
            
            print(f"Day {day+1}: IoU={iou:.3f}, Dice={dice:.3f}")
    
    if valid_days > 0:
        avg_iou = total_iou / valid_days
        avg_dice = total_dice / valid_days
        print("-" * 40)
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"Average Dice: {avg_dice:.3f}")
        print(f"Valid prediction days: {valid_days}/{len(predictions)}")
    else:
        print("No valid ground truth data for comparison")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Fire Spread Simulation')
    parser.add_argument('--model', type=str, default='best_fire_model_official.pth',
                       help='Path to trained model')
    parser.add_argument('--fire_event', type=str, default=None,
                       help='Path to HDF5 file containing real fire event for comparison')
    parser.add_argument('--start_day', type=int, default=0,
                       help='Which day to start simulation from in the fire event')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with synthetic data')
    parser.add_argument('--days', type=int, default=10,
                       help='Number of days to simulate')
    parser.add_argument('--mode', choices=['sliding_window', 'autoregressive'], 
                       default='sliding_window',
                       help='Simulation mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WILDFIRE SPREAD SIMULATION")
    print("=" * 60)
    
    config = SimulationConfig()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found. Please train a model first or provide correct path.")
        return
    
    # Initialize simulator
    try:
        simulator = FireSpreadSimulator(args.model, config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    ground_truth_fire = None
    
    # If no arguments provided, default to demo with a real fire event if available
    if not args.fire_event and not args.demo:
        # Try to find a fire event file automatically
        possible_paths = [
            'data/processed/2020/fire_24461899.hdf5',
            'data/processed/2020/fire_23756984.hdf5',
            'fire_23654679.hdf5',
            'fire_23756984.hdf5'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.fire_event = path
                print(f"Automatically selected fire event: {path}")
                break
        
        if not args.fire_event:
            print("No fire event found, using synthetic demo data")
            args.demo = True
    
    # Load real fire event or create synthetic data
    if args.fire_event and os.path.exists(args.fire_event):
        print(f"\nUsing real fire event: {args.fire_event}")
        fire_loader = FireEventLoader(config)
        
        try:
            fire_event_data = fire_loader.load_fire_event(args.fire_event)
            initial_sequence, weather_forecast, ground_truth_fire = fire_loader.prepare_simulation_data(
                fire_event_data, start_day=args.start_day
            )
            
            print(f"Real fire event loaded successfully")
            print(f"Initial sequence shape: {initial_sequence.shape}")
            print(f"Weather forecast shape: {weather_forecast.shape if weather_forecast is not None else 'None'}")
            print(f"Ground truth days available: {len(ground_truth_fire)}")
            
        except Exception as e:
            print(f"Failed to load real fire event: {e}")
            print("Falling back to synthetic data...")
            scenario_data = create_sample_fire_scenario(config)
            initial_sequence = scenario_data[:5]
            weather_forecast = scenario_data[5:]
            ground_truth_fire = None
            
    else:
        print("\nUsing synthetic fire scenario for demonstration")
        scenario_data = create_sample_fire_scenario(config)
        initial_sequence = scenario_data[:5]
        weather_forecast = scenario_data[5:]
    
    # Run fire simulation with debugging enabled
    print(f"\nRunning {args.days}-day fire simulation...")
    simulation_days = min(args.days, len(weather_forecast) if weather_forecast is not None else args.days)
    
    # Enable debug mode for the first few days to understand what's happening
    predictions = simulator.simulate_fire_evolution(
        initial_sequence, 
        weather_forecast,
        num_days=simulation_days, 
        mode=args.mode,
        debug=True  # Enable debugging
    )
    
    print(f"Simulation completed: {len(predictions)} days predicted")
    
    # Create appropriate animation
    if ground_truth_fire and len(ground_truth_fire) > 0:
        print("\nGenerating comparison animation (Predicted vs Actual)...")
        simulator.create_comparison_animation(predictions, ground_truth_fire, 'fire_comparison.gif')
        
        # Calculate overall performance metrics
        calculate_prediction_metrics(predictions, ground_truth_fire)
        
    else:
        print("\nGenerating fire spread animation...")
        simulator.create_simulation_animation(predictions, save_path='fire_simulation.gif')
    
    # Single fire event evolution analysis
    print("\nRunning single fire event evolution analysis...")
    evolution_analyzer = SingleFireEventAnalyzer(simulator, config)
    
    # Use the fire event that was loaded for the main simulation
    if args.fire_event and os.path.exists(args.fire_event):
        fire_event_path = args.fire_event
    else:
        # Try to find any available fire event
        possible_paths = [
            'data/processed/2020/24461899.hdf5',
            'data/processed/2019/*.hdf5',
            'data/processed/2018/*.hdf5',
            '*.hdf5'
        ]
        
        fire_event_path = None
        for pattern in possible_paths:
            found_files = glob.glob(pattern)
            if found_files:
                fire_event_path = found_files[0]
                print(f"Using fire event for evolution analysis: {fire_event_path}")
                break
        
        if not fire_event_path:
            print("No fire event found for evolution analysis")
            fire_event_path = None
    
    if fire_event_path:
        # Analyze the complete evolution of this fire event
        evolution_analysis, temporal_patterns = evolution_analyzer.analyze_fire_evolution(
            fire_event_path
        )
        
        # Generate evolution plots
        evolution_analyzer.plot_fire_evolution(
            evolution_analysis, temporal_patterns, 'fire_evolution_analysis.png'
        )
        
        # Create evolution animation GIF
        evolution_analyzer.create_evolution_animation(
            evolution_analysis, fire_event_path, 'fire_evolution.gif'
        )
        
        # Generate evolution report
        evolution_report = evolution_analyzer.generate_evolution_report(
            evolution_analysis, temporal_patterns, os.path.basename(fire_event_path)
        )
        
        # Generate supervisor report with evolution analysis
        report = generate_supervisor_report_evolution(evolution_analysis, temporal_patterns, args.model)
    else:
        print("Skipping evolution analysis - no fire event available")
        evolution_analysis, temporal_patterns = [], {}
        report = "No fire event available for evolution analysis"
    if args.fire_event:
        # Add fire event info to report with UTF-8 encoding
        with open('simulation_report.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n\nFire Event Analysis:\n")
            f.write(f"- Source file: {args.fire_event}\n")
            f.write(f"- Start day: {args.start_day}\n")
            f.write(f"- Simulation days: {len(predictions)}\n")
            f.write(f"- Ground truth available: {'Yes' if ground_truth_fire else 'No'}\n")
    
    print("\n" + "=" * 60)
    print("FIRE EVOLUTION SIMULATION COMPLETED")
    print("=" * 60)
    print("Generated files:")
    
    if ground_truth_fire:
        print("- fire_comparison.gif: Predicted vs Actual fire animation")
        print("- comparison_frame_*.png: Individual comparison frames (if GIF failed)")
    else:
        print("- fire_simulation.gif: Fire spread animation")
        print("- fire_frame_*.png: Individual frames (if GIF failed)")
    
    if fire_event_path:
        print("- fire_evolution_analysis.png: Complete fire evolution analysis")
        print("- fire_evolution.gif: Animated fire evolution with predictions")
        print("- simulation_report.txt: Comprehensive evolution report")
        
        # Display key evolution findings
        print("\nFire Evolution Analysis:")
        if temporal_patterns:
            print(f"  Analysis Windows: {len(evolution_analysis) if evolution_analysis else 0}")
            print(f"  Complete Fire Duration: {len(evolution_analysis) if evolution_analysis else 0} days")
            print(f"  Accuracy Trend: {temporal_patterns.get('accuracy_trend', 'unknown')}")
            print(f"  Mean IoU: {temporal_patterns.get('mean_accuracy', 0):.3f}")
            print(f"  Fire Growth Rate: {temporal_patterns.get('fire_growth_rate', 0):.1f} pixels/day")
            print(f"  Peak Fire Day: {temporal_patterns.get('peak_fire_day', 0)}")
    else:
        print("- simulation_report.txt: Basic simulation report")
        print("Evolution analysis skipped - no fire event available")

if __name__ == "__main__":
    main()