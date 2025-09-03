#!/usr/bin/env python3
"""
Quick Baseline Models for Fire Prediction Comparison

Creates simple baseline models to compare against the main UNet model.
Focus: Fast implementation, reasonable performance benchmarks.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Compatibility classes for model loading
class WildFireConfig:
    pass

class FirePredictionConfig:
    pass

# Register classes for safe loading
import sys
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# Import from your existing code
import sys
sys.path.append('.')
from simple_feature_sensitivity import load_fire_event_data, SimpleConfig

class QuickBaselines:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def persistence_model(self, sequence):
        """
        Persistence Model: Tomorrow's fire = Today's fire
        The simplest possible baseline!
        """
        # Use the last frame's fire channel as prediction
        last_frame = sequence[-1]  # [features, h, w]
        fire_channel = last_frame[-1]  # Active_Fire is the last feature
        return fire_channel.numpy()
    
    def mean_baseline(self, train_sequences, train_targets, test_sequence):
        """
        Mean Baseline: Predict the average fire probability from training data
        """
        print("  Computing Mean Baseline...")
        
        # Calculate mean fire probability from training targets
        all_targets = torch.stack(train_targets)
        mean_fire_prob = all_targets.mean().item()
        
        print(f"    Mean fire probability: {mean_fire_prob:.4f}")
        
        # Return constant prediction
        return np.full(self.config.SPATIAL_SIZE, mean_fire_prob)
    
    def simple_cnn_model(self, train_sequences, train_targets, test_sequence):
        """
        Simple 2-layer CNN (only uses last frame)
        """
        print("  Training Simple CNN...")
        
        class SimpleCNN(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
                self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
                self.dropout = nn.Dropout2d(0.2)
            
            def forward(self, x):
                # Only use the last frame
                x = x[:, -1]  # [batch, features, h, w]
                x = F.relu(self.conv1(x))
                x = self.dropout(x)
                x = F.relu(self.conv2(x))
                x = torch.sigmoid(self.conv3(x))
                return x.squeeze(1)
        
        # Create model
        model = SimpleCNN(len(self.config.BEST_FEATURES)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Quick training (just a few epochs)
        model.train()
        for epoch in range(5):  # Very quick training
            total_loss = 0
            n_batches = min(10, len(train_sequences))  # Use subset for speed
            for seq, target in zip(train_sequences[:n_batches], train_targets[:n_batches]):
                seq = seq.unsqueeze(0).to(self.device)
                target = target.unsqueeze(0).to(self.device)  # Add batch dimension
                
                optimizer.zero_grad()
                pred = model(seq)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"    Epoch {epoch+1}/5, Loss: {total_loss/n_batches:.4f}")
        
        # Predict
        model.eval()
        with torch.no_grad():
            test_input = test_sequence.unsqueeze(0).to(self.device)
            pred = model(test_input)
            return pred.cpu().numpy().squeeze()

def run_baseline_comparison(fire_event_path="data/processed/2020/fire_24461899.hdf5"):
    """
    Run all baseline models and compare performance
    """
    print("🔥 QUICK BASELINE COMPARISON")
    print("="*50)
    
    config = SimpleConfig()
    baselines = QuickBaselines(config)
    
    # Load data for multiple days to create train/test split
    print("Loading fire event data...")
    all_sequences = []
    all_targets = []
    
    # Load multiple days as training data
    for day in range(5):  # Use first 5 days as training
        try:
            seq, _, gt, _ = load_fire_event_data(fire_event_path, config, start_day=day)
            if len(gt) > 0:
                all_sequences.append(seq)
                all_targets.append(torch.tensor(gt[0], dtype=torch.float32))
        except:
            break
    
    # Use day 6 as test
    test_seq, _, test_gt, _ = load_fire_event_data(fire_event_path, config, start_day=6)
    test_target = torch.tensor(test_gt[0], dtype=torch.float32) if len(test_gt) > 0 else torch.zeros(config.SPATIAL_SIZE)
    
    print(f"Training data: {len(all_sequences)} sequences")
    print(f"Test target shape: {test_target.shape}")
    
    # Store results
    results = {}
    
    # 1. Persistence Model
    print("\n1. PERSISTENCE MODEL")
    start_time = time.time()
    persistence_pred = baselines.persistence_model(test_seq)
    persistence_time = time.time() - start_time
    
    # Calculate metrics
    persistence_ap = average_precision_score(test_target.flatten(), persistence_pred.flatten())
    results['Persistence'] = {
        'prediction': persistence_pred,
        'ap_score': persistence_ap,
        'time': persistence_time
    }
    print(f"  ✓ AP Score: {persistence_ap:.4f}")
    print(f"  ✓ Time: {persistence_time:.4f}s")
    
    # 2. Mean Baseline
    print("\n2. MEAN BASELINE")
    start_time = time.time()
    try:
        mean_pred = baselines.mean_baseline(all_sequences, all_targets, test_seq)
        mean_time = time.time() - start_time
        mean_ap = average_precision_score(test_target.flatten(), mean_pred.flatten())
        results['Mean'] = {
            'prediction': mean_pred,
            'ap_score': mean_ap,
            'time': mean_time
        }
        print(f"  ✓ AP Score: {mean_ap:.4f}")
        print(f"  ✓ Time: {mean_time:.4f}s")
    except Exception as e:
        print(f"  ✗ Mean baseline failed: {e}")
        results['Mean'] = None
    
    # 3. Simple CNN
    print("\n3. SIMPLE CNN")
    start_time = time.time()
    try:
        cnn_pred = baselines.simple_cnn_model(all_sequences, all_targets, test_seq)
        cnn_time = time.time() - start_time
        cnn_ap = average_precision_score(test_target.flatten(), cnn_pred.flatten())
        results['SimpleCNN'] = {
            'prediction': cnn_pred,
            'ap_score': cnn_ap,
            'time': cnn_time
        }
        print(f"  ✓ AP Score: {cnn_ap:.4f}")
        print(f"  ✓ Time: {cnn_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Simple CNN failed: {e}")
        results['SimpleCNN'] = None
    
    # Test main UNet model for comparison
    print("\n4. MAIN UNET MODEL")
    start_time = time.time()
    try:
        from simple_feature_sensitivity import load_model_with_compatibility, SimpleFireSimulator
        
        # Load main model
        model = load_model_with_compatibility('best_fire_model_official.pth', 13, 5, baselines.device)
        simulator = SimpleFireSimulator(model, config, baselines.device)
        
        # Predict with main model
        main_pred = simulator.predict_single_step(test_seq.unsqueeze(0), debug=False)
        main_time = time.time() - start_time
        main_pred_np = main_pred.numpy().squeeze()
        main_ap = average_precision_score(test_target.flatten(), main_pred_np.flatten())
        
        results['Main UNet'] = {
            'prediction': main_pred_np,
            'ap_score': main_ap,
            'time': main_time
        }
        print(f"  ✓ AP Score: {main_ap:.4f}")
        print(f"  ✓ Time: {main_time:.2f}s")
        print(f"  🚀 {main_ap/max(r['ap_score'] for r in results.values() if r is not None and 'Main' not in r):.1f}x better than best baseline!")
        
    except Exception as e:
        print(f"  ✗ Main UNet failed: {e}")
        results['Main UNet'] = None
    
    # Create comparison visualization (now including main model)
    create_baseline_comparison_plot(results, test_target)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 COMPLETE MODEL COMPARISON")
    print("="*50)
    
    # Sort by AP score for better display
    sorted_results = sorted([(name, result) for name, result in results.items() if result is not None], 
                           key=lambda x: x[1]['ap_score'], reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:12}: AP={result['ap_score']:.4f}, Time={result['time']:.2f}s")
    
    print(f"\n📁 Complete comparison saved: baseline_comparison.png")
    
    return results

def create_baseline_comparison_plot(results, ground_truth):
    """
    Create a visual comparison of all model predictions (baselines + main model)
    """
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_models = len(valid_results)
    
    if n_models == 0:
        print("No valid results to plot")
        return
    
    # Sort by AP score for better visualization
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['ap_score'], reverse=True)
    
    # Create subplot grid - adjust for more models
    cols = min(n_models + 1, 5)  # Max 5 columns
    rows = 2 * ((n_models + 1 + cols - 1) // cols)  # Calculate needed rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot ground truth
    axes[0, 0].imshow(ground_truth, cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].text(0.5, 0.5, f'Ground Truth\n\nFire pixels: {(ground_truth > 0.5).sum()}', 
                   ha='center', va='center', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    axes[1, 0].axis('off')
    
    # Plot each model (sorted by performance)
    for i, (name, result) in enumerate(sorted_results):
        if i + 1 >= cols * rows // 2:  # Skip if we run out of space
            break
            
        pred = result['prediction']
        ap_score = result['ap_score']
        time_taken = result['time']
        
        col = (i + 1) % cols
        row_offset = 0 if (i + 1) < cols else 2
        
        # Prediction plot
        cmap = 'Blues' if 'Main' in name else 'Oranges'  # Different color for main model
        axes[row_offset, col].imshow(pred, cmap=cmap, vmin=0, vmax=pred.max())
        title = f'{name}\nAP: {ap_score:.3f}'
        if 'Main' in name:
            title += ' 🏆'  # Crown for main model
        axes[row_offset, col].set_title(title, fontweight='bold')
        axes[row_offset, col].axis('off')
        
        # Stats
        fire_pixels = (pred > 0.1).sum()
        max_pred = pred.max()
        
        stats_text = f'{name}\n\nAP Score: {ap_score:.4f}\nTime: {time_taken:.2f}s\nFire pixels: {fire_pixels}\nMax pred: {max_pred:.3f}'
        
        color = 'lightgreen' if 'Main' in name else 'lightyellow'
        axes[row_offset + 1, col].text(0.5, 0.5, stats_text,
                           ha='center', va='center', transform=axes[row_offset + 1, col].transAxes,
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        axes[row_offset + 1, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(sorted_results) + 1, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'🔥 Fire Prediction Model Comparison\n{n_models} Models + Ground Truth', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_baseline_comparison()
