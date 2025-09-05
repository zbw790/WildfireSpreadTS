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

# Register compatibility classes for model loading
import sys
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# Import from your existing code
import sys
sys.path.append('.')
from simple_feature_sensitivity import load_fire_event_data

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

def analyze_fire_no_fire_distribution(test_targets):
    """
    详细分析测试数据中有火天和无火天的分布
    """
    print("\n🔍 FIRE/NO-FIRE DAY ANALYSIS")
    print("="*50)
    
    fire_days = []
    no_fire_days = []
    
    for day_idx, target in enumerate(test_targets):
        fire_pixels = (target > 0.5).sum().item()
        total_pixels = target.numel()
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
    
    print(f"📊 Data Distribution:")
    print(f"  • Fire days: {len(fire_days)}")
    print(f"  • No-fire days: {len(no_fire_days)}")
    print(f"  • Total days: {len(test_targets)}")
    if len(test_targets) > 0:
        print(f"  • Fire day ratio: {len(fire_days)/len(test_targets)*100:.1f}%")
    else:
        print(f"  • Fire day ratio: N/A (no test data)")
    
    if fire_days:
        print(f"\n🔥 Fire Days Details:")
        for day_info in fire_days:
            print(f"  Day {day_info['day']}: {day_info['fire_pixels']} pixels ({day_info['fire_ratio']*100:.3f}%)")
    
    if no_fire_days:
        print(f"\n❄️ No-Fire Days: {len(no_fire_days)} days")
    
    return len(fire_days), len(no_fire_days)

def calculate_fair_ap_with_analysis(predictions, targets, model_name):
    """
    计算AP并提供详细分析，特别关注有火天和无火天的影响
    """
    print(f"\n📊 AP Analysis for {model_name}")
    print("-" * 30)
    
    # 分析每天的情况
    daily_results = []
    fire_day_predictions = []
    fire_day_targets = []
    no_fire_day_predictions = []
    no_fire_day_targets = []
    
    for day_idx, (pred, target) in enumerate(zip(predictions, targets)):
        fire_pixels = (target > 0.5).sum().item()
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        if fire_pixels > 0:  # 有火天
            fire_day_predictions.append(pred_flat)
            fire_day_targets.append(target_flat)
            daily_results.append({
                'day': day_idx,
                'type': 'fire',
                'fire_pixels': fire_pixels,
                'pred_mean': pred_flat.mean(),
                'pred_max': pred_flat.max()
            })
        else:  # 无火天
            no_fire_day_predictions.append(pred_flat)
            no_fire_day_targets.append(target_flat)
            daily_results.append({
                'day': day_idx,
                'type': 'no_fire',
                'fire_pixels': fire_pixels,
                'pred_mean': pred_flat.mean(),
                'pred_max': pred_flat.max()
            })
    
    # 计算不同方式的AP
    results = {}
    
    # 方法1: 所有天合并计算（当前使用的方法）
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_targets = np.concatenate([t.flatten() for t in targets])
    
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
    daily_aps = []
    for day_idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        if target_flat.sum() > 0:
            daily_ap = average_precision_score(target_flat, pred_flat)
        else:
            daily_ap = 0.0  # 无火天设为0
        daily_aps.append(daily_ap)
    
    results['daily_average_ap'] = np.mean(daily_aps)
    
    # 打印详细分析
    print(f"  🔥 Fire days: {len(fire_day_predictions)}")
    print(f"  ❄️ No-fire days: {len(no_fire_day_predictions)}")
    print(f"")
    print(f"  📈 AP Calculation Methods:")
    print(f"    Combined (all days): {results['combined_ap']:.4f}")
    print(f"    Fire days only:      {results['fire_days_only_ap']:.4f}")
    print(f"    Daily average:       {results['daily_average_ap']:.4f}")
    
    if results['fire_days_only_ap'] > 0 and results['combined_ap'] > 0:
        ratio = results['fire_days_only_ap'] / results['combined_ap']
        print(f"    Fire-only vs Combined: {ratio:.2f}x")
    
    return results['combined_ap'], results

def run_baseline_comparison(fire_event_path="data/processed/2020/fire_24461899.hdf5"):
    """
    Run all baseline models and compare performance using multiple days for fair evaluation
    """
    print("🔥 QUICK BASELINE COMPARISON - MULTI-DAY EVALUATION")
    print("="*50)
    
    # Import SimpleConfig here to avoid conflicts
    from simple_feature_sensitivity import SimpleConfig
    config = SimpleConfig()
    baselines = QuickBaselines(config)
    
    # Load data for multiple days - use more days for fairer evaluation
    print("Loading fire event data for multi-day evaluation...")
    all_sequences = []
    all_targets = []
    test_sequences = []
    test_targets = []
    
    # Load first 5 days as training data
    for day in range(5):
        try:
            seq, _, gt, _ = load_fire_event_data(fire_event_path, config, start_day=day)
            if len(gt) > 0:
                all_sequences.append(seq)
                all_targets.append(torch.tensor(gt[0], dtype=torch.float32))
        except:
            break
    
    # Load multiple test days (6-15) for fair evaluation
    print("Loading multiple test days for comprehensive evaluation...")
    for day in range(6, 16):  # Use days 6-15 as test (10 days)
        try:
            seq, _, gt, _ = load_fire_event_data(fire_event_path, config, start_day=day)
            if len(gt) > 0:
                test_sequences.append(seq)
                test_targets.append(torch.tensor(gt[0], dtype=torch.float32))
        except:
            break
    
    if len(test_sequences) == 0:
        print("No valid test sequences found, using single day")
        test_seq, _, test_gt, _ = load_fire_event_data(fire_event_path, config, start_day=6)
        if test_seq is not None:
            test_sequences = [test_seq]
            if len(test_gt) > 0:
                test_targets = [torch.tensor(test_gt[0], dtype=torch.float32)]
            else:
                test_targets = [torch.zeros(config.SPATIAL_SIZE)]
        else:
            print("Failed to load any test data")
            return
    
    print(f"Training data: {len(all_sequences)} sequences")
    print(f"Test data: {len(test_sequences)} sequences")
    
    # Analyze fire/no-fire distribution
    fire_days, no_fire_days = analyze_fire_no_fire_distribution(test_targets)
    
    # Store results
    results = {}
    
    # 1. Persistence Model
    print("\n1. PERSISTENCE MODEL")
    start_time = time.time()
    
    # Predict on all test days
    all_persistence_preds = []
    for test_seq in test_sequences:
        pred = baselines.persistence_model(test_seq)
        all_persistence_preds.append(pred)
    
    persistence_time = time.time() - start_time
    
    # Calculate AP with detailed analysis
    persistence_ap, persistence_analysis = calculate_fair_ap_with_analysis(
        all_persistence_preds, test_targets, "Persistence"
    )
    
    results['Persistence'] = {
        'predictions': all_persistence_preds,
        'ap_score': persistence_ap,
        'time': persistence_time,
        'analysis': persistence_analysis
    }
    print(f"  ✓ Time: {persistence_time:.4f}s")
    
    # 2. Mean Baseline
    print("\n2. MEAN BASELINE")
    start_time = time.time()
    try:
        # Predict on all test days
        all_mean_preds = []
        for test_seq in test_sequences:
            pred = baselines.mean_baseline(all_sequences, all_targets, test_seq)
            all_mean_preds.append(pred)
        
        mean_time = time.time() - start_time
        
        # Calculate AP with detailed analysis
        mean_ap, mean_analysis = calculate_fair_ap_with_analysis(
            all_mean_preds, test_targets, "Mean Baseline"
        )
            
        results['Mean'] = {
            'predictions': all_mean_preds,
            'ap_score': mean_ap,
            'time': mean_time,
            'analysis': mean_analysis
        }
        print(f"  ✓ Time: {mean_time:.4f}s")
    except Exception as e:
        print(f"  ✗ Mean baseline failed: {e}")
        results['Mean'] = None
    
    # 3. Simple CNN
    print("\n3. SIMPLE CNN")
    start_time = time.time()
    try:
        # Predict on all test days
        all_cnn_preds = []
        for test_seq in test_sequences:
            pred = baselines.simple_cnn_model(all_sequences, all_targets, test_seq)
            all_cnn_preds.append(pred)
        
        cnn_time = time.time() - start_time
        
        # Calculate AP with detailed analysis
        cnn_ap, cnn_analysis = calculate_fair_ap_with_analysis(
            all_cnn_preds, test_targets, "Simple CNN"
        )
            
        results['SimpleCNN'] = {
            'predictions': all_cnn_preds,
            'ap_score': cnn_ap,
            'time': cnn_time,
            'analysis': cnn_analysis
        }
        print(f"  ✓ Time: {cnn_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Simple CNN failed: {e}")
        results['SimpleCNN'] = None
    
    # Test main UNet model for comparison
    print("\n4. MAIN UNET MODEL")
    start_time = time.time()
    try:
        # Use the exact same method as simple_feature_sensitivity.py
        from simple_feature_sensitivity import load_model_with_compatibility, SimpleFireSimulator, SimpleConfig
        
        # Create config exactly like simple_feature_sensitivity
        main_config = SimpleConfig()
        
        print("Loading model using simple_feature_sensitivity method...")
        # Load model with exact same parameters as simple_feature_sensitivity
        model = load_model_with_compatibility(
            'best_fire_model_official.pth', 
            len(main_config.BEST_FEATURES),  # 13 features
            main_config.SEQUENCE_LENGTH,     # 5 sequence length
            baselines.device
        )
        
        if model is None:
            print("Failed to load model")
            results['Main UNet'] = None
        else:
            # Initialize simulator exactly like simple_feature_sensitivity
            simulator = SimpleFireSimulator(model, main_config, baselines.device)
            
            # Predict on all test days
            all_main_preds = []
            for test_seq in test_sequences:
                pred = simulator.predict_single_step(test_seq.unsqueeze(0), debug=False)
                all_main_preds.append(pred.numpy().squeeze())
            
            main_time = time.time() - start_time
            
            # Calculate AP with detailed analysis
            main_ap, main_analysis = calculate_fair_ap_with_analysis(
                all_main_preds, test_targets, "Main UNet"
            )
            
            results['Main UNet'] = {
                'predictions': all_main_preds,
                'ap_score': main_ap,
                'time': main_time,
                'analysis': main_analysis
            }
            print(f"  ✓ Time: {main_time:.2f}s")
        
        # Calculate improvement over best baseline
        baseline_aps = [r['ap_score'] for r in results.values() if r is not None and 'Main' not in str(r)]
        if baseline_aps:
            best_baseline_ap = max(baseline_aps)
            if best_baseline_ap > 0:
                improvement = main_ap / best_baseline_ap
                print(f"  🚀 {improvement:.1f}x better than best baseline!")
        
    except Exception as e:
        print(f"  ✗ Main UNet failed: {e}")
        results['Main UNet'] = None
    
    # Create comparison visualization (now including main model)
    # Use first test target for visualization
    first_test_target = test_targets[0] if len(test_targets) > 0 else torch.zeros(config.SPATIAL_SIZE)
    create_baseline_comparison_plot(results, first_test_target)
    
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
            
        # Handle both old (single prediction) and new (multiple predictions) format
        if 'prediction' in result:
            pred = result['prediction']  # Old format
        elif 'predictions' in result and len(result['predictions']) > 0:
            pred = result['predictions'][0]  # New format - use first prediction for visualization
        else:
            continue  # Skip if no valid prediction
            
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
