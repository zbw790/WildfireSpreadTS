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
        Enhanced Persistence Model: Tomorrow's fire = Today's fire with decay and smoothing
        """
        # Use the last frame's fire channel as prediction
        last_frame = sequence[-1]  # [features, h, w]
        fire_channel = last_frame[-1]  # Active_Fire is the last feature
        
        # Apply some fire physics: decay + spatial smoothing
        prediction = fire_channel.numpy()
        
        # 1. Apply fire decay (fires naturally diminish)
        prediction = prediction * 0.85  # 15% decay rate
        
        # 2. Add spatial spreading (fires can spread to neighbors)
        from scipy import ndimage
        # Slight expansion with Gaussian kernel
        prediction = ndimage.gaussian_filter(prediction, sigma=0.8, mode='constant')
        
        # 3. Apply threshold to maintain fire intensity
        prediction = np.where(prediction > 0.1, prediction * 1.2, prediction)
        
        # 4. Clip to valid range
        prediction = np.clip(prediction, 0, 1)
        
        return prediction
    
    def linear_regression_model(self, train_sequences, train_targets, test_sequence):
        """
        Enhanced Linear Model: Logistic Regression with key features only
        """
        print("  Training Enhanced Linear Model...")
        
        # Use only the most important features to avoid memory issues
        # Focus on: NDVI, EVI2, VIIRS_M11, Max_Temp, Min_Temp, Fire
        key_features = [3, 4, 0, 8, 9, -1]  # NDVI, EVI2, VIIRS_M11, temps, fire
        
        # Prepare training data (sample to avoid memory issues)
        n_samples = min(1000, len(train_sequences))  # Limit samples
        sampled_indices = np.random.choice(len(train_sequences), n_samples, replace=False)
        
        X_train = []
        y_train = []
        
        for idx in sampled_indices:
            seq = train_sequences[idx]
            target = train_targets[idx]
            
            # Use last frame, selected features
            last_frame = seq[-1][key_features]  # [key_features, h, w]
            
            # Sample pixels (not all pixels to avoid memory issues)
            h, w = last_frame.shape[1], last_frame.shape[2]
            n_pixels = min(1000, h * w)  # Sample max 1000 pixels per image
            pixel_indices = np.random.choice(h * w, n_pixels, replace=False)
            
            # Flatten and sample
            X_sample = last_frame.view(len(key_features), -1)[:, pixel_indices].T  # [n_pixels, features]
            y_sample = target.view(-1)[pixel_indices]  # [n_pixels]
            
            X_train.append(X_sample.numpy())
            y_train.append((y_sample > 0.5).float().numpy())
        
        # Combine all training data
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        
        print(f"    Training on {X_train.shape[0]} pixel samples")
        print(f"    Positive rate: {y_train.mean():.4f}")
        
        # Train logistic regression with balanced class weights
        model = LogisticRegression(
            class_weight='balanced',  # Handle imbalance
            max_iter=100,  # Quick training
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predict on test sequence
        test_last_frame = test_sequence[-1][key_features]  # [key_features, h, w]
        h, w = test_last_frame.shape[1], test_last_frame.shape[2]
        
        # Reshape for prediction
        X_test = test_last_frame.view(len(key_features), -1).T.numpy()  # [h*w, features]
        
        # Predict probabilities
        pred_probs = model.predict_proba(X_test)[:, 1]  # Get positive class probabilities
        
        # Reshape back to spatial format
        prediction = pred_probs.reshape(h, w)
        
        print(f"    Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
        
        return prediction
    
    def simple_cnn_model(self, train_sequences, train_targets, test_sequence):
        """
        Enhanced Simple CNN with better architecture and training
        """
        print("  Training Enhanced Simple CNN...")
        
        class EnhancedSimpleCNN(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                # More sophisticated architecture
                self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2)  # Larger kernel
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(16)
                self.conv4 = nn.Conv2d(16, 1, 1)  # 1x1 conv for final prediction
                self.dropout = nn.Dropout2d(0.3)
            
            def forward(self, x):
                # Only use the last frame
                x = x[:, -1]  # [batch, features, h, w]
                
                # First conv block
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.dropout(x)
                
                # Second conv block
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.dropout(x)
                
                # Third conv block
                x = F.relu(self.bn3(self.conv3(x)))
                
                # Final prediction
                x = torch.sigmoid(self.conv4(x))
                return x.squeeze(1)
        
        # Create model
        model = EnhancedSimpleCNN(len(self.config.BEST_FEATURES)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Use weighted BCE loss to handle class imbalance
        pos_weight = torch.tensor([10.0]).to(self.device)  # Weight positive class more
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Better training with more epochs and validation
        model.train()
        n_batches = min(20, len(train_sequences))  # Use more batches
        
        for epoch in range(10):  # More training epochs
            total_loss = 0
            n_processed = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(train_sequences))[:n_batches]
            
            for idx in indices:
                seq = train_sequences[idx].unsqueeze(0).to(self.device)
                target = train_targets[idx].unsqueeze(0).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (remove sigmoid since we use BCEWithLogitsLoss)
                output = model.conv1(seq[:, -1])
                output = F.relu(model.bn1(output))
                output = model.dropout(output)
                output = F.relu(model.bn2(model.conv2(output)))
                output = model.dropout(output)
                output = F.relu(model.bn3(model.conv3(output)))
                logits = model.conv4(output).squeeze(1)  # Raw logits
                
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_processed += 1
            
            if epoch % 3 == 0:
                avg_loss = total_loss / max(n_processed, 1)
                print(f"    Epoch {epoch+1}/10, Avg Loss: {avg_loss:.4f}")
        
        # Predict
        model.eval()
        with torch.no_grad():
            test_input = test_sequence.unsqueeze(0).to(self.device)
            pred = model(test_input)  # This will apply sigmoid
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
            result = load_fire_event_data(fire_event_path, config, start_day=day)
            if len(result) == 5:  # Handle 5 return values
                seq, _, gt, _, _ = result
            else:  # Handle 4 return values
                seq, _, gt, _ = result
            if len(gt) > 0:
                all_sequences.append(seq)
                all_targets.append(torch.tensor(gt[0], dtype=torch.float32))
        except Exception as e:
            print(f"Error loading training day {day}: {e}")
            break
    
    # Load multiple test days (6-15) for fair evaluation
    print("Loading multiple test days for comprehensive evaluation...")
    for day in range(6, 16):  # Use days 6-15 as test (10 days)
        try:
            result = load_fire_event_data(fire_event_path, config, start_day=day)
            if len(result) == 5:  # Handle 5 return values
                seq, _, gt, _, _ = result
            else:  # Handle 4 return values
                seq, _, gt, _ = result
            if len(gt) > 0:
                test_sequences.append(seq)
                test_targets.append(torch.tensor(gt[0], dtype=torch.float32))
        except Exception as e:
            print(f"Error loading test day {day}: {e}")
            break
    
    if len(test_sequences) == 0:
        print("No valid test sequences found, using single day")
        try:
            result = load_fire_event_data(fire_event_path, config, start_day=6)
            if len(result) == 5:  # Handle 5 return values
                test_seq, _, test_gt, _, _ = result
            else:  # Handle 4 return values
                test_seq, _, test_gt, _ = result
            if test_seq is not None:
                test_sequences = [test_seq]
                if len(test_gt) > 0:
                    test_targets = [torch.tensor(test_gt[0], dtype=torch.float32)]
                else:
                    test_targets = [torch.zeros(config.SPATIAL_SIZE)]
            else:
                print("Failed to load any test data")
                return
        except Exception as e:
            print(f"Failed to load single test day: {e}")
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
    
    # 2. Enhanced Linear Model
    print("\n2. ENHANCED LINEAR MODEL")
    start_time = time.time()
    try:
        # Predict on all test days
        all_linear_preds = []
        for test_seq in test_sequences:
            pred = baselines.linear_regression_model(all_sequences, all_targets, test_seq)
            all_linear_preds.append(pred)
        
        linear_time = time.time() - start_time
        
        # Calculate AP with detailed analysis
        linear_ap, linear_analysis = calculate_fair_ap_with_analysis(
            all_linear_preds, test_targets, "Enhanced Linear"
        )
            
        results['Enhanced Linear'] = {
            'predictions': all_linear_preds,
            'ap_score': linear_ap,
            'time': linear_time,
            'analysis': linear_analysis
        }
        print(f"  ✓ Time: {linear_time:.4f}s")
    except Exception as e:
        print(f"  ✗ Enhanced Linear failed: {e}")
        results['Enhanced Linear'] = None
    
    # 3. Enhanced Simple CNN
    print("\n3. ENHANCED SIMPLE CNN")
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
            all_cnn_preds, test_targets, "Enhanced CNN"
        )
            
        results['Enhanced CNN'] = {
            'predictions': all_cnn_preds,
            'ap_score': cnn_ap,
            'time': cnn_time,
            'analysis': cnn_analysis
        }
        print(f"  ✓ Time: {cnn_time:.2f}s")
    except Exception as e:
        print(f"  ✗ Enhanced CNN failed: {e}")
        results['Enhanced CNN'] = None
    
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
