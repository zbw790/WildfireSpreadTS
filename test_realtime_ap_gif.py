#!/usr/bin/env python3
"""
测试GIF中的实时AP显示功能
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from simple_feature_sensitivity import (
    calculate_cumulative_ap,
    create_enhanced_feature_sensitivity_gif
)

def create_demo_data_for_gif():
    """创建演示数据用于测试GIF生成"""
    print("🔍 Creating demo data for real-time AP GIF...")
    
    # 创建10天的测试数据
    ground_truth = []
    baseline_predictions = []
    perturbation_predictions = {-40: [], -20: [], 0: [], 20: [], 40: [], 60: [], 80: []}
    
    for day in range(10):
        # 地面真实数据：前6天有火，逐渐减少
        gt = np.zeros((64, 64))  # 较小尺寸用于快速测试
        if day < 6:
            fire_intensity = max(0.1, 1.0 - day * 0.15)
            num_fire_pixels = int(200 * fire_intensity)
            fire_indices = np.random.choice(64*64, num_fire_pixels, replace=False)
            gt.flat[fire_indices] = 1.0
        
        ground_truth.append(gt)
        
        # 基线预测：70%准确率
        baseline_pred = np.zeros((64, 64))
        if day < 6:
            correct_pixels = int(num_fire_pixels * 0.7)
            baseline_pred.flat[fire_indices[:correct_pixels]] = 0.8
            # 添加误报
            false_positive = np.random.choice(64*64, num_fire_pixels//4, replace=False)
            baseline_pred.flat[false_positive] = 0.3
        else:
            baseline_pred = np.random.random((64, 64)) * 0.05
        
        baseline_predictions.append(baseline_pred)
        
        # 扰动预测：不同性能
        for perturbation in [-40, -20, 0, 20, 40, 60, 80]:
            if day < 6:
                # 扰动影响准确性
                accuracy_modifier = 1.0 + perturbation * 0.01  # -40%到+80%
                accuracy = max(0.1, min(0.95, 0.7 * accuracy_modifier))
                
                perturb_pred = np.zeros((64, 64))
                correct_pixels = int(num_fire_pixels * accuracy)
                if correct_pixels > 0:
                    perturb_pred.flat[fire_indices[:correct_pixels]] = 0.8
                
                # 误报率
                false_positive_rate = max(0.05, min(0.4, 0.25 * (1 - perturbation * 0.01)))
                false_positive = np.random.choice(64*64, int(num_fire_pixels * false_positive_rate), replace=False)
                perturb_pred.flat[false_positive] = 0.3
            else:
                perturb_pred = np.random.random((64, 64)) * 0.05
            
            perturbation_predictions[perturbation].append(perturb_pred)
    
    return ground_truth, baseline_predictions, perturbation_predictions

def test_cumulative_ap():
    """测试累积AP计算函数"""
    print("\n📊 Testing cumulative AP calculation...")
    
    # 创建简单测试数据
    predictions = [
        np.array([[0.8, 0.2], [0.1, 0.9]]),  # Day 1
        np.array([[0.7, 0.3], [0.2, 0.8]]),  # Day 2
        np.array([[0.1, 0.1], [0.1, 0.1]])   # Day 3 (no fire)
    ]
    
    targets = [
        np.array([[1, 0], [0, 1]]),  # Day 1: fire at (0,0) and (1,1)
        np.array([[1, 0], [0, 1]]),  # Day 2: same pattern
        np.array([[0, 0], [0, 0]])   # Day 3: no fire
    ]
    
    for day in range(3):
        ap = calculate_cumulative_ap(predictions, targets, day)
        print(f"  Day {day+1} cumulative AP: {ap:.4f}")
    
    print("✅ Cumulative AP calculation test completed!")

def main():
    """主测试函数"""
    print("🔥 REAL-TIME AP GIF DEMO")
    print("=" * 50)
    
    # 测试累积AP计算
    test_cumulative_ap()
    
    # 创建演示数据
    ground_truth, baseline_predictions, perturbation_predictions = create_demo_data_for_gif()
    
    # 添加baseline到perturbation_predictions
    perturbation_predictions[0] = baseline_predictions
    perturbation_levels = [-40, -20, 0, 20, 40, 60, 80]
    
    print(f"\n🎬 Creating enhanced GIF with real-time AP display...")
    print(f"   Ground truth days: {len(ground_truth)}")
    print(f"   Perturbation levels: {perturbation_levels}")
    
    # 创建GIF
    success = create_enhanced_feature_sensitivity_gif(
        "DEMO_REALTIME_AP",
        "demo_realtime_ap_output",
        ground_truth,
        baseline_predictions,
        perturbation_predictions,
        perturbation_levels
    )
    
    if success:
        print("\n✅ Real-time AP GIF Demo Completed!")
        print("📁 Results saved in: demo_realtime_ap_output/")
        print("🎬 Generated file: DEMO_REALTIME_AP_enhanced_evolution.gif")
        print("\n💡 Each subplot now shows:")
        print("   - Real-time cumulative AP (color-coded)")
        print("   - Prediction pixel count")
        print("   - Maximum prediction value")
        print("   - Ground truth fire statistics")
    else:
        print("❌ GIF creation failed!")

if __name__ == "__main__":
    main()
