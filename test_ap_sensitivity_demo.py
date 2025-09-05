#!/usr/bin/env python3
"""
测试 simple_feature_sensitivity.py 的AP分析功能
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from simple_feature_sensitivity import (
    analyze_fire_no_fire_distribution, 
    calculate_comprehensive_ap_analysis,
    create_feature_ap_summary
)

def create_demo_data():
    """创建演示数据"""
    print("🔍 Creating demo data for AP analysis...")
    
    # 创建26天的地面真实数据（模拟火灾事件）
    ground_truth = []
    baseline_predictions = []
    perturbation_predictions = {-0.4: [], -0.2: [], 0.2: [], 0.4: []}
    
    for day in range(26):
        # 地面真实数据：前10天有火，逐渐减少
        gt = np.zeros((128, 128))
        if day < 10:
            fire_intensity = max(0.1, 1.0 - day * 0.1)
            num_fire_pixels = int(500 * fire_intensity)
            fire_indices = np.random.choice(128*128, num_fire_pixels, replace=False)
            gt.flat[fire_indices] = 1.0
        
        ground_truth.append(gt)
        
        # 基线预测：有一定准确性
        baseline_pred = np.zeros((128, 128))
        if day < 10:
            # 70%准确率
            correct_pixels = int(num_fire_pixels * 0.7)
            baseline_pred.flat[fire_indices[:correct_pixels]] = 0.8
            # 添加一些误报
            false_positive = np.random.choice(128*128, num_fire_pixels//4, replace=False)
            baseline_pred.flat[false_positive] = 0.3
        else:
            # 无火天的小预测值
            baseline_pred = np.random.random((128, 128)) * 0.05
        
        baseline_predictions.append(baseline_pred)
        
        # 扰动预测：模拟不同特征变化的影响
        for perturbation in [-0.4, -0.2, 0.2, 0.4]:
            if day < 10:
                # 扰动影响预测准确性
                accuracy_modifier = 1.0 + perturbation * 0.5  # -40%到+40%的扰动
                accuracy = max(0.1, min(0.95, 0.7 * accuracy_modifier))
                
                perturb_pred = np.zeros((128, 128))
                correct_pixels = int(num_fire_pixels * accuracy)
                perturb_pred.flat[fire_indices[:correct_pixels]] = 0.8
                
                # 误报也受扰动影响
                false_positive_rate = max(0.05, min(0.4, 0.25 * (1 - perturbation)))
                false_positive = np.random.choice(128*128, int(num_fire_pixels * false_positive_rate), replace=False)
                perturb_pred.flat[false_positive] = 0.3
            else:
                # 无火天
                perturb_pred = np.random.random((128, 128)) * 0.05
            
            perturbation_predictions[perturbation].append(perturb_pred)
    
    return ground_truth, baseline_predictions, perturbation_predictions

def main():
    """主测试函数"""
    print("🔥 SIMPLE FEATURE SENSITIVITY AP ANALYSIS DEMO")
    print("=" * 60)
    
    # 创建演示数据
    ground_truth, baseline_predictions, perturbation_predictions = create_demo_data()
    
    # 测试分析函数
    print("\n📊 Testing AP analysis functions...")
    
    # 1. 测试火天/无火天分布分析
    fire_days, no_fire_days = analyze_fire_no_fire_distribution(ground_truth)
    print(f"✅ Fire/No-fire analysis: {len(fire_days)} fire days, {len(no_fire_days)} no-fire days")
    
    # 2. 测试单个场景的AP分析
    baseline_analysis = calculate_comprehensive_ap_analysis(
        baseline_predictions, ground_truth, "Baseline Test"
    )
    print(f"✅ Baseline AP analysis: Combined={baseline_analysis['combined_ap']:.4f}")
    
    # 3. 测试完整的特征AP总结
    perturbation_levels = [-0.4, -0.2, 0.0, 0.2, 0.4]
    
    # 添加baseline到perturbation_predictions
    perturbation_predictions[0.0] = baseline_predictions
    
    print("\n🎯 Creating comprehensive feature AP summary...")
    ap_results = create_feature_ap_summary(
        "DEMO_FEATURE", 
        "demo_ap_output",
        ground_truth,
        baseline_predictions,
        perturbation_predictions,
        perturbation_levels
    )
    
    print("\n✅ AP Analysis Demo Completed!")
    print(f"📁 Results saved in: demo_ap_output/")
    print("📊 Generated files:")
    print("   - DEMO_FEATURE_AP_Summary.txt")
    print("   - DEMO_FEATURE_AP_Analysis.json")
    
    print("\n💡 This demonstrates the AP analysis that will be added to each feature in simple_feature_sensitivity.py!")

if __name__ == "__main__":
    main()
