#!/usr/bin/env python3
"""
测试Ground Truth连续值显示功能
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from simple_feature_sensitivity import (
    load_fire_event_data, 
    SimpleConfig,
    create_enhanced_feature_sensitivity_gif
)

def test_ground_truth_display():
    """测试ground truth的连续值显示"""
    print("🔥 TESTING CONTINUOUS GROUND TRUTH DISPLAY")
    print("=" * 60)
    
    config = SimpleConfig()
    fire_event_path = "data/processed/2020/fire_24461899.hdf5"
    
    if not os.path.exists(fire_event_path):
        print("❌ Fire event file not found!")
        return
    
    # 测试修改后的数据加载函数
    print("📊 Testing modified load_fire_event_data function...")
    initial_seq, weather_data, ground_truth, ground_truth_raw, max_days = load_fire_event_data(
        fire_event_path, config, start_day=0
    )
    
    if ground_truth is None or ground_truth_raw is None:
        print("❌ Failed to load ground truth data!")
        return
    
    print(f"✅ Successfully loaded data:")
    print(f"   Ground truth (binary): {len(ground_truth)} days")
    print(f"   Ground truth (raw): {len(ground_truth_raw)} days")
    print(f"   Max days: {max_days}")
    
    # 比较原始值和二值化值
    print(f"\n📈 COMPARING RAW vs BINARY VALUES:")
    print("-" * 50)
    print(f"{'Day':<5} {'Raw Min':<8} {'Raw Max':<8} {'Raw Mean':<10} {'Binary Sum':<10}")
    print("-" * 50)
    
    for day in range(min(10, len(ground_truth))):
        raw_data = ground_truth_raw[day]
        binary_data = ground_truth[day]
        
        print(f"{day+1:<5} {raw_data.min():<8.3f} {raw_data.max():<8.3f} {raw_data.mean():<10.3f} {binary_data.sum():<10.0f}")
    
    # 创建简单的测试预测数据
    print(f"\n🎬 Creating test GIF with continuous ground truth display...")
    
    # 创建模拟预测数据
    baseline_predictions = []
    perturbation_predictions = {-20: [], 0: [], 20: []}
    
    for day in range(min(5, len(ground_truth))):  # 只测试前5天
        # 基线预测
        baseline_pred = np.random.random((128, 128)) * 0.5
        baseline_predictions.append(baseline_pred)
        
        # 扰动预测
        for perturbation in [-20, 0, 20]:
            modifier = 1.0 + perturbation * 0.01
            perturb_pred = baseline_pred * modifier
            perturb_pred = np.clip(perturb_pred, 0, 1)
            perturbation_predictions[perturbation].append(perturb_pred)
    
    # 创建测试GIF
    success = create_enhanced_feature_sensitivity_gif(
        "TEST_CONTINUOUS_GT",
        "test_continuous_gt_output",
        ground_truth[:5],  # 只使用前5天
        ground_truth_raw[:5],  # 原始值
        baseline_predictions,
        perturbation_predictions,
        [-20, 0, 20]
    )
    
    if success:
        print("✅ Test GIF created successfully!")
        print("📁 Check: test_continuous_gt_output/TEST_CONTINUOUS_GT_enhanced_evolution.gif")
        print("\n💡 Key improvements:")
        print("   • Ground truth now shows continuous values with depth/intensity")
        print("   • Statistics show both binary count and raw value info")
        print("   • Visual consistency with prediction displays")
    else:
        print("❌ Test GIF creation failed!")

def main():
    """主测试函数"""
    test_ground_truth_display()
    
    print(f"\n🎯 SUMMARY:")
    print(f"• Ground truth now displays original continuous values")
    print(f"• Binary values still used for AP calculation (consistency)")
    print(f"• Enhanced statistics show both binary and continuous info")
    print(f"• Visual depth matching prediction displays")

if __name__ == "__main__":
    main()
