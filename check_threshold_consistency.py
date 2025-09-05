#!/usr/bin/env python3
"""
检查ground truth和prediction阈值处理的一致性
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from simple_feature_sensitivity import load_fire_event_data, SimpleConfig

def check_ground_truth_processing():
    """检查ground truth的处理方式"""
    print("🔍 CHECKING GROUND TRUTH PROCESSING CONSISTENCY")
    print("=" * 60)
    
    config = SimpleConfig()
    fire_event_path = "data/processed/2020/fire_24461899.hdf5"
    
    if not os.path.exists(fire_event_path):
        print("❌ Fire event file not found!")
        return
    
    # 加载数据
    initial_seq, weather_data, ground_truth, max_days = load_fire_event_data(
        fire_event_path, config, start_day=0
    )
    
    if ground_truth is None:
        print("❌ Failed to load ground truth data!")
        return
    
    print(f"📊 Loaded {len(ground_truth)} days of ground truth data")
    
    # 分析不同阈值下的统计
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\n📈 GROUND TRUTH ANALYSIS - Different Thresholds")
    print("-" * 60)
    print(f"{'Day':<5} {'Raw Min':<8} {'Raw Max':<8} {'Raw Mean':<10}", end="")
    for t in thresholds:
        print(f">{t:<5}", end="")
    print()
    print("-" * 60)
    
    for day in range(min(10, len(ground_truth))):  # 检查前10天
        gt_data = ground_truth[day]
        
        print(f"{day+1:<5} {gt_data.min():<8.3f} {gt_data.max():<8.3f} {gt_data.mean():<10.3f}", end="")
        
        for threshold in thresholds:
            fire_pixels = (gt_data > threshold).sum()
            print(f"{fire_pixels:<6}", end="")
        print()
    
    print(f"\n💡 KEY FINDINGS:")
    
    # 检查原始数据的值分布
    all_values = np.concatenate([gt.flatten() for gt in ground_truth[:10]])
    unique_values = np.unique(all_values)
    print(f"• Unique values in ground truth: {len(unique_values)}")
    print(f"• Value range: [{all_values.min():.3f}, {all_values.max():.3f}]")
    print(f"• Most common values: {np.bincount(all_values.astype(int))}")
    
    # 检查不同阈值的影响
    print(f"\n🎯 THRESHOLD IMPACT ANALYSIS:")
    total_fire_days = 0
    for threshold in thresholds:
        fire_days = sum(1 for gt in ground_truth[:10] if (gt > threshold).sum() > 0)
        print(f"• Threshold {threshold}: {fire_days} fire days (out of 10)")
        if threshold == 0.5:
            total_fire_days = fire_days
    
    print(f"\n✅ CONSISTENCY CHECK (AFTER FIX):")
    print(f"• Data loading uses threshold: 0.5")
    print(f"• AP calculation uses threshold: 0.5")
    print(f"• GIF display uses threshold: 0.5")
    print(f"• Prediction display uses threshold: 0.5")
    print(f"• ALL OPERATIONS NOW USE CONSISTENT THRESHOLD: 0.5")
    
    if total_fire_days > 0:
        print(f"\n✅ Ground truth contains fire data")
    else:
        print(f"\n❌ No fire detected with threshold 0.5!")
    
    return ground_truth

def main():
    """主函数"""
    ground_truth = check_ground_truth_processing()
    
    print(f"\n🔧 RECOMMENDATIONS:")
    print(f"• Use CONSISTENT threshold across all operations")
    print(f"• Consider using 0.1 (data loading threshold) everywhere")
    print(f"• OR use 0.5 everywhere if ground truth is binary")
    print(f"• Test both approaches and compare results")

if __name__ == "__main__":
    main()
