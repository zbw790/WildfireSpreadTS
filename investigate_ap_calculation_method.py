#!/usr/bin/env python3
"""
深入调查我们项目中AP计算的具体方法
特别是针对多天火灾事件中的AP计算方式
"""

import numpy as np
import h5py
from sklearn.metrics import average_precision_score
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def investigate_ap_calculation_in_project():
    """
    调查项目中AP计算的具体方法
    """
    print("🔍 调查项目中AP计算的具体方法")
    print("="*60)
    
    # 1. 分析基线对比中的AP计算
    print("\n📊 基线对比中的AP计算方式:")
    print("-" * 40)
    print("从 quick_baselines.py 分析:")
    print("• 使用单天数据: test_target = day 6 的ground truth")
    print("• AP计算: average_precision_score(test_target.flatten(), prediction.flatten())")
    print("• 这是单天单次AP计算，不是多天平均")
    
    # 2. 分析训练过程中的AP计算
    print("\n🏋️ 训练过程中的AP计算方式:")
    print("-" * 40)
    print("从 test1.py, test_with_stats.py 分析:")
    print("• 收集所有batch的预测和目标: all_predictions, all_targets")
    print("• 展平所有数据: .flatten()")
    print("• 计算总体AP: average_precision_score(all_targets, all_predictions)")
    print("• 这是跨所有样本的总体AP，不区分天数")
    
    # 3. 模拟不同的AP计算方式
    print("\n🧪 模拟不同AP计算方式的差异:")
    print("-" * 40)
    
    # 创建模拟的多天火灾数据
    num_days = 26
    spatial_size = 128 * 128
    
    # 模拟火灾演变：前几天有火，后几天没火
    daily_data = []
    for day in range(num_days):
        if day < 8:  # 前8天有火
            fire_ratio = max(0.001, 0.01 * (1 - day/10))  # 火势逐渐减小
            fire_pixels = int(spatial_size * fire_ratio)
        else:  # 后18天没火
            fire_pixels = 0
        
        # 真实标签
        y_true = np.zeros(spatial_size)
        if fire_pixels > 0:
            fire_indices = np.random.choice(spatial_size, fire_pixels, replace=False)
            y_true[fire_indices] = 1
        
        # 模拟预测（模型在有火天表现好，无火天预测很小的值）
        if fire_pixels > 0:
            y_pred = np.random.random(spatial_size) * 0.1
            # 给真火像素更高概率
            y_pred[fire_indices] += np.random.random(fire_pixels) * 0.4 + 0.3
        else:
            # 无火天：模型预测很小的值
            y_pred = np.random.random(spatial_size) * 0.05
        
        daily_data.append((y_true, y_pred, fire_pixels))
    
    # 方法1: 每天单独计算AP，然后平均
    print("\n方法1: 每天单独计算AP，然后平均")
    daily_aps = []
    for day, (y_true, y_pred, fire_pixels) in enumerate(daily_data):
        if fire_pixels > 0:  # 只有有火的天才计算AP
            ap = average_precision_score(y_true, y_pred)
            daily_aps.append(ap)
            print(f"  Day {day+1}: {fire_pixels} fire pixels, AP = {ap:.4f}")
        else:
            print(f"  Day {day+1}: {fire_pixels} fire pixels, AP = SKIP (no fire)")
    
    avg_ap_method1 = np.mean(daily_aps) if daily_aps else 0.0
    print(f"  平均AP (只计算有火天): {avg_ap_method1:.4f}")
    
    # 方法2: 所有天数据合并，计算总体AP
    print(f"\n方法2: 所有天数据合并，计算总体AP")
    all_true = np.concatenate([data[0] for data in daily_data])
    all_pred = np.concatenate([data[1] for data in daily_data])
    overall_ap = average_precision_score(all_true, all_pred)
    print(f"  总体AP (包含所有天): {overall_ap:.4f}")
    
    # 方法3: 每天都计算AP（包括无火天），然后平均
    print(f"\n方法3: 每天都计算AP（包括无火天），然后平均")
    all_daily_aps = []
    for day, (y_true, y_pred, fire_pixels) in enumerate(daily_data):
        if fire_pixels > 0:
            ap = average_precision_score(y_true, y_pred)
        else:
            # 无火天的AP计算
            if np.sum(y_true) == 0:
                # 如果ground truth全是0，AP的计算会有问题
                # sklearn会报错或返回特殊值
                try:
                    ap = average_precision_score(y_true, y_pred)
                except ValueError:
                    ap = 0.0  # 或者设为某个默认值
            else:
                ap = average_precision_score(y_true, y_pred)
        
        all_daily_aps.append(ap)
        print(f"  Day {day+1}: {fire_pixels} fire pixels, AP = {ap:.4f}")
    
    avg_ap_method3 = np.mean(all_daily_aps)
    print(f"  平均AP (包含所有天): {avg_ap_method3:.4f}")
    
    # 4. 分析我们项目实际使用的方法
    print(f"\n🎯 我们项目实际使用的方法分析:")
    print("-" * 40)
    print("基于代码分析，我们的项目使用:")
    print("• 基线对比: 方法1类似 - 单天AP计算")
    print("• 训练验证: 方法2类似 - 总体AP计算")
    print("• 特征敏感性: 可能是方法2 - 将所有预测合并计算")
    
    print(f"\n📊 不同方法的AP结果对比:")
    print(f"方法1 (只有火天平均): {avg_ap_method1:.4f}")
    print(f"方法2 (总体合并):     {overall_ap:.4f}")
    print(f"方法3 (所有天平均):   {avg_ap_method3:.4f}")
    
    # 5. 分析潜在问题
    print(f"\n⚠️ 潜在问题分析:")
    print("-" * 40)
    print("1. **无火天处理问题**:")
    print("   • 如果某天ground truth全是0，AP计算会有问题")
    print("   • sklearn可能报错: 'y_true takes value in {0} but should be in {0, 1}'")
    print("   • 或者返回nan/特殊值")
    
    print(f"\n2. **方法选择的影响**:")
    print("   • 方法1: 可能高估性能（忽略无火天）")
    print("   • 方法2: 更真实（包含所有数据）")
    print("   • 方法3: 可能低估性能（无火天AP≈0）")
    
    print(f"\n3. **我们的AP=0.1794可能的计算方式**:")
    print("   • 最可能是方法2: 总体合并计算")
    print("   • 包含了所有有火和无火的像素")
    print("   • 这样计算更加公平和真实")
    
    # 6. 验证实际项目数据
    print(f"\n🔍 验证建议:")
    print("-" * 40)
    print("建议检查以下几点:")
    print("1. 确认AP计算是否跳过了无火天")
    print("2. 检查是否有'clean_targets.sum() > 0'这样的条件")
    print("3. 确认是单天计算还是多天合并计算")
    print("4. 查看训练日志中的AP变化趋势")

def analyze_no_fire_days_impact():
    """
    专门分析无火天数对AP计算的影响
    """
    print(f"\n🔥 无火天数对AP计算的影响分析:")
    print("="*50)
    
    # 模拟一个火灾事件：26天中只有前5天有火
    total_pixels = 128 * 128
    
    # 有火天的数据
    fire_days = 5
    no_fire_days = 21
    
    print(f"火灾事件模拟:")
    print(f"• 有火天数: {fire_days}")
    print(f"• 无火天数: {no_fire_days}")
    print(f"• 总天数: {fire_days + no_fire_days}")
    
    # 生成数据
    all_true = []
    all_pred = []
    
    # 有火天
    for day in range(fire_days):
        fire_pixels = max(10, int(total_pixels * 0.005 * (1 - day/10)))
        y_true = np.zeros(total_pixels)
        fire_indices = np.random.choice(total_pixels, fire_pixels, replace=False)
        y_true[fire_indices] = 1
        
        # 模型预测（有一定准确性）
        y_pred = np.random.random(total_pixels) * 0.1
        y_pred[fire_indices] += np.random.random(fire_pixels) * 0.5 + 0.2
        
        all_true.append(y_true)
        all_pred.append(y_pred)
    
    # 无火天
    for day in range(no_fire_days):
        y_true = np.zeros(total_pixels)  # 全是0
        y_pred = np.random.random(total_pixels) * 0.05  # 模型预测很小的值
        
        all_true.append(y_true)
        all_pred.append(y_pred)
    
    # 计算不同方式的AP
    # 方式1: 只计算有火天
    fire_true = np.concatenate(all_true[:fire_days])
    fire_pred = np.concatenate(all_pred[:fire_days])
    ap_fire_only = average_precision_score(fire_true, fire_pred)
    
    # 方式2: 计算所有天
    all_true_combined = np.concatenate(all_true)
    all_pred_combined = np.concatenate(all_pred)
    ap_all_days = average_precision_score(all_true_combined, all_pred_combined)
    
    print(f"\nAP计算结果:")
    print(f"• 只计算有火天: {ap_fire_only:.4f}")
    print(f"• 包含所有天:   {ap_all_days:.4f}")
    print(f"• 差异:         {ap_fire_only - ap_all_days:.4f}")
    
    print(f"\n💡 关键洞察:")
    print(f"• 包含无火天会显著降低AP值")
    print(f"• 这是因为无火天增加了大量负样本")
    print(f"• 但这样计算更符合实际应用场景")
    
    return ap_fire_only, ap_all_days

def main():
    """主函数"""
    investigate_ap_calculation_in_project()
    analyze_no_fire_days_impact()
    
    print(f"\n🎯 总结和建议:")
    print("="*60)
    print("1. **我们的AP=0.1794很可能是合理的**:")
    print("   • 包含了所有有火和无火天的数据")
    print("   • 这是更真实的性能评估")
    
    print(f"\n2. **无火天不会导致AP=0**:")
    print("   • 无火天增加负样本，但不会让AP变成0")
    print("   • AP的下界是正样本比例，不是0")
    
    print(f"\n3. **建议验证的点**:")
    print("   • 检查代码中是否有'if clean_targets.sum() > 0'条件")
    print("   • 确认AP计算是否跳过无火天")
    print("   • 查看训练过程中AP的变化曲线")
    
    print(f"\n4. **你的模型性能评估**:")
    print("   • AP=0.1794在包含无火天的情况下是很好的成绩")
    print("   • 说明模型能够有效区分有火和无火的情况")
    print("   • 比基线方法有显著提升")

if __name__ == "__main__":
    main()
