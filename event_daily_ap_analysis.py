#!/usr/bin/env python3
"""
分析单个火灾事件内部每天AP计算的关键问题
用户担心：无火天AP=0会拉低有火天的真实准确度
"""

import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def analyze_daily_ap_within_event():
    """
    分析单个火灾事件内部每天的AP计算问题
    """
    print("🔥 单个火灾事件内部每天AP计算分析")
    print("="*60)
    
    print("📋 用户的核心担心:")
    print("-" * 40)
    print("• 一个火灾事件有26天")
    print("• 其中只有8天有火，18天无火")
    print("• 有火天：计算真实AP (比如0.3)")
    print("• 无火天：AP设为0.0")
    print("• 最终：18个0值拉低了8个真实AP值")
    print("• 结果：最终AP被严重低估")
    
    # 模拟一个真实的火灾事件
    print(f"\n🧪 模拟真实火灾事件的AP计算:")
    print("-" * 40)
    
    # 火灾事件参数
    total_days = 26
    fire_days = 8  # 前8天有火
    no_fire_days = 18  # 后18天无火
    
    print(f"火灾事件设定:")
    print(f"• 总天数: {total_days}")
    print(f"• 有火天数: {fire_days}")
    print(f"• 无火天数: {no_fire_days}")
    
    # 模拟每天的AP值
    daily_aps = []
    daily_info = []
    
    # 有火天的AP
    for day in range(fire_days):
        # 模拟真实的AP值 (有火天模型表现较好)
        true_ap = 0.25 + np.random.random() * 0.15  # 0.25-0.40之间
        daily_aps.append(true_ap)
        daily_info.append(f"Day {day+1}: 有火 -> AP = {true_ap:.4f}")
    
    # 无火天的AP
    for day in range(fire_days, total_days):
        # 根据当前代码逻辑，无火天AP=0.0
        false_ap = 0.0
        daily_aps.append(false_ap)
        daily_info.append(f"Day {day+1}: 无火 -> AP = {false_ap:.4f} (设为0)")
    
    # 显示每天的AP
    print(f"\n📊 每天的AP值:")
    for info in daily_info:
        print(f"  {info}")
    
    # 计算不同方式的最终AP
    method1_ap = np.mean(daily_aps)  # 包含0值的平均
    method2_ap = np.mean([ap for ap in daily_aps if ap > 0])  # 只计算有火天
    
    print(f"\n📈 不同计算方式的结果:")
    print(f"方法1 (包含无火天0值): {method1_ap:.4f}")
    print(f"方法2 (只计算有火天):   {method2_ap:.4f}")
    print(f"差异: {method2_ap - method1_ap:.4f} ({(method2_ap/method1_ap - 1)*100:.1f}%)")
    
    return daily_aps, method1_ap, method2_ap

def analyze_current_implementation():
    """
    分析当前代码实现是否真的是按天计算AP
    """
    print(f"\n🔍 分析当前代码的真实实现方式:")
    print("="*50)
    
    print("关键问题：代码是按天计算AP还是按事件计算AP？")
    print("-" * 40)
    
    print("从代码分析得出：")
    print("1. **训练过程 (test_with_stats.py)**:")
    print("   • 每个epoch收集所有batch的数据")
    print("   • 合并：all_predictions, all_targets")
    print("   • 计算：一个总体AP值")
    print("   • 结论：不是按天计算")
    
    print("\n2. **基线对比 (quick_baselines.py)**:")
    print("   • 使用单天数据测试")
    print("   • 计算：单天的AP值")
    print("   • 结论：是按天计算")
    
    print("\n3. **特征敏感性分析**:")
    print("   • 可能按天计算，然后某种方式聚合")
    print("   • 需要进一步确认")
    
    print(f"\n🎯 关键发现:")
    print("-" * 20)
    print("• 不同脚本可能使用不同的AP计算方式")
    print("• 训练：总体AP (所有数据合并)")
    print("• 测试：可能是每天AP的平均")
    print("• 用户的担心在测试阶段可能是对的！")

def simulate_different_ap_calculation_methods():
    """
    模拟不同AP计算方法的影响
    """
    print(f"\n🧪 模拟不同AP计算方法的影响:")
    print("="*50)
    
    # 创建一个火灾事件的数据
    spatial_size = 128 * 128
    days = 26
    
    all_daily_data = []
    
    for day in range(days):
        if day < 8:  # 有火天
            # 创建有火的数据
            fire_pixels = max(50, int(spatial_size * 0.002 * (1 - day/10)))
            y_true = np.zeros(spatial_size)
            fire_indices = np.random.choice(spatial_size, fire_pixels, replace=False)
            y_true[fire_indices] = 1
            
            # 模型预测 (有一定准确性)
            y_pred = np.random.random(spatial_size) * 0.1
            y_pred[fire_indices] += np.random.random(fire_pixels) * 0.4 + 0.2
            
            all_daily_data.append((y_true, y_pred, True, fire_pixels))
        else:  # 无火天
            # 创建无火的数据
            y_true = np.zeros(spatial_size)
            y_pred = np.random.random(spatial_size) * 0.05  # 模型预测很小的值
            
            all_daily_data.append((y_true, y_pred, False, 0))
    
    print("模拟火灾事件数据生成完成")
    
    # 方法1：每天计算AP，然后平均 (包含0值)
    daily_aps_with_zeros = []
    for day, (y_true, y_pred, has_fire, fire_pixels) in enumerate(all_daily_data):
        if has_fire and fire_pixels > 0:
            ap = average_precision_score(y_true, y_pred)
            daily_aps_with_zeros.append(ap)
            print(f"Day {day+1}: AP = {ap:.4f} (有火)")
        else:
            ap = 0.0  # 按当前代码逻辑
            daily_aps_with_zeros.append(ap)
            print(f"Day {day+1}: AP = {ap:.4f} (无火，设为0)")
    
    method1_result = np.mean(daily_aps_with_zeros)
    
    # 方法2：每天计算AP，但跳过无火天
    daily_aps_skip_zeros = []
    for day, (y_true, y_pred, has_fire, fire_pixels) in enumerate(all_daily_data):
        if has_fire and fire_pixels > 0:
            ap = average_precision_score(y_true, y_pred)
            daily_aps_skip_zeros.append(ap)
    
    method2_result = np.mean(daily_aps_skip_zeros) if daily_aps_skip_zeros else 0.0
    
    # 方法3：所有天数据合并，计算总体AP
    all_true = np.concatenate([data[0] for data in all_daily_data])
    all_pred = np.concatenate([data[1] for data in all_daily_data])
    method3_result = average_precision_score(all_true, all_pred)
    
    print(f"\n📊 三种方法的AP结果对比:")
    print(f"方法1 (每天AP平均，包含0): {method1_result:.4f}")
    print(f"方法2 (每天AP平均，跳过0): {method2_result:.4f}")  
    print(f"方法3 (所有数据合并):     {method3_result:.4f}")
    
    print(f"\n💡 差异分析:")
    print(f"方法1 vs 方法2: {method2_result - method1_result:.4f} ({(method2_result/method1_result - 1)*100:.1f}%)")
    print(f"方法1 vs 方法3: {method3_result - method1_result:.4f} ({(method3_result/method1_result - 1)*100:.1f}%)")
    print(f"方法2 vs 方法3: {method3_result - method2_result:.4f}")
    
    return method1_result, method2_result, method3_result

def main():
    """主函数"""
    analyze_daily_ap_within_event()
    analyze_current_implementation()
    simulate_different_ap_calculation_methods()
    
    print(f"\n🎯 回答用户的核心担心:")
    print("="*60)
    print("问题：无火天的0值会拉低有火天的真实准确度吗？")
    print("")
    print("答案：**是的，如果按天计算AP然后平均的话！**")
    print("")
    print("🔍 具体分析:")
    print("1. **如果是按天计算AP然后平均**:")
    print("   • 无火天AP=0.0会严重拉低最终结果")
    print("   • 26天中18天是0，会让最终AP降低60-70%")
    print("   • 这确实是一个严重的评估偏差")
    print("")
    print("2. **如果是所有数据合并计算总体AP**:")
    print("   • 无火天作为负样本参与计算")
    print("   • 这样计算更公平，不会被0值拉低")
    print("   • 但仍然会因为大量负样本而降低AP")
    print("")
    print("3. **我们需要确认项目中实际使用哪种方法**:")
    print("   • 训练过程：很可能是方法2 (总体AP)")
    print("   • 测试评估：可能是方法1 (每天平均)")
    print("   • 特征敏感性：需要进一步确认")
    print("")
    print("🚨 **你的担心完全正确！**")
    print("这是火灾预测评估中的一个重要问题！")

if __name__ == "__main__":
    main()
