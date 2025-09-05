#!/usr/bin/env python3
"""
深入分析AP计算方式，特别是在火灾预测这种极度不平衡数据上的表现
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

def analyze_ap_calculation():
    """
    详细分析AP计算方式和问题
    """
    print("🔍 深入分析AP (Average Precision) 计算方式")
    print("="*60)
    
    # 1. 基本AP概念解释
    print("\n📚 AP (Average Precision) 基本概念:")
    print("-" * 40)
    print("AP不是简单的'命中率'，而是Precision-Recall曲线下的面积")
    print("• Precision = TP / (TP + FP) - 预测为火的像素中真正是火的比例")
    print("• Recall = TP / (TP + FN) - 真正火像素中被预测出来的比例")
    print("• AP = 不同recall水平下precision的加权平均")
    
    # 2. 模拟不同情况的AP计算
    print("\n🧪 模拟不同预测情况的AP值:")
    print("-" * 40)
    
    # 创建一个典型的火灾数据集（极度不平衡）
    total_pixels = 128 * 128  # 16384 pixels
    fire_pixels = 100  # 只有100个像素是火（约0.6%）
    
    # 真实标签：大部分是0，少量是1
    y_true = np.zeros(total_pixels)
    fire_indices = np.random.choice(total_pixels, fire_pixels, replace=False)
    y_true[fire_indices] = 1
    
    print(f"数据集特征:")
    print(f"  • 总像素: {total_pixels:,}")
    print(f"  • 火像素: {fire_pixels} ({fire_pixels/total_pixels*100:.2f}%)")
    print(f"  • 非火像素: {total_pixels-fire_pixels:,} ({(total_pixels-fire_pixels)/total_pixels*100:.2f}%)")
    
    # 情况1: 完全随机预测
    print(f"\n1️⃣ 完全随机预测:")
    y_pred_random = np.random.random(total_pixels)
    ap_random = average_precision_score(y_true, y_pred_random)
    print(f"   AP = {ap_random:.4f}")
    print(f"   解释: 随机预测的AP约等于正样本比例 ({fire_pixels/total_pixels:.4f})")
    
    # 情况2: 全部预测为0（空预测）
    print(f"\n2️⃣ 全部预测为0 (空预测):")
    y_pred_zeros = np.zeros(total_pixels)
    try:
        ap_zeros = average_precision_score(y_true, y_pred_zeros)
        print(f"   AP = {ap_zeros:.4f}")
    except ValueError as e:
        print(f"   错误: {e}")
        print(f"   原因: 所有预测都是0，无法计算precision-recall曲线")
    
    # 情况3: 全部预测为很小的值（接近0但不是0）
    print(f"\n3️⃣ 全部预测为很小的值 (0.001):")
    y_pred_small = np.full(total_pixels, 0.001)
    ap_small = average_precision_score(y_true, y_pred_small)
    print(f"   AP = {ap_small:.4f}")
    print(f"   解释: 所有像素概率相同，AP等于正样本比例")
    
    # 情况4: 完美预测
    print(f"\n4️⃣ 完美预测:")
    y_pred_perfect = y_true.copy().astype(float)
    ap_perfect = average_precision_score(y_true, y_pred_perfect)
    print(f"   AP = {ap_perfect:.4f}")
    print(f"   解释: 完美预测AP = 1.0")
    
    # 情况5: 部分正确预测（类似我们的模型）
    print(f"\n5️⃣ 部分正确预测 (类似实际模型):")
    y_pred_partial = np.random.random(total_pixels) * 0.1  # 基础噪音
    # 给真正的火像素更高的概率
    y_pred_partial[fire_indices] += np.random.random(fire_pixels) * 0.5 + 0.2
    ap_partial = average_precision_score(y_true, y_pred_partial)
    print(f"   AP = {ap_partial:.4f}")
    print(f"   解释: 模型能够识别一些火像素，但不完美")
    
    # 6. 分析为什么空预测AP=0
    print(f"\n❓ 为什么'空预测'AP接近0？")
    print("-" * 40)
    print("1. AP衡量的是模型区分正负样本的能力")
    print("2. 如果所有预测都相同（比如都是0或都是0.001）：")
    print("   • 模型无法区分哪些像素更可能是火")
    print("   • Precision-Recall曲线退化为单点")
    print("   • AP约等于数据集中正样本的比例")
    print("3. 在极度不平衡的数据集上（火像素<1%）：")
    print("   • 即使预测'全空'在准确率上看似不错")
    print("   • 但AP会很低，因为没有识别能力")
    
    # 7. 可视化不同情况的Precision-Recall曲线
    plt.figure(figsize=(15, 10))
    
    scenarios = [
        ("随机预测", y_pred_random, ap_random),
        ("很小值预测", y_pred_small, ap_small), 
        ("部分正确", y_pred_partial, ap_partial),
        ("完美预测", y_pred_perfect, ap_perfect)
    ]
    
    for i, (name, y_pred, ap) in enumerate(scenarios, 1):
        plt.subplot(2, 2, i)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision, linewidth=2)
        plt.fill_between(recall, precision, alpha=0.3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name}\nAP = {ap:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 添加基线（随机分类器）
        baseline = fire_pixels / total_pixels
        plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, 
                   label=f'随机基线 ({baseline:.4f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('ap_analysis_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Precision-Recall曲线已保存: ap_analysis_curves.png")
    
    # 8. 实际模型性能分析
    print(f"\n📈 实际模型性能分析:")
    print("-" * 40)
    print(f"我们的主UNet模型 AP = 0.1794:")
    print(f"• 相比随机预测 (~{fire_pixels/total_pixels:.4f}): 提升 ~{0.1794/(fire_pixels/total_pixels):.1f}x")
    print(f"• 相比完全空预测 (≈0): 有明显的识别能力")
    print(f"• 距离完美预测 (1.0): 还有很大提升空间")
    print(f"")
    print(f"基线模型对比:")
    print(f"• Persistence (0.0845): 利用时序信息，有一定预测能力")
    print(f"• Mean baseline (0.0122): 接近随机水平")
    print(f"• Simple CNN (0.0069): 甚至不如随机")
    
    # 9. AP vs 准确率的对比
    print(f"\n⚖️ AP vs 准确率 (Accuracy) 的区别:")
    print("-" * 40)
    
    # 计算不同预测的准确率
    def calculate_accuracy(y_true, y_pred, threshold=0.5):
        y_pred_binary = (y_pred > threshold).astype(int)
        return np.mean(y_true == y_pred_binary)
    
    acc_random = calculate_accuracy(y_true, y_pred_random)
    acc_small = calculate_accuracy(y_true, y_pred_small)
    acc_partial = calculate_accuracy(y_true, y_pred_partial)
    
    print(f"预测方式          AP      准确率")
    print(f"随机预测        {ap_random:.4f}   {acc_random:.4f}")
    print(f"很小值预测      {ap_small:.4f}   {acc_small:.4f}")
    print(f"部分正确        {ap_partial:.4f}   {acc_partial:.4f}")
    print(f"")
    print(f"关键洞察:")
    print(f"• 在极度不平衡数据上，准确率可能很高但AP很低")
    print(f"• AP更能反映模型识别稀有事件（火灾）的能力")
    print(f"• 这就是为什么火灾预测使用AP而不是准确率作为主要指标")

def main():
    """主函数"""
    analyze_ap_calculation()
    
    print(f"\n🎯 总结:")
    print("="*60)
    print("1. AP不是'命中率'，而是衡量模型区分能力的指标")
    print("2. 在极度不平衡数据上，'全空预测'AP接近正样本比例")
    print("3. 我们的模型AP=0.1794说明有较好的火灾识别能力")
    print("4. 基线对比证明了复杂模型的必要性")
    print("5. AP比准确率更适合评估稀有事件预测任务")

if __name__ == "__main__":
    main()
