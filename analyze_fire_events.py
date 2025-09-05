#!/usr/bin/env python3
"""
分析所有火灾事件，找出最具代表性和持续时间最长的事件
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import h5py
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_single_fire_event(fire_path):
    """分析单个火灾事件"""
    try:
        with h5py.File(fire_path, 'r') as f:
            # 尝试不同的数据集名称
            dataset_names = ['data', 'fire_data', 'dataset']
            data = None
            
            for name in dataset_names:
                if name in f:
                    data = f[name][:]
                    break
            
            if data is None:
                print(f"  ❌ No recognized dataset in {fire_path.name}")
                return None
            
            # 分析数据维度和火灾特征
            days, features, height, width = data.shape
            
            # 假设Active_Fire是最后一个特征
            fire_data = data[:, -1, :, :]  # 取最后一个特征作为火灾数据
            
            # 分析每天的火灾情况
            daily_fire_pixels = []
            daily_fire_area = []
            fire_days = 0
            max_fire_pixels = 0
            total_fire_area = 0
            
            for day in range(days):
                day_fire = fire_data[day]
                fire_pixels = (day_fire > 0.1).sum()  # 使用较低阈值检测火灾
                fire_area = fire_pixels / (height * width) * 100  # 火灾覆盖百分比
                
                daily_fire_pixels.append(fire_pixels)
                daily_fire_area.append(fire_area)
                
                if fire_pixels > 0:
                    fire_days += 1
                    max_fire_pixels = max(max_fire_pixels, fire_pixels)
                    total_fire_area += fire_area
            
            # 计算火灾持续性指标
            fire_intensity_variance = np.var(daily_fire_pixels)
            fire_spread_pattern = np.std(daily_fire_area)
            
            # 计算火灾演变模式
            if fire_days > 1:
                fire_trend = np.polyfit(range(len(daily_fire_pixels)), daily_fire_pixels, 1)[0]
            else:
                fire_trend = 0
            
            # 计算代表性评分
            representativeness_score = (
                fire_days * 0.3 +                    # 持续天数 (30%)
                (max_fire_pixels / 1000) * 0.25 +    # 最大火灾规模 (25%)
                (total_fire_area / fire_days if fire_days > 0 else 0) * 0.2 +  # 平均火灾强度 (20%)
                (fire_spread_pattern / 10) * 0.15 +  # 空间变化 (15%)
                min(abs(fire_trend) / 100, 1) * 0.1  # 时间演变 (10%)
            )
            
            return {
                'file_path': str(fire_path),
                'file_name': fire_path.name,
                'total_days': days,
                'fire_days': fire_days,
                'no_fire_days': days - fire_days,
                'fire_day_ratio': fire_days / days if days > 0 else 0,
                'max_fire_pixels': int(max_fire_pixels),
                'total_fire_area': total_fire_area,
                'avg_fire_area': total_fire_area / fire_days if fire_days > 0 else 0,
                'fire_intensity_variance': fire_intensity_variance,
                'fire_spread_pattern': fire_spread_pattern,
                'fire_trend': fire_trend,
                'representativeness_score': representativeness_score,
                'daily_fire_pixels': daily_fire_pixels,
                'daily_fire_area': daily_fire_area,
                'data_shape': list(data.shape)
            }
            
    except Exception as e:
        print(f"  ❌ Error analyzing {fire_path.name}: {e}")
        return None

def find_fire_events():
    """查找所有火灾事件文件"""
    fire_events = []
    
    # 搜索可能的数据目录
    search_paths = [
        Path("data/processed"),
        Path("data"),
        Path(".")
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            # 递归查找所有HDF5文件
            for fire_file in search_path.rglob("*.hdf5"):
                if "fire_" in fire_file.name.lower():
                    fire_events.append(fire_file)
            
            # 也查找.h5文件
            for fire_file in search_path.rglob("*.h5"):
                if "fire_" in fire_file.name.lower():
                    fire_events.append(fire_file)
    
    return sorted(fire_events)

def create_analysis_report(results, output_dir="fire_events_analysis"):
    """创建分析报告"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 按不同指标排序
    by_duration = sorted(results, key=lambda x: x['fire_days'], reverse=True)
    by_representativeness = sorted(results, key=lambda x: x['representativeness_score'], reverse=True)
    by_intensity = sorted(results, key=lambda x: x['max_fire_pixels'], reverse=True)
    by_consistency = sorted(results, key=lambda x: x['fire_day_ratio'], reverse=True)
    
    # 创建文本报告
    with open(output_path / "fire_events_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("🔥 FIRE EVENTS COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Total fire events analyzed: {len(results)}\n")
        f.write(f"• Average duration: {np.mean([r['fire_days'] for r in results]):.1f} days\n")
        f.write(f"• Average fire day ratio: {np.mean([r['fire_day_ratio'] for r in results])*100:.1f}%\n")
        f.write(f"• Max fire duration: {max(r['fire_days'] for r in results)} days\n\n")
        
        # Top 5 by different criteria
        f.write("🏆 TOP 5 BY DURATION (持续时间最长)\n")
        f.write("-" * 50 + "\n")
        for i, event in enumerate(by_duration[:5], 1):
            f.write(f"{i}. {event['file_name']}\n")
            f.write(f"   Duration: {event['fire_days']}/{event['total_days']} days ({event['fire_day_ratio']*100:.1f}%)\n")
            f.write(f"   Max pixels: {event['max_fire_pixels']:,}\n")
            f.write(f"   Avg area: {event['avg_fire_area']:.2f}%\n")
            f.write(f"   Score: {event['representativeness_score']:.2f}\n\n")
        
        f.write("🎯 TOP 5 BY REPRESENTATIVENESS (最具代表性)\n")
        f.write("-" * 50 + "\n")
        for i, event in enumerate(by_representativeness[:5], 1):
            f.write(f"{i}. {event['file_name']}\n")
            f.write(f"   Score: {event['representativeness_score']:.2f}\n")
            f.write(f"   Duration: {event['fire_days']}/{event['total_days']} days\n")
            f.write(f"   Max pixels: {event['max_fire_pixels']:,}\n")
            f.write(f"   Fire ratio: {event['fire_day_ratio']*100:.1f}%\n\n")
        
        f.write("🔥 TOP 5 BY INTENSITY (火灾强度最高)\n")
        f.write("-" * 50 + "\n")
        for i, event in enumerate(by_intensity[:5], 1):
            f.write(f"{i}. {event['file_name']}\n")
            f.write(f"   Max pixels: {event['max_fire_pixels']:,}\n")
            f.write(f"   Duration: {event['fire_days']} days\n")
            f.write(f"   Avg area: {event['avg_fire_area']:.2f}%\n")
            f.write(f"   Score: {event['representativeness_score']:.2f}\n\n")
        
        f.write("📈 RECOMMENDATIONS FOR PRESENTATION\n")
        f.write("-" * 50 + "\n")
        
        # 找出最佳展示候选
        best_overall = by_representativeness[0]
        longest_duration = by_duration[0]
        highest_intensity = by_intensity[0]
        
        f.write(f"🥇 BEST OVERALL: {best_overall['file_name']}\n")
        f.write(f"   • Representativeness score: {best_overall['representativeness_score']:.2f}\n")
        f.write(f"   • Duration: {best_overall['fire_days']} days\n")
        f.write(f"   • Fire coverage: {best_overall['fire_day_ratio']*100:.1f}%\n")
        f.write(f"   • Max intensity: {best_overall['max_fire_pixels']:,} pixels\n\n")
        
        if longest_duration['file_name'] != best_overall['file_name']:
            f.write(f"⏱️ LONGEST DURATION: {longest_duration['file_name']}\n")
            f.write(f"   • Duration: {longest_duration['fire_days']} days\n")
            f.write(f"   • Score: {longest_duration['representativeness_score']:.2f}\n\n")
        
        if highest_intensity['file_name'] not in [best_overall['file_name'], longest_duration['file_name']]:
            f.write(f"🔥 HIGHEST INTENSITY: {highest_intensity['file_name']}\n")
            f.write(f"   • Max pixels: {highest_intensity['max_fire_pixels']:,}\n")
            f.write(f"   • Score: {highest_intensity['representativeness_score']:.2f}\n\n")
    
    # 保存详细JSON数据
    with open(output_path / "fire_events_detailed_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 创建可视化
    create_visualization(results, output_path)
    
    return best_overall, longest_duration, highest_intensity

def create_visualization(results, output_path):
    """创建可视化图表"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fire Events Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Duration vs Representativeness
    durations = [r['fire_days'] for r in results]
    scores = [r['representativeness_score'] for r in results]
    axes[0, 0].scatter(durations, scores, alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Fire Duration (days)')
    axes[0, 0].set_ylabel('Representativeness Score')
    axes[0, 0].set_title('Duration vs Representativeness')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Fire Day Ratio Distribution
    ratios = [r['fire_day_ratio'] * 100 for r in results]
    axes[0, 1].hist(ratios, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Fire Day Ratio (%)')
    axes[0, 1].set_ylabel('Number of Events')
    axes[0, 1].set_title('Fire Day Ratio Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Max Fire Pixels vs Duration
    max_pixels = [r['max_fire_pixels'] for r in results]
    axes[1, 0].scatter(durations, max_pixels, alpha=0.7, s=60, c='red')
    axes[1, 0].set_xlabel('Fire Duration (days)')
    axes[1, 0].set_ylabel('Max Fire Pixels')
    axes[1, 0].set_title('Duration vs Max Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top Events Comparison
    top_5 = sorted(results, key=lambda x: x['representativeness_score'], reverse=True)[:5]
    names = [r['file_name'][:15] + '...' if len(r['file_name']) > 15 else r['file_name'] for r in top_5]
    top_scores = [r['representativeness_score'] for r in top_5]
    
    bars = axes[1, 1].bar(range(len(names)), top_scores, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Fire Events')
    axes[1, 1].set_ylabel('Representativeness Score')
    axes[1, 1].set_title('Top 5 Most Representative Events')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, top_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "fire_events_analysis_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主分析函数"""
    print("🔥 FIRE EVENTS COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # 查找所有火灾事件
    print("📁 Searching for fire event files...")
    fire_events = find_fire_events()
    
    if not fire_events:
        print("❌ No fire event files found!")
        print("💡 Make sure fire event files are in data/processed/ or data/ directories")
        return
    
    print(f"✅ Found {len(fire_events)} fire event files")
    
    # 分析每个事件
    results = []
    print(f"\n📊 Analyzing fire events...")
    
    for i, fire_path in enumerate(fire_events, 1):
        print(f"  {i}/{len(fire_events)}: {fire_path.name}")
        result = analyze_single_fire_event(fire_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid fire events could be analyzed!")
        return
    
    print(f"\n✅ Successfully analyzed {len(results)} fire events")
    
    # 创建分析报告
    print(f"\n📝 Creating analysis report...")
    best_overall, longest_duration, highest_intensity = create_analysis_report(results)
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"📁 Results saved in: fire_events_analysis/")
    print(f"📊 Dashboard: fire_events_analysis_dashboard.png")
    print(f"📋 Report: fire_events_analysis_report.txt")
    print(f"📄 Data: fire_events_detailed_analysis.json")
    
    print(f"\n🏆 TOP RECOMMENDATIONS FOR PRESENTATION:")
    print(f"🥇 Best Overall: {best_overall['file_name']}")
    print(f"   Score: {best_overall['representativeness_score']:.2f}, Duration: {best_overall['fire_days']} days")
    
    if longest_duration['file_name'] != best_overall['file_name']:
        print(f"⏱️ Longest Duration: {longest_duration['file_name']}")
        print(f"   Duration: {longest_duration['fire_days']} days, Score: {longest_duration['representativeness_score']:.2f}")
    
    print(f"\n💡 Use the best overall event for your presentation!")

if __name__ == "__main__":
    main()
