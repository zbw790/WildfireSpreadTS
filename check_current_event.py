#!/usr/bin/env python3
"""
检查当前使用的火灾事件在分析中的排名
"""

import json

def check_current_event():
    """检查当前事件的排名"""
    with open('fire_events_analysis/fire_events_detailed_analysis.json', 'r') as f:
        data = json.load(f)
    
    # 查找当前使用的事件
    current_event = None
    for event in data:
        if 'fire_24461899' in event['file_name']:
            current_event = event
            break
    
    if not current_event:
        print("❌ Current event fire_24461899.hdf5 not found in analysis!")
        return
    
    # 按代表性排序所有事件
    sorted_by_score = sorted(data, key=lambda x: x['representativeness_score'], reverse=True)
    
    # 找到当前事件的排名
    current_rank = None
    for i, event in enumerate(sorted_by_score, 1):
        if 'fire_24461899' in event['file_name']:
            current_rank = i
            break
    
    print("🔍 CURRENT EVENT ANALYSIS")
    print("=" * 50)
    print(f"📁 Current Event: fire_24461899.hdf5")
    print(f"📊 Ranking: #{current_rank} out of {len(data)} events")
    print(f"⏱️  Duration: {current_event['fire_days']} days")
    print(f"📈 Fire day ratio: {current_event['fire_day_ratio']*100:.1f}%")
    print(f"🔥 Max fire pixels: {current_event['max_fire_pixels']:,}")
    print(f"🎯 Representativeness score: {current_event['representativeness_score']:.2f}")
    
    # 显示前5名对比
    print(f"\n🏆 TOP 5 EVENTS FOR COMPARISON:")
    print("-" * 50)
    for i, event in enumerate(sorted_by_score[:5], 1):
        name = event['file_name'].split('/')[-1]  # 只显示文件名
        print(f"{i}. {name}")
        print(f"   Score: {event['representativeness_score']:.2f}, Duration: {event['fire_days']} days")
        print(f"   Fire ratio: {event['fire_day_ratio']*100:.1f}%, Max pixels: {event['max_fire_pixels']:,}")
        if 'fire_24461899' in event['file_name']:
            print("   ⭐ <- CURRENT EVENT")
        print()
    
    # 建议
    if current_rank <= 5:
        print(f"✅ EXCELLENT CHOICE! Your current event is in TOP 5!")
    elif current_rank <= 20:
        print(f"👍 GOOD CHOICE! Your current event is in TOP 20.")
    else:
        print(f"💡 CONSIDER UPGRADING: Your current event ranks #{current_rank}.")
        print(f"   Top recommendation: {sorted_by_score[0]['file_name'].split('/')[-1]}")
        print(f"   Score: {sorted_by_score[0]['representativeness_score']:.2f} vs {current_event['representativeness_score']:.2f}")

if __name__ == "__main__":
    check_current_event()
