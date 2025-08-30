#!/usr/bin/env python3
"""
训练速度诊断工具
分析和解决训练越来越慢的问题
"""

import torch
import psutil
import time
import gc
import numpy as np
from pathlib import Path

def diagnose_training_issues():
    """诊断训练速度问题"""
    
    print("🔍 训练速度诊断报告")
    print("=" * 50)
    
    # 1. 系统资源检查
    print("\n1. 系统资源状态:")
    print(f"   CPU使用率: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"   内存使用: {psutil.virtual_memory().percent:.1f}%")
    print(f"   可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 2. GPU状态检查
    if torch.cuda.is_available():
        print("\n2. GPU状态:")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   内存已用: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"   内存缓存: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
            print(f"   内存总量: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        print("\n2. GPU状态: 未检测到CUDA设备")
    
    # 3. PyTorch设置检查
    print("\n3. PyTorch配置:")
    print(f"   版本: {torch.__version__}")
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   CuDNN启用: {torch.backends.cudnn.enabled}")
    print(f"   CuDNN基准: {torch.backends.cudnn.benchmark}")
    
    # 4. 常见问题检查
    print("\n4. 常见问题检查:")
    
    issues_found = []
    
    # 检查GPU内存碎片化
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        if reserved > allocated * 1.5:
            issues_found.append("GPU内存碎片化严重")
    
    # 检查内存使用
    if psutil.virtual_memory().percent > 80:
        issues_found.append("系统内存使用过高")
    
    # 检查CPU使用
    cpu_usage = psutil.cpu_percent(interval=1)
    if cpu_usage > 90:
        issues_found.append("CPU使用率过高")
    
    if issues_found:
        print("   ⚠️  发现问题:")
        for issue in issues_found:
            print(f"      - {issue}")
    else:
        print("   ✅ 未发现明显的硬件问题")
    
    return issues_found

def fix_common_issues():
    """修复常见问题"""
    
    print("\n🔧 应用常见修复方案:")
    
    # 1. 清理GPU缓存
    if torch.cuda.is_available():
        print("   清理GPU缓存...")
        torch.cuda.empty_cache()
        print("   ✅ GPU缓存已清理")
    
    # 2. 强制垃圾回收
    print("   执行垃圾回收...")
    gc.collect()
    print("   ✅ 垃圾回收完成")
    
    # 3. 优化PyTorch设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("   ✅ 启用CuDNN基准模式")
    
    print("\n📋 推荐的训练优化:")
    print("   1. 减小batch_size (当前建议: 2-4)")
    print("   2. 使用梯度累积替代大batch")
    print("   3. 定期清理GPU缓存 (每5-10个epoch)")
    print("   4. 使用pin_memory=True")
    print("   5. 设置num_workers=0 (Windows)")
    print("   6. 避免在训练循环中累积数据")

def create_optimized_training_tips():
    """创建优化训练的具体建议"""
    
    tips = """
🚀 训练速度优化指南:

1. 内存管理:
   - 使用 optimizer.zero_grad(set_to_none=True)
   - 避免在循环中累积tensor
   - 定期调用 torch.cuda.empty_cache()
   - 使用 del 显式删除大tensor

2. 数据加载优化:
   - 设置 pin_memory=True
   - Windows上使用 num_workers=0
   - 使用 non_blocking=True
   - 避免数据转换在训练循环中

3. 模型优化:
   - 使用 inplace=True 操作
   - 启用 torch.backends.cudnn.benchmark
   - 使用混合精度训练 (autocast)
   - 减少不必要的计算

4. 训练循环优化:
   - 避免频繁的CPU-GPU数据传输
   - 使用 .item() 获取标量值
   - 不要在循环中保存完整预测结果
   - 定期检查内存使用情况

5. 监控和调试:
   - 记录每个epoch的时间
   - 监控GPU内存使用
   - 检查是否有内存泄漏
   - 使用性能分析工具
"""
    
    return tips

def test_memory_efficiency():
    """测试内存效率"""
    
    print("\n🧪 内存效率测试:")
    
    if not torch.cuda.is_available():
        print("   跳过GPU测试 (无CUDA设备)")
        return
    
    device = torch.device('cuda')
    
    # 记录初始内存
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"   初始GPU内存: {initial_memory / (1024**2):.1f} MB")
    
    # 创建测试张量
    test_tensors = []
    for i in range(10):
        tensor = torch.randn(100, 100, device=device)
        test_tensors.append(tensor)
    
    peak_memory = torch.cuda.memory_allocated(device)
    print(f"   峰值GPU内存: {peak_memory / (1024**2):.1f} MB")
    
    # 清理测试
    del test_tensors
    torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated(device)
    print(f"   清理后内存: {final_memory / (1024**2):.1f} MB")
    
    if final_memory <= initial_memory * 1.1:
        print("   ✅ 内存清理正常")
    else:
        print("   ⚠️  可能存在内存泄漏")

def main():
    """主诊断函数"""
    
    print("🏥 野火CNN训练速度诊断工具")
    print("分析训练越来越慢的原因并提供解决方案")
    
    # 执行诊断
    issues = diagnose_training_issues()
    
    # 修复常见问题
    fix_common_issues()
    
    # 内存效率测试
    test_memory_efficiency()
    
    # 提供优化建议
    tips = create_optimized_training_tips()
    print(tips)
    
    # 总结建议
    print("\n🎯 针对您的问题的具体建议:")
    print("1. 立即运行 optimized_wildfire_cnn.py (已修复内存泄漏)")
    print("2. 将batch_size设置为2-4")
    print("3. 每5个epoch清理一次GPU缓存")
    print("4. 监控训练时间变化")
    print("5. 如果仍然变慢，考虑重启训练进程")

if __name__ == "__main__":
    main() 