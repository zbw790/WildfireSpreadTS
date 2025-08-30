#!/usr/bin/env python3
"""
GPU加速测试脚本
"""

import torch
import time

def test_gpu_acceleration():
    """测试GPU加速是否可用"""
    
    print("🔧 GPU加速测试")
    print("=" * 50)
    
    # 1. 基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
            print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("❌ CUDA不可用")
        return False
    
    # 2. 性能测试
    print("\n🚀 性能对比测试:")
    
    # 创建测试数据
    size = (1000, 1000)
    cpu_tensor1 = torch.randn(size)
    cpu_tensor2 = torch.randn(size)
    
    if torch.cuda.is_available():
        gpu_tensor1 = cpu_tensor1.cuda()
        gpu_tensor2 = cpu_tensor2.cuda()
        
        # CPU测试
        start_time = time.time()
        for _ in range(100):
            result_cpu = torch.mm(cpu_tensor1, cpu_tensor2)
        cpu_time = time.time() - start_time
        
        # GPU测试
        torch.cuda.synchronize()  # 确保GPU操作完成
        start_time = time.time()
        for _ in range(100):
            result_gpu = torch.mm(gpu_tensor1, gpu_tensor2)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"CPU时间: {cpu_time:.3f}秒")
        print(f"GPU时间: {gpu_time:.3f}秒")
        print(f"加速比: {cpu_time / gpu_time:.1f}x")
        
        # 验证结果一致性
        diff = torch.abs(result_cpu - result_gpu.cpu()).max()
        print(f"结果差异: {diff:.6f} (应该很小)")
        
        return True
    else:
        return False

def test_cnn_on_gpu():
    """测试CNN模型在GPU上的运行"""
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，跳过CNN测试")
        return
    
    print("\n🧠 CNN模型GPU测试:")
    
    # 创建简单的CNN模型
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.fc = torch.nn.Linear(64 * 32 * 32, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.nn.functional.adaptive_avg_pool2d(x, (32, 32))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 创建模型和数据
    model = SimpleCNN()
    data = torch.randn(4, 3, 128, 128)  # batch_size=4, channels=3, size=128x128
    
    # CPU测试
    start_time = time.time()
    with torch.no_grad():
        output_cpu = model(data)
    cpu_time = time.time() - start_time
    
    # GPU测试
    model_gpu = model.cuda()
    data_gpu = data.cuda()
    
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output_gpu = model_gpu(data_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"CNN CPU推理时间: {cpu_time:.3f}秒")
    print(f"CNN GPU推理时间: {gpu_time:.3f}秒")
    print(f"CNN加速比: {cpu_time / gpu_time:.1f}x")
    
    # 验证结果
    diff = torch.abs(output_cpu - output_gpu.cpu()).max()
    print(f"输出差异: {diff:.6f}")
    print("✅ CNN GPU测试成功!")

if __name__ == "__main__":
    print("🔥 野火CNN GPU加速测试")
    
    # 测试GPU基本功能
    gpu_available = test_gpu_acceleration()
    
    if gpu_available:
        # 测试CNN模型
        test_cnn_on_gpu()
        
        print("\n🎉 GPU加速测试完成!")
        print("✅ 您的环境支持GPU加速")
        print("💡 现在可以使用GPU训练野火CNN模型了")
    else:
        print("\n❌ GPU加速不可用")
        print("💡 请安装GPU版本的PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121") 