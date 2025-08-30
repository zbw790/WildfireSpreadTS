"""
测试完整的野火CNN建模框架
验证所有组件的功能和集成
"""

import sys
import os
sys.path.append('models')

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler

def test_framework():
    """测试完整的CNN建模框架"""
    print("🧪 测试CNN建模框架完整性...")
    
    # 1. 检查基础依赖
    try:
        import torch
        import numpy as np
        from sklearn.preprocessing import RobustScaler
        print("✅ 基础依赖检查通过")
    except ImportError as e:
        print(f"❌ 基础依赖缺失: {e}")
        return False
    
    # 2. 检查自定义模块导入
    try:
        from wildfire_cnn_model import WildfireCNN, count_parameters
        from wildfire_losses import WildfireLossFactory, FocalLoss
        from wildfire_metrics import WildfireMetrics
        from wildfire_preprocessing import WildfireNormalizer
        print("✅ 自定义模块导入成功")
    except ImportError as e:
        print(f"❌ 自定义模块导入失败: {e}")
        return False
    
    # 3. 测试模型创建
    try:
        model = WildfireCNN(input_channels=23, sequence_length=5)
        param_count = count_parameters(model)
        print(f"✅ 模型创建成功，参数数量: {param_count:,}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 4. 测试损失函数
    try:
        loss_fn = WildfireLossFactory.get_recommended_loss(1000.0)
        loss_name = type(loss_fn).__name__
        print(f"✅ 损失函数创建成功: {loss_name}")
    except Exception as e:
        print(f"❌ 损失函数创建失败: {e}")
        return False
    
    # 5. 测试评估指标
    try:
        metrics = WildfireMetrics()
        test_pred = torch.rand(2, 1, 64, 64)
        test_target = torch.randint(0, 2, (2, 64, 64)).float()
        result = metrics.compute_metrics(test_pred, test_target)
        auprc_score = result["auprc"]
        print(f"✅ 评估指标计算成功，AUPRC: {auprc_score:.4f}")
    except Exception as e:
        print(f"❌ 评估指标计算失败: {e}")
        return False
    
    # 6. 测试预处理
    try:
        normalizer = WildfireNormalizer(method='robust')
        test_data = np.random.randn(10, 23)
        normalized = normalizer.fit_transform(test_data)
        print(f"✅ 数据预处理成功，形状: {normalized.shape}")
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return False
    
    # 7. 测试完整前向传播
    try:
        model.eval()
        batch_size, seq_len, channels, height, width = 2, 5, 23, 64, 64
        test_input = torch.randn(batch_size, seq_len, channels, height, width)
        
        # 设置土地覆盖通道(16)为合理的类别值(1-16)
        test_input[:, :, 16, :, :] = torch.randint(1, 17, (batch_size, seq_len, height, width)).float()
        
        # 前向传播
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (batch_size, 1, height, width)
        if output.shape == expected_shape:
            print(f"✅ 前向传播成功，输出形状: {output.shape}")
        else:
            print(f"❌ 输出形状不正确: 期望 {expected_shape}, 得到 {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    # 8. 测试损失计算
    try:
        target = torch.randint(0, 2, (batch_size, height, width)).float()
        loss = loss_fn(output, target)
        print(f"✅ 损失计算成功，损失值: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        return False
    
    print("🎉 CNN建模框架完整性测试完成！所有组件正常工作")
    return True

def test_mini_training():
    """测试迷你训练流程"""
    print("\n🚀 测试迷你训练流程...")
    
    try:
        from wildfire_cnn_model import WildfireCNN
        from wildfire_losses import FocalLoss
        from wildfire_metrics import WildfireMetrics
        
        # 创建模型和组件
        device = torch.device('cpu')  # 使用CPU避免GPU依赖
        model = WildfireCNN(input_channels=23, sequence_length=3, 
                           unet_features=[32, 64], lstm_hidden_dims=[64])
        model = model.to(device)
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics_calc = WildfireMetrics(device=device)
        
        # 创建模拟数据
        batch_size, seq_len, channels = 2, 3, 23
        height, width = 32, 32  # 小尺寸快速测试
        
        train_data = torch.randn(batch_size, seq_len, channels, height, width).to(device)
        # 设置土地覆盖通道为合理值
        train_data[:, :, 16, :, :] = torch.randint(1, 17, (batch_size, seq_len, height, width)).float().to(device)
        train_targets = torch.randint(0, 2, (batch_size, height, width)).float().to(device)
        
        # 模拟几步训练
        model.train()
        train_losses = []
        
        for step in range(3):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(train_data)
            loss = criterion(outputs, train_targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            print(f"  Step {step+1}: Loss = {loss.item():.6f}")
        
        # 测试评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(train_data)
            val_metrics = metrics_calc.compute_metrics(val_outputs, train_targets)
        
        print(f"✅ 迷你训练完成")
        print(f"   最终损失: {train_losses[-1]:.6f}")
        print(f"   验证AUPRC: {val_metrics['auprc']:.4f}")
        print(f"   验证IoU: {val_metrics['iou']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 迷你训练失败: {e}")
        return False

if __name__ == "__main__":
    # 测试框架
    framework_ok = test_framework()
    
    if framework_ok:
        # 测试训练流程
        training_ok = test_mini_training()
        
        if training_ok:
            print("\n🎉 完整的CNN建模框架测试通过！")
            print("📋 框架包含以下组件:")
            print("   • 23通道多模态数据加载器")
            print("   • U-Net + ConvLSTM时空CNN模型")
            print("   • Focal Loss处理类别不平衡")
            print("   • 分类别特征标准化")
            print("   • 物理一致性数据增强")
            print("   • 专业野火评估指标(AUPRC, IoU等)")
            print("   • 完整训练和验证流程")
            print("\n🚀 准备好开始正式训练了！")
        else:
            print("\n❌ 训练流程测试失败")
    else:
        print("\n❌ 框架组件测试失败") 