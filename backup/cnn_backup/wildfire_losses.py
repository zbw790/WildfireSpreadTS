"""
WildFire Loss Functions
专用于野火传播预测的损失函数，特别针对严重类别不平衡问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    paper: Focal Loss for Dense Object Detection
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) logits
            targets: (N, H, W) ground truth labels (0 or 1)
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # 计算alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # 计算focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions
            targets: (N, H, W) ground truth labels
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算intersection和union
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_score

class IoULoss(nn.Module):
    """
    IoU Loss (Jaccard Loss) for segmentation
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions
            targets: (N, H, W) ground truth labels
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算intersection和union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou

class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions
            targets: (N, H, W) ground truth labels
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算true positives, false positives, false negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class ComboLoss(nn.Module):
    """
    Combination of multiple losses for wildfire prediction
    """
    
    def __init__(
        self, 
        focal_weight: float = 0.5,
        dice_weight: float = 0.3,
        iou_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super(ComboLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions
            targets: (N, H, W) ground truth labels
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        iou = self.iou_loss(inputs, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.iou_weight * iou)
        
        return total_loss

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions
            targets: (N, H, W) ground truth labels
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 动态计算pos_weight如果没有提供
        if self.pos_weight is None:
            neg_count = (targets == 0).sum().float()
            pos_count = (targets == 1).sum().float()
            pos_weight = neg_count / (pos_count + 1e-8)
        else:
            pos_weight = self.pos_weight
        
        loss = F.binary_cross_entropy(
            inputs, targets, 
            weight=None,
            reduction='none'
        )
        
        # 对正样本加权
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        weighted_loss = loss * (pos_mask * pos_weight + neg_mask)
        
        return weighted_loss.mean()

class LovaszSigmoidLoss(nn.Module):
    """
    Lovász-Sigmoid loss for binary segmentation
    """
    
    def __init__(self):
        super(LovaszSigmoidLoss, self).__init__()
    
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovász extension w.r.t sorted errors
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if p == 0:
            return gt_sorted
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C, H, W) or (N, H, W) predictions (logits)
            targets: (N, H, W) ground truth labels
        """
        # 确保输入形状正确
        if inputs.dim() == 4 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)  # (N, H, W)
        
        # 展平
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1).byte()
        
        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # 计算errors
        errors = (inputs_flat - targets_flat.float()).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        targets_sorted = targets_flat[perm]
        
        return torch.dot(errors_sorted, self.lovasz_grad(targets_sorted))

class WildfireLossFactory:
    """
    野火损失函数工厂类
    """
    
    @staticmethod
    def create_loss(
        loss_type: str,
        class_imbalance_ratio: float = 100.0,
        **kwargs
    ) -> nn.Module:
        """
        创建损失函数
        
        Args:
            loss_type: 损失函数类型
            class_imbalance_ratio: 类别不平衡比例 (neg/pos)
            **kwargs: 其他参数
        """
        
        if loss_type == 'focal':
            # 根据类别不平衡程度调整alpha
            alpha = min(0.25, 1.0 / (1.0 + class_imbalance_ratio))
            gamma = kwargs.get('gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        
        elif loss_type == 'dice':
            return DiceLoss()
        
        elif loss_type == 'iou':
            return IoULoss()
        
        elif loss_type == 'tversky':
            # 对于不平衡数据，增加对false negatives的惩罚
            alpha = kwargs.get('alpha', 0.3)
            beta = kwargs.get('beta', 0.7)
            return TverskyLoss(alpha=alpha, beta=beta)
        
        elif loss_type == 'combo':
            # 组合损失，自动调整权重
            focal_alpha = min(0.25, 1.0 / (1.0 + class_imbalance_ratio))
            return ComboLoss(
                focal_weight=0.5,
                dice_weight=0.3,
                iou_weight=0.2,
                focal_alpha=focal_alpha
            )
        
        elif loss_type == 'weighted_bce':
            return WeightedBCELoss(pos_weight=class_imbalance_ratio)
        
        elif loss_type == 'lovasz':
            return LovaszSigmoidLoss()
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def get_recommended_loss(class_imbalance_ratio: float) -> nn.Module:
        """
        根据类别不平衡程度推荐最佳损失函数
        
        Args:
            class_imbalance_ratio: 类别不平衡比例 (neg/pos)
        """
        if class_imbalance_ratio > 1000:
            # 严重不平衡：使用组合损失
            print(f"检测到严重类别不平衡 ({class_imbalance_ratio:.0f}:1)，推荐使用ComboLoss")
            return WildfireLossFactory.create_loss('combo', class_imbalance_ratio)
        
        elif class_imbalance_ratio > 100:
            # 中等不平衡：使用Focal Loss
            print(f"检测到中等类别不平衡 ({class_imbalance_ratio:.0f}:1)，推荐使用FocalLoss")
            return WildfireLossFactory.create_loss('focal', class_imbalance_ratio)
        
        elif class_imbalance_ratio > 10:
            # 轻微不平衡：使用加权BCE
            print(f"检测到轻微类别不平衡 ({class_imbalance_ratio:.0f}:1)，推荐使用WeightedBCE")
            return WildfireLossFactory.create_loss('weighted_bce', class_imbalance_ratio)
        
        else:
            # 平衡数据：使用标准BCE
            print(f"数据相对平衡 ({class_imbalance_ratio:.0f}:1)，使用标准BCE")
            return nn.BCELoss()

def test_losses():
    """测试损失函数"""
    print("🧪 测试野火损失函数...")
    
    # 创建测试数据
    batch_size, height, width = 4, 64, 64
    
    # 模拟严重不平衡的数据（火点很少）
    predictions = torch.rand(batch_size, 1, height, width)
    targets = torch.zeros(batch_size, height, width)
    
    # 添加少量火点
    targets[:, 10:15, 10:15] = 1.0  # 小块火区
    targets[:, 40:42, 40:42] = 1.0  # 更小的火点
    
    fire_ratio = targets.mean().item()
    imbalance_ratio = (1 - fire_ratio) / (fire_ratio + 1e-8)
    
    print(f"火点比例: {fire_ratio:.4f} ({fire_ratio*100:.2f}%)")
    print(f"不平衡比例: {imbalance_ratio:.1f}:1")
    
    # 测试不同损失函数
    losses = {
        'BCE': nn.BCELoss(),
        'FocalLoss': FocalLoss(alpha=0.25, gamma=2.0),
        'DiceLoss': DiceLoss(),
        'IoULoss': IoULoss(),
        'ComboLoss': ComboLoss(),
        'WeightedBCE': WeightedBCELoss(),
    }
    
    print("\n📊 损失函数对比:")
    for name, loss_fn in losses.items():
        try:
            loss_value = loss_fn(predictions, targets)
            print(f"  {name:12}: {loss_value.item():.6f}")
        except Exception as e:
            print(f"  {name:12}: Error - {e}")
    
    # 测试推荐系统
    print(f"\n💡 推荐损失函数:")
    recommended_loss = WildfireLossFactory.get_recommended_loss(imbalance_ratio)
    recommended_value = recommended_loss(predictions, targets)
    print(f"   推荐损失值: {recommended_value.item():.6f}")
    
    print("✅ 损失函数测试完成！")

if __name__ == "__main__":
    test_losses() 