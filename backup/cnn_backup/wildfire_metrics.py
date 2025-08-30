"""
WildFire Evaluation Metrics
专用于野火传播预测的评估指标模块
包含AUPRC、IoU、F1-Score等针对类别不平衡的专业指标
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score
)
import warnings

class WildfireMetrics:
    """
    野火传播预测评估指标计算器
    """
    
    def __init__(self, device: torch.device = None, threshold: float = 0.5):
        """
        Args:
            device: 计算设备
            threshold: 二值化阈值
        """
        self.device = device if device else torch.device('cpu')
        self.threshold = threshold
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: (N, 1, H, W) 或 (N, H, W) 预测概率
            targets: (N, H, W) 目标标签 (0 or 1)
            
        Returns:
            metrics: 指标字典
        """
        # 确保形状正确
        if predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)  # (N, H, W)
        
        # 展平为向量
        pred_flat = predictions.view(-1).cpu().numpy()
        target_flat = targets.view(-1).cpu().numpy()
        
        # 过滤有效值
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(pred_flat) == 0:
            warnings.warn("No valid predictions for metric calculation")
            return self._empty_metrics()
        
        # 基础指标
        metrics = {}
        
        # 1. AUPRC (Area Under Precision-Recall Curve) - 主要指标
        try:
            metrics['auprc'] = average_precision_score(target_flat, pred_flat)
        except ValueError:
            metrics['auprc'] = 0.0
        
        # 2. AUC-ROC
        try:
            if len(np.unique(target_flat)) > 1:
                metrics['auc_roc'] = roc_auc_score(target_flat, pred_flat)
            else:
                metrics['auc_roc'] = 0.5
        except ValueError:
            metrics['auc_roc'] = 0.5
        
        # 二值化预测
        pred_binary = (pred_flat > self.threshold).astype(int)
        
        # 3. Confusion Matrix指标
        cm = confusion_matrix(target_flat, pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 基础分类指标
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['f1_score'] = f1_score(target_flat, pred_binary, zero_division=0)
        
        # 特异性和敏感性
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']  # 敏感性 = 召回率
        
        # 4. IoU (Intersection over Union) - 分割任务重要指标
        metrics['iou'] = self._compute_iou(pred_binary, target_flat)
        
        # 5. Dice系数
        metrics['dice'] = self._compute_dice(pred_binary, target_flat)
        
        # 6. 类别不平衡相关指标
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # MCC (Matthews Correlation Coefficient)
        metrics['mcc'] = self._compute_mcc(tp, tn, fp, fn)
        
        # 7. 火点特定指标
        fire_metrics = self._compute_fire_specific_metrics(
            predictions.cpu().numpy(), 
            targets.cpu().numpy()
        )
        metrics.update(fire_metrics)
        
        return metrics
    
    def _compute_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算IoU"""
        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection
        return intersection / union if union > 0 else 0.0
    
    def _compute_dice(self, pred: np.ndarray, target: np.ndarray) -> float:
        """计算Dice系数"""
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        return (2 * intersection) / total if total > 0 else 0.0
    
    def _compute_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """计算Matthews相关系数"""
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    
    def _compute_fire_specific_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        计算野火特定的评估指标
        
        Args:
            predictions: (N, H, W) 预测概率
            targets: (N, H, W) 目标标签
        """
        metrics = {}
        
        # 转换为二值预测
        pred_binary = (predictions > self.threshold).astype(int)
        
        # 1. 火点检测率 (Fire Detection Rate)
        fire_pixels_true = (targets == 1).sum()
        fire_pixels_detected = (pred_binary * targets).sum()
        metrics['fire_detection_rate'] = fire_pixels_detected / fire_pixels_true if fire_pixels_true > 0 else 0.0
        
        # 2. 误报率 (False Alarm Rate)
        non_fire_pixels = (targets == 0).sum()
        false_alarms = (pred_binary * (1 - targets)).sum()
        metrics['false_alarm_rate'] = false_alarms / non_fire_pixels if non_fire_pixels > 0 else 0.0
        
        # 3. 火点覆盖度 (Fire Coverage)
        total_pixels = targets.size
        fire_coverage_true = fire_pixels_true / total_pixels
        fire_coverage_pred = pred_binary.sum() / total_pixels
        metrics['fire_coverage_true'] = fire_coverage_true
        metrics['fire_coverage_pred'] = fire_coverage_pred
        
        # 4. 空间连通性指标
        spatial_metrics = self._compute_spatial_metrics(pred_binary, targets)
        metrics.update(spatial_metrics)
        
        return metrics
    
    def _compute_spatial_metrics(
        self, 
        pred_binary: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """计算空间连通性相关指标"""
        metrics = {}
        
        # 简化的空间指标计算
        # 在实际应用中，可以使用更复杂的空间分析
        
        try:
            from scipy import ndimage
            from scipy.spatial.distance import directed_hausdorff
            
            # 1. 连通区域数量
            pred_labeled, pred_num_features = ndimage.label(pred_binary)
            target_labeled, target_num_features = ndimage.label(targets)
            
            metrics['pred_num_regions'] = pred_num_features
            metrics['target_num_regions'] = target_num_features
            
            # 2. 最大连通区域大小
            if pred_num_features > 0:
                pred_sizes = ndimage.sum(pred_binary, pred_labeled, range(1, pred_num_features + 1))
                metrics['pred_max_region_size'] = np.max(pred_sizes) if len(pred_sizes) > 0 else 0
            else:
                metrics['pred_max_region_size'] = 0
            
            if target_num_features > 0:
                target_sizes = ndimage.sum(targets, target_labeled, range(1, target_num_features + 1))
                metrics['target_max_region_size'] = np.max(target_sizes) if len(target_sizes) > 0 else 0
            else:
                metrics['target_max_region_size'] = 0
            
            # 3. Hausdorff距离 (边界相似性)
            if pred_binary.sum() > 0 and targets.sum() > 0:
                # 获取边界点
                pred_edges = self._get_edge_points(pred_binary)
                target_edges = self._get_edge_points(targets)
                
                if len(pred_edges) > 0 and len(target_edges) > 0:
                    hausdorff_dist = max(
                        directed_hausdorff(pred_edges, target_edges)[0],
                        directed_hausdorff(target_edges, pred_edges)[0]
                    )
                    metrics['hausdorff_distance'] = hausdorff_dist
                else:
                    metrics['hausdorff_distance'] = float('inf')
            else:
                metrics['hausdorff_distance'] = float('inf')
                
        except ImportError:
            # 如果没有scipy，返回基础指标
            metrics['pred_num_regions'] = 0
            metrics['target_num_regions'] = 0
            metrics['pred_max_region_size'] = 0
            metrics['target_max_region_size'] = 0
            metrics['hausdorff_distance'] = float('inf')
        except Exception as e:
            # 处理其他异常
            warnings.warn(f"Error computing spatial metrics: {e}")
            metrics['pred_num_regions'] = 0
            metrics['target_num_regions'] = 0
            metrics['pred_max_region_size'] = 0
            metrics['target_max_region_size'] = 0
            metrics['hausdorff_distance'] = float('inf')
        
        return metrics
    
    def _get_edge_points(self, binary_mask: np.ndarray) -> np.ndarray:
        """获取二值掩码的边界点"""
        if binary_mask.ndim == 3:
            # 处理3D数组，取第一个非零切片
            for i in range(binary_mask.shape[0]):
                if binary_mask[i].sum() > 0:
                    binary_mask = binary_mask[i]
                    break
            else:
                return np.array([])
        
        try:
            from scipy import ndimage
            # 计算边界
            eroded = ndimage.binary_erosion(binary_mask)
            boundary = binary_mask.astype(int) - eroded.astype(int)
            edge_points = np.argwhere(boundary > 0)
            return edge_points
        except ImportError:
            # 简单的边界检测
            if binary_mask.ndim == 2:
                h, w = binary_mask.shape
                edge_points = []
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        if binary_mask[i, j] == 1:
                            # 检查是否为边界点
                            neighbors = binary_mask[i-1:i+2, j-1:j+2]
                            if neighbors.sum() < 9:  # 不是完全被包围
                                edge_points.append([i, j])
                return np.array(edge_points)
            return np.array([])
    
    def compute_threshold_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, List[float]]:
        """
        计算不同阈值下的指标
        
        Args:
            predictions: 预测概率
            targets: 目标标签
            thresholds: 阈值列表
            
        Returns:
            threshold_metrics: 各阈值下的指标
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()
        
        threshold_metrics = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': [],
            'dice': []
        }
        
        # 展平数据
        if predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        
        pred_flat = predictions.view(-1).cpu().numpy()
        target_flat = targets.view(-1).cpu().numpy()
        
        # 过滤有效值
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        for threshold in thresholds:
            pred_binary = (pred_flat > threshold).astype(int)
            
            # 计算指标
            precision = precision_score(target_flat, pred_binary, zero_division=0)
            recall = recall_score(target_flat, pred_binary, zero_division=0)
            f1 = f1_score(target_flat, pred_binary, zero_division=0)
            iou = self._compute_iou(pred_binary, target_flat)
            dice = self._compute_dice(pred_binary, target_flat)
            
            threshold_metrics['precision'].append(precision)
            threshold_metrics['recall'].append(recall)
            threshold_metrics['f1_score'].append(f1)
            threshold_metrics['iou'].append(iou)
            threshold_metrics['dice'].append(dice)
        
        return threshold_metrics
    
    def find_optimal_threshold(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        寻找最优阈值
        
        Args:
            predictions: 预测概率
            targets: 目标标签
            metric: 优化的指标 ('f1_score', 'iou', 'dice')
            
        Returns:
            optimal_threshold, optimal_score
        """
        thresholds = np.arange(0.01, 1.0, 0.01).tolist()
        threshold_metrics = self.compute_threshold_metrics(predictions, targets, thresholds)
        
        if metric not in threshold_metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores = threshold_metrics[metric]
        best_idx = np.argmax(scores)
        
        return thresholds[best_idx], scores[best_idx]
    
    def save_detailed_results(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        save_path: str
    ):
        """
        保存详细的评估结果
        
        Args:
            predictions: 预测概率
            targets: 目标标签
            save_path: 保存路径
        """
        # 基础指标
        basic_metrics = self.compute_metrics(predictions, targets)
        
        # 不同阈值下的指标
        threshold_metrics = self.compute_threshold_metrics(predictions, targets)
        
        # 最优阈值
        optimal_f1_th, optimal_f1_score = self.find_optimal_threshold(predictions, targets, 'f1_score')
        optimal_iou_th, optimal_iou_score = self.find_optimal_threshold(predictions, targets, 'iou')
        
        # PR曲线数据
        pred_flat = predictions.view(-1).cpu().numpy()
        target_flat = targets.view(-1).cpu().numpy()
        
        # 过滤有效值
        valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(np.unique(target_flat)) > 1:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(target_flat, pred_flat)
            pr_curve_data = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        else:
            pr_curve_data = None
        
        # 组织结果
        detailed_results = {
            'basic_metrics': basic_metrics,
            'threshold_analysis': threshold_metrics,
            'optimal_thresholds': {
                'f1_score': {'threshold': optimal_f1_th, 'score': optimal_f1_score},
                'iou': {'threshold': optimal_iou_th, 'score': optimal_iou_score}
            },
            'pr_curve': pr_curve_data,
            'data_statistics': {
                'total_pixels': int(targets.numel()),
                'fire_pixels': int(targets.sum().item()),
                'fire_ratio': float(targets.mean().item()),
                'prediction_mean': float(predictions.mean().item()),
                'prediction_std': float(predictions.std().item())
            }
        }
        
        # 保存结果
        with open(save_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"💾 详细评估结果已保存至: {save_path}")
    
    def _empty_metrics(self) -> Dict[str, float]:
        """返回空指标字典"""
        return {
            'auprc': 0.0,
            'auc_roc': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'sensitivity': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'balanced_accuracy': 0.0,
            'mcc': 0.0,
            'fire_detection_rate': 0.0,
            'false_alarm_rate': 0.0,
            'fire_coverage_true': 0.0,
            'fire_coverage_pred': 0.0
        }

def test_metrics():
    """测试评估指标"""
    print("🧪 测试野火评估指标...")
    
    # 创建测试数据
    batch_size, height, width = 4, 64, 64
    
    # 模拟预测和目标
    predictions = torch.rand(batch_size, 1, height, width)
    targets = torch.zeros(batch_size, height, width)
    
    # 添加一些火点
    targets[:, 10:20, 10:20] = 1.0
    targets[:, 40:45, 40:45] = 1.0
    
    # 创建评估器
    metrics_calculator = WildfireMetrics()
    
    # 计算指标
    metrics = metrics_calculator.compute_metrics(predictions, targets)
    
    print(f"🔥 火点像素比例: {targets.mean():.4f}")
    print(f"📊 评估指标:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:20}: {value:.4f}")
    
    # 测试阈值分析
    print(f"\n🎯 最优阈值分析:")
    optimal_f1_th, optimal_f1_score = metrics_calculator.find_optimal_threshold(
        predictions, targets, 'f1_score'
    )
    print(f"  最优F1阈值: {optimal_f1_th:.3f} (F1: {optimal_f1_score:.4f})")
    
    optimal_iou_th, optimal_iou_score = metrics_calculator.find_optimal_threshold(
        predictions, targets, 'iou'
    )
    print(f"  最优IoU阈值: {optimal_iou_th:.3f} (IoU: {optimal_iou_score:.4f})")
    
    print("✅ 指标测试完成！")

if __name__ == "__main__":
    test_metrics() 