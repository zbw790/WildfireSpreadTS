"""
WildFire CNN Training Script
完整的野火传播预测CNN训练脚本
整合数据加载、模型、损失函数、评估指标等所有组件
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
# 可选导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# 添加models目录到路径
sys.path.append('models')

# 导入自定义模块
from wildfire_cnn_dataloader import create_dataloaders, WildfireDataset
from wildfire_cnn_model import WildfireCNN, WildfireResNet, count_parameters
from wildfire_losses import WildfireLossFactory
from wildfire_metrics import WildfireMetrics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WildfireTrainer:
    """
    野火CNN训练器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
        
        # 初始化组件
        self._setup_logging()
        self._load_data()
        self._build_model()
        self._setup_loss_and_optimizer()
        self._setup_metrics()
        self._setup_monitoring()
        
        logger.info(f"野火CNN训练器初始化完成")
        logger.info(f"模型参数数量: {count_parameters(self.model):,}")
        logger.info(f"使用设备: {self.device}")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _load_data(self):
        """加载数据"""
        logger.info("加载数据...")
        
        data_config = self.config['data']
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=data_config['data_dir'],
            train_years=data_config['train_years'],
            val_years=data_config['val_years'],
            test_years=data_config['test_years'],
            batch_size=data_config['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            sequence_length=data_config.get('sequence_length', 5),
            crop_size=data_config.get('crop_size', 128),
            stride=data_config.get('stride', 64),
            fire_threshold=data_config.get('fire_threshold', 0.5),
            augment=data_config.get('augment', True)
        )
        
        # 估算类别不平衡比例
        self.class_imbalance_ratio = self._estimate_class_imbalance()
        logger.info(f"类别不平衡比例: {self.class_imbalance_ratio:.1f}:1")
    
    def _estimate_class_imbalance(self) -> float:
        """估算类别不平衡比例"""
        logger.info("分析类别分布...")
        
        fire_pixels = 0
        total_pixels = 0
        sample_count = min(100, len(self.train_loader))
        
        for i, (_, targets) in enumerate(self.train_loader):
            if i >= sample_count:
                break
            
            fire_pixels += targets.sum().item()
            total_pixels += targets.numel()
        
        fire_ratio = fire_pixels / (total_pixels + 1e-8)
        imbalance_ratio = (1 - fire_ratio) / (fire_ratio + 1e-8)
        
        return imbalance_ratio
    
    def _build_model(self):
        """构建模型"""
        logger.info("构建模型...")
        
        model_config = self.config['model']
        model_type = model_config.get('type', 'WildfireCNN')
        
        if model_type == 'WildfireCNN':
            self.model = WildfireCNN(
                input_channels=model_config.get('input_channels', 23),
                sequence_length=model_config.get('sequence_length', 5),
                unet_features=model_config.get('unet_features', [64, 128, 256, 512]),
                lstm_hidden_dims=model_config.get('lstm_hidden_dims', [128, 64]),
                num_classes=model_config.get('num_classes', 1),
                dropout=model_config.get('dropout', 0.1),
                use_attention=model_config.get('use_attention', True)
            )
        elif model_type == 'WildfireResNet':
            self.model = WildfireResNet(
                input_channels=model_config.get('input_channels', 23),
                num_classes=model_config.get('num_classes', 1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # 模型并行
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行并行训练")
            self.model = nn.DataParallel(self.model)
    
    def _setup_loss_and_optimizer(self):
        """设置损失函数和优化器"""
        logger.info("设置损失函数和优化器...")
        
        # 损失函数
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'auto')
        
        if loss_type == 'auto':
            self.criterion = WildfireLossFactory.get_recommended_loss(self.class_imbalance_ratio)
        else:
            self.criterion = WildfireLossFactory.create_loss(
                loss_type=loss_type,
                class_imbalance_ratio=self.class_imbalance_ratio,
                **loss_config.get('params', {})
            )
        
        # 优化器
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'Adam')
        
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-3),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                weight_decay=optimizer_config.get('weight_decay', 1e-2)
            )
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-2),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # 监控验证AUPRC（越大越好）
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def _setup_metrics(self):
        """设置评估指标"""
        self.metrics = WildfireMetrics(device=self.device)
    
    def _setup_monitoring(self):
        """设置监控"""
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Weights & Biases
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get('wandb_project', 'wildfire-cnn'),
                config=self.config,
                name=self.config.get('experiment_name', f"wildfire-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            )
        elif self.config.get('use_wandb', False) and not WANDB_AVAILABLE:
            logger.warning("Weights & Biases requested but not available. Install with: pip install wandb")
        
        # 最佳模型跟踪
        self.best_val_auprc = 0.0
        self.best_model_path = self.output_dir / 'best_model.pth'
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            # 数据转移到设备
            features = features.to(self.device, non_blocking=True)  # (B, T, C, H, W)
            targets = targets.to(self.device, non_blocking=True)    # (B, H, W)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features)  # (B, 1, H, W)
            
            # 调试信息
            if batch_idx == 0:
                print(f"Debug - Features range: [{features.min().item():.6f}, {features.max().item():.6f}]")
                print(f"Debug - Features shape: {features.shape}")
                print(f"Debug - Features has NaN: {torch.isnan(features).any().item()}")
                print(f"Debug - Features has Inf: {torch.isinf(features).any().item()}")
                print(f"Debug - Landcover channel (16) range: [{features[:, :, 16, :, :].min().item():.2f}, {features[:, :, 16, :, :].max().item():.2f}]")
                print(f"Debug - Outputs range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                print(f"Debug - Targets range: [{targets.min().item():.6f}, {targets.max().item():.6f}]")
                print(f"Debug - Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")
                print(f"Debug - Outputs has NaN: {torch.isnan(outputs).any().item()}")
                print(f"Debug - Outputs has Inf: {torch.isinf(outputs).any().item()}")
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validation"):
                # 数据转移到设备
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(features)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测和目标
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics.compute_metrics(all_outputs, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """测试模型"""
        logger.info("测试模型...")
        
        # 加载最佳模型
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"加载最佳模型: {self.best_model_path}")
        
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.test_loader, desc="Testing"):
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        test_metrics = self.metrics.compute_metrics(all_outputs, all_targets)
        test_metrics['loss'] = total_loss / len(self.test_loader)
        
        # 保存详细测试结果
        self.metrics.save_detailed_results(
            all_outputs, all_targets, 
            self.output_dir / 'test_results.json'
        )
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, nn.DataParallel) 
                               else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'class_imbalance_ratio': self.class_imbalance_ratio
        }
        
        # 保存最新检查点
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            logger.info(f"保存最佳模型: AUPRC = {metrics['auprc']:.4f}")
    
    def train(self):
        """完整训练流程"""
        logger.info("开始训练...")
        
        training_config = self.config['training']
        epochs = training_config['epochs']
        
        # 训练历史
        train_history = {'loss': []}
        val_history = {'loss': [], 'auprc': [], 'iou': [], 'f1_score': []}
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            train_history['loss'].append(train_metrics['loss'])
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            for key, value in val_metrics.items():
                if key not in val_history:
                    val_history[key] = []
                val_history[key].append(value)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auprc'])
                else:
                    self.scheduler.step()
            
            # 检查是否为最佳模型
            is_best = val_metrics['auprc'] > self.best_val_auprc
            if is_best:
                self.best_val_auprc = val_metrics['auprc']
            
            # 保存检查点
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 记录到监控系统
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # 打印进度
            logger.info(f"Train Loss: {train_metrics['loss']:.6f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.6f}, "
                       f"AUPRC: {val_metrics['auprc']:.4f}, "
                       f"IoU: {val_metrics['iou']:.4f}, "
                       f"F1: {val_metrics['f1_score']:.4f}")
            
            # 早停检查
            if self._early_stopping_check(val_history, training_config.get('early_stopping_patience', 10)):
                logger.info("早停触发，停止训练")
                break
        
        # 保存训练历史
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': train_history,
                'val': val_history,
                'best_val_auprc': self.best_val_auprc
            }, f, indent=2)
        
        # 测试最佳模型
        test_metrics = self.test_model()
        
        # 保存最终结果
        results = {
            'best_val_auprc': self.best_val_auprc,
            'test_metrics': test_metrics,
            'model_parameters': count_parameters(self.model),
            'training_config': self.config
        }
        
        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"训练完成!")
        logger.info(f"最佳验证AUPRC: {self.best_val_auprc:.4f}")
        logger.info(f"测试AUPRC: {test_metrics['auprc']:.4f}")
        logger.info(f"结果保存至: {self.output_dir}")
        
        return results
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """记录指标到监控系统"""
        # TensorBoard
        self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{key.upper()}', value, epoch)
        
        # Weights & Biases
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
    
    def _early_stopping_check(self, val_history: Dict[str, List], patience: int) -> bool:
        """早停检查"""
        if len(val_history['auprc']) < patience:
            return False
        
        recent_scores = val_history['auprc'][-patience:]
        return max(recent_scores) <= val_history['auprc'][-(patience+1)]

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'data': {
            'data_dir': 'data/processed',
            'train_years': [2018, 2019],
            'val_years': [2020],
            'test_years': [2021],
            'batch_size': 8,
            'num_workers': 4,
            'sequence_length': 5,
            'crop_size': 128,
            'stride': 64,
            'fire_threshold': 0.5,
            'augment': True
        },
        'model': {
            'type': 'WildfireCNN',
            'input_channels': 23,
            'sequence_length': 5,
            'unet_features': [64, 128, 256, 512],
            'lstm_hidden_dims': [128, 64],
            'num_classes': 1,
            'dropout': 0.1,
            'use_attention': True
        },
        'loss': {
            'type': 'auto'  # 自动选择
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-3,
            'weight_decay': 1e-2
        },
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 5
        },
        'training': {
            'epochs': 50,
            'early_stopping_patience': 10
        },
        'device': 'cuda',
        'seed': 42,
        'output_dir': 'outputs/wildfire_cnn',
        'use_wandb': False,
        'experiment_name': f'wildfire-cnn-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WildFire CNN Training")
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs/wildfire_cnn', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--model_type', type=str, default='WildfireCNN', 
                       choices=['WildfireCNN', 'WildfireResNet'], help='模型类型')
    parser.add_argument('--use_wandb', action='store_true', help='使用Weights & Biases')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.use_wandb:
        config['use_wandb'] = True
    
    # 保存配置
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器并开始训练
    trainer = WildfireTrainer(config)
    results = trainer.train()
    
    print(f"\n训练完成！")
    print(f"最佳验证AUPRC: {results['best_val_auprc']:.4f}")
    print(f"测试AUPRC: {results['test_metrics']['auprc']:.4f}")
    print(f"结果保存至: {config['output_dir']}")

if __name__ == "__main__":
    main() 