# trainer.py - 优化版本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
import json
import copy
import torch.nn.functional as F
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd

try:
    from src.utils import _chinese_font_available
except ImportError:
    _chinese_font_available = False

from src.models import create_model
from config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    weight: 类别权重 (1D Tensor, 长度 = num_classes) 或 None
    gamma: 聚焦参数，越大越关注难分样本（常用 2）
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight.clone().detach())
        else:
            self.register_buffer('weight', torch.ones(1))  # 占位，后续会被覆盖

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FatigueTrainer:
    """疲劳检测模型训练器 - 优化版（支持余弦退火）"""

    def __init__(self, model: nn.Module, config: Config, device: Optional[torch.device] = None):
        self.model = model
        self.config = config

        self.device = torch.device(config.device) if device is None else device
        self.model.to(self.device)

        self.task_type = getattr(config, 'task_type',
                                 getattr(config, 'classification_type', 'classification'))
        self._train_loader_for_weights = None

        # 1. 确定度量模式
        if self.task_type == 'classification':
            self.metric_mode = 'max'
        else:
            self.metric_mode = 'min'

        # 2. 损失函数
        class_weights = self._compute_class_weights_from_dataloader()
        if self.task_type == 'classification':
            if class_weights is None:
                class_weights = torch.ones(config.num_classes, device=self.device)
            self.criterion = FocalLoss(weight=class_weights, gamma=2)
        else:
            self.criterion = nn.MSELoss()

        # 3. 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # 4. 学习率调度器 —— 根据配置选择
        scheduler_type = getattr(config, 'scheduler_type', 'plateau')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            if self.task_type == 'classification':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.5, patience=config.patience, verbose=True
                )
            else:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=config.patience, verbose=True
                )

        self.scaler = GradScaler(enabled=config.use_mixed_precision)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # 最佳模型状态
        self.best_metric = float('inf') if self.metric_mode == 'min' else 0.0
        self.best_model_state = None
        self.patience_counter = 0

        print(f"训练器初始化完成 | 设备: {self.device} | 混合精度: {config.use_mixed_precision}")
        print(f"任务类型: {'分类' if self.task_type == 'classification' else '回归'}")
        print(f"调度器: {scheduler_type}")

    def _compute_class_weights_from_dataloader(self) -> Optional[torch.Tensor]:
        """从训练数据加载器计算类别权重"""
        if self.task_type != 'classification' or self._train_loader_for_weights is None:
            return None

        all_labels = []
        for batch in self._train_loader_for_weights:
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().numpy())
            else:
                all_labels.extend(labels)

        unique, counts = np.unique(all_labels, return_counts=True)
        if len(unique) < 2:
            return None

        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.sum() * len(unique)
        # 可在此对轻度疲劳（index=1）增加权重，如：class_weights[1] *= 1.5
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        print(f"类别权重: {dict(zip(unique, class_weights))}")
        return weights_tensor

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d} [训练]', leave=False)

        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg'].to(self.device)
            labels = batch['label'].to(self.device)

            eog = batch.get('eog').to(self.device) if 'eog' in batch and self.config.use_eog else None

            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(eeg, eog)

                if self.task_type == 'classification':
                    labels_long = labels.squeeze().long()
                    loss = self.criterion(outputs, labels_long)
                else:
                    labels_float = labels.float().squeeze()
                    pred = outputs.squeeze() if outputs.numel() == labels_float.numel() else outputs
                    loss = self.criterion(pred, labels_float)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.task_type == 'classification':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                preds = outputs.detach().squeeze().cpu().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds])

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}',
                              'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})

        train_loss = total_loss / len(train_loader)
        train_metrics = self._compute_metrics(all_labels, all_preds)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

        return train_loss, train_metrics

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='[验证]', leave=False):
                eeg = batch['eeg'].to(self.device)
                labels = batch['label'].to(self.device)

                eog = batch.get('eog').to(self.device) if 'eog' in batch and self.config.use_eog else None

                outputs = self.model(eeg, eog)

                if self.task_type == 'classification':
                    labels_long = labels.squeeze().long()
                    loss = self.criterion(outputs, labels_long)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                else:
                    labels_float = labels.float().squeeze()
                    pred = outputs.squeeze() if outputs.numel() == labels_float.numel() else outputs
                    loss = self.criterion(pred, labels_float)
                    preds = pred.cpu().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])

                total_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_metrics = self._compute_metrics(all_labels, all_preds, all_probs if all_probs else None)

        return val_loss, val_metrics, (all_preds, all_labels, all_probs)

    def save_full_training_history(self, filepath=None, overwrite=True):
        if not self.history or not self.history.get('train_loss'):
            print("没有训练历史数据，无法保存。")
            return False

        if filepath is None:
            filepath = os.path.join(self.config.result_dir, 'training_history.xlsx')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        epochs = list(range(1, len(self.history['train_loss']) + 1))
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']

        if self.task_type == 'classification':
            train_f1 = [m['f1'] for m in self.history['train_metrics']]
            val_f1 = [m['f1'] for m in self.history['val_metrics']]
            train_acc = [m['accuracy'] for m in self.history['train_metrics']]
            val_acc = [m['accuracy'] for m in self.history['val_metrics']]

            df = pd.DataFrame({
                'Epoch': epochs,
                'Train_Loss': train_loss,
                'Val_Loss': val_loss,
                'Train_F1': train_f1,
                'Val_F1': val_f1,
                'Train_Accuracy': train_acc,
                'Val_Accuracy': val_acc
            })
        else:
            train_rmse = [m['rmse'] for m in self.history['train_metrics']]
            val_rmse = [m['rmse'] for m in self.history['val_metrics']]
            train_r2 = [m['r2'] for m in self.history['train_metrics']]
            val_r2 = [m['r2'] for m in self.history['val_metrics']]

            df = pd.DataFrame({
                'Epoch': epochs,
                'Train_Loss': train_loss,
                'Val_Loss': val_loss,
                'Train_RMSE': train_rmse,
                'Val_RMSE': val_rmse,
                'Train_R2': train_r2,
                'Val_R2': val_r2
            })

        if self.history.get('learning_rates'):
            df['Learning_Rate'] = self.history['learning_rates']

        try:
            if overwrite:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Training_History', index=False)
            else:
                if os.path.exists(filepath):
                    with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                        sheet_name = f'Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"已追加新工作表 {sheet_name} 到 {filepath}")
                else:
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Training_History', index=False)
            print(f"训练历史已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"保存训练历史失败: {e}")
            return False

    def _compute_metrics(self, true_labels, pred_labels, pred_probs=None):
        true_labels = np.array(true_labels).flatten()
        pred_labels = np.array(pred_labels).flatten()

        metrics = {}

        if self.task_type == 'classification':
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted', zero_division=0
            )
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            if pred_probs is not None:
                try:
                    if isinstance(pred_probs, list):
                        pred_probs = np.array(pred_probs)
                    if pred_probs.ndim == 2 and pred_probs.shape[1] > 1:
                        auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='weighted')
                        metrics['auc'] = auc
                except Exception as e:
                    print(f"AUC计算失败: {e}")
                    metrics['auc'] = 0.0

            metrics['kappa'] = cohen_kappa_score(true_labels, pred_labels)
            metrics['mcc'] = matthews_corrcoef(true_labels, pred_labels)
            metrics['main_metric'] = f1
        else:
            mse = mean_squared_error(true_labels, pred_labels)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_labels, pred_labels)
            r2 = r2_score(true_labels, pred_labels)
            correlation = np.corrcoef(true_labels, pred_labels)[0, 1] if len(true_labels) > 1 else 0.0
            metrics.update({
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'main_metric': rmse
            })

        return metrics

    def train(self, train_loader, val_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.epochs

        self._train_loader_for_weights = train_loader
        # 开始训练前更新权重（若FocalLoss支持）
        if self.task_type == 'classification':
            weights = self._compute_class_weights_from_dataloader()
            if weights is not None and hasattr(self.criterion, 'weight'):
                self.criterion.weight.copy_(weights)

        print(f"\n开始训练，共 {num_epochs} 个 epochs")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print('='*60)

            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(train_metrics)

            val_loss, val_metrics, _ = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # 根据调度器类型更新学习率
            scheduler_type = getattr(self.config, 'scheduler_type', 'plateau')
            if scheduler_type == 'cosine':
                self.scheduler.step()  # 余弦退火不需要指标
            else:
                self.scheduler.step(val_metrics['main_metric'])

            self._print_metrics(train_loss, train_metrics, val_loss, val_metrics)

            self.save_full_training_history()

            self._log_to_tensorboard(epoch, train_loss, train_metrics, val_loss, val_metrics)

            current_metric = val_metrics['main_metric']
            if self.metric_mode == 'min':
                is_better = current_metric < self.best_metric
            else:
                is_better = current_metric > self.best_metric

            if is_better:
                improvement = abs(current_metric - self.best_metric)
                print(f"🎉 发现更好的模型: {current_metric:.4f} "
                      f"({'<' if self.metric_mode == 'min' else '>'} {self.best_metric:.4f}) "
                      f"提升 {improvement:.4f}")
                self.best_metric = current_metric
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1

            if epoch % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            if epoch % self.config.plot_freq == 0:
                self.plot_training_history()

            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"⏹️  早停触发，在 epoch {epoch} 停止训练")
                break

        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成！总时间: {training_time:.2f} 秒")
        print(f"最佳指标: {self.best_metric:.4f}")
        print('='*60)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            print("已加载最佳模型状态")

        self.plot_training_history()

        return self.best_metric

    def _print_metrics(self, train_loss, train_metrics, val_loss, val_metrics):
        print(f"\n训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        if self.task_type == 'classification':
            print(f"训练准确率: {train_metrics['accuracy']:.4f} | 验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"训练 F1: {train_metrics['f1']:.4f} | 验证 F1: {val_metrics['f1']:.4f}")
            if 'auc' in val_metrics:
                print(f"验证 AUC: {val_metrics['auc']:.4f}")
        else:
            print(f"训练 RMSE: {train_metrics['rmse']:.4f} | 验证 RMSE: {val_metrics['rmse']:.4f}")
            print(f"训练 R²: {train_metrics['r2']:.4f} | 验证 R²: {val_metrics['r2']:.4f}")

    def _log_to_tensorboard(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        self.writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        self.writer.add_scalars('Main_Metric', {
            'Train': train_metrics['main_metric'],
            'Validation': val_metrics['main_metric']
        }, epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

        if self.task_type == 'classification':
            self.writer.add_scalars('Accuracy', {'Train': train_metrics['accuracy'],
                                                 'Validation': val_metrics['accuracy']}, epoch)
            self.writer.add_scalars('F1_Score', {'Train': train_metrics['f1'],
                                                 'Validation': val_metrics['f1']}, epoch)
            if 'auc' in val_metrics:
                self.writer.add_scalar('Validation_AUC', val_metrics['auc'], epoch)
        else:
            self.writer.add_scalars('RMSE', {'Train': train_metrics['rmse'],
                                             'Validation': val_metrics['rmse']}, epoch)
            self.writer.add_scalars('R2', {'Train': train_metrics['r2'],
                                           'Validation': val_metrics['r2']}, epoch)

    def plot_training_history(self, save_path=None):
        if not self.history['train_loss']:
            print("没有训练历史数据，跳过绘图")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 损失曲线
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('训练和验证损失', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 主指标曲线
        ax = axes[1]
        train_main = [m['main_metric'] for m in self.history['train_metrics']]
        val_main = [m['main_metric'] for m in self.history['val_metrics']]
        ax.plot(epochs, train_main, 'b-', label='训练', linewidth=2)
        ax.plot(epochs, val_main, 'r-', label='验证', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('主要指标')
        title = 'F1分数' if self.task_type == 'classification' else 'RMSE'
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 学习率曲线
        ax = axes[2]
        ax.plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('学习率')
        ax.set_title('学习率变化', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if self.task_type == 'classification':
            ax = axes[3]
            train_acc = [m['accuracy'] for m in self.history['train_metrics']]
            val_acc = [m['accuracy'] for m in self.history['val_metrics']]
            ax.plot(epochs, train_acc, 'b-', label='训练准确率', linewidth=2)
            ax.plot(epochs, val_acc, 'r-', label='验证准确率', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('准确率')
            ax.set_title('准确率', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[4]
            train_f1 = [m['f1'] for m in self.history['train_metrics']]
            val_f1 = [m['f1'] for m in self.history['val_metrics']]
            ax.plot(epochs, train_f1, 'b-', label='训练F1', linewidth=2)
            ax.plot(epochs, val_f1, 'r-', label='验证F1', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1分数')
            ax.set_title('F1分数', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[5]
            if self.history['val_metrics'] and 'auc' in self.history['val_metrics'][0]:
                val_auc = [m.get('auc', 0) for m in self.history['val_metrics']]
                ax.plot(epochs, val_auc, 'purple', label='验证AUC', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('AUC')
                ax.set_title('AUC曲线', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        else:
            # 回归绘图（略，同上结构）
            pass

        plt.suptitle('训练历史可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.config.figure_dir, 'training_history.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'best_metric': self.best_metric,
            'config': self.config
        }
        save_path = os.path.join(self.config.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"检查点保存到: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        print(f"从 {checkpoint_path} 加载检查点 (epoch {checkpoint.get('epoch', '?')})")

    def evaluate(self, test_loader, save_figures=True):
        val_loss, val_metrics, (preds, labels, probs) = self.validate(test_loader)

        print(f"\n{'='*60}")
        print("模型评估结果")
        print('='*60)

        if self.task_type == 'classification':
            print(f"准确率: {val_metrics['accuracy']:.4f}")
            print(f"精确率: {val_metrics['precision']:.4f}")
            print(f"召回率: {val_metrics['recall']:.4f}")
            print(f"F1分数: {val_metrics['f1']:.4f}")
            print(f"Kappa系数: {val_metrics['kappa']:.4f}")
            print(f"马修斯相关系数: {val_metrics['mcc']:.4f}")
            if 'auc' in val_metrics:
                print(f"AUC: {val_metrics['auc']:.4f}")

            target_names = ['非疲劳', '轻度疲劳', '重度疲劳']
            print("\n详细分类报告:")
            print(classification_report(labels, preds, target_names=target_names))

            cm = confusion_matrix(labels, preds)
            if save_figures:
                self.plot_confusion_matrix(cm)
            else:
                print(f"混淆矩阵:\n{cm}")

            if probs is not None and len(np.unique(labels)) > 1:
                if save_figures:
                    self.plot_roc_curves(labels, probs)
        else:
            print(f"均方误差 (MSE): {val_metrics['mse']:.4f}")
            print(f"均方根误差 (RMSE): {val_metrics['rmse']:.4f}")
            print(f"平均绝对误差 (MAE): {val_metrics['mae']:.4f}")
            print(f"决定系数 (R²): {val_metrics['r2']:.4f}")
            print(f"皮尔逊相关系数: {val_metrics['correlation']:.4f}")
            if save_figures:
                self.plot_regression_results(labels, preds)

        return val_metrics

    def plot_confusion_matrix(self, cm, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['非疲劳', '轻度疲劳', '重度疲劳'],
                    yticklabels=['非疲劳', '轻度疲劳', '重度疲劳'])
        plt.title('疲劳状态混淆矩阵', fontsize=16, fontweight='bold')
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.config.figure_dir, 'confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, labels, probs, save_path=None):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        labels = np.array(labels).flatten()
        probs = np.array(probs)

        n_classes = probs.shape[1] if probs.ndim > 1 else 1
        if n_classes == 1:
            print("二分类暂不支持多类ROC，跳过")
            return

        labels_bin = label_binarize(labels, classes=list(range(n_classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            try:
                fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            except Exception as e:
                print(f"类别 {i} ROC 计算失败: {e}")

        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        class_names = ['非疲劳', '轻度疲劳', '重度疲劳'][:n_classes]

        for i in range(n_classes):
            if i in fpr:
                plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                         label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
        plt.title('多类ROC曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path is None:
            save_path = os.path.join(self.config.figure_dir, 'roc_curves.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_regression_results(self, true_values, pred_values, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ax = axes[0]
        ax.scatter(true_values, pred_values, alpha=0.6, s=50)
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
        ax.set_xlabel('真实值', fontsize=12)
        ax.set_ylabel('预测值', fontsize=12)
        ax.set_title('预测值 vs 真实值', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        residuals = np.array(pred_values) - np.array(true_values)
        ax.scatter(pred_values, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('预测值', fontsize=12)
        ax.set_ylabel('残差', fontsize=12)
        ax.set_title('残差图', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle('回归分析结果', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.config.figure_dir, 'regression_results.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== 交叉验证与超参数调优 ====================
def cross_validation_training(config: Config, subject_ids: Optional[List[int]] = None,
                              n_folds: int = 5, verbose: bool = True) -> Dict:
    from sklearn.model_selection import KFold, StratifiedKFold
    from src.data_loader import SEEDVIGDataset

    if subject_ids is None:
        subject_ids = list(range(1, config.num_subjects + 1))

    dataset = SEEDVIGDataset(config, subject_ids=subject_ids, mode='train')

    task_type = getattr(config, 'task_type', 'classification')
    if task_type == 'classification':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        split_generator = kf.split(dataset.data['eeg'], dataset.data['labels'])
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        split_generator = kf.split(dataset.data['eeg'])

    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"开始 {n_folds} 折交叉验证")
    print('='*60)

    for fold, (train_idx, val_idx) in enumerate(split_generator, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}")
        print('='*60)

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        fold_config = copy.deepcopy(config)
        model = create_model(fold_config)
        trainer = FatigueTrainer(model, fold_config)
        best_metric = trainer.train(train_loader, val_loader)
        val_metrics = trainer.evaluate(val_loader, save_figures=False)
        fold_metrics.append(val_metrics)

        print(f"Fold {fold} 完成 | 最佳主指标: {best_metric:.4f}")

    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print('='*60)

    avg_metrics = {}
    metric_keys = [k for k in fold_metrics[0].keys() if k != 'main_metric']
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    main_values = [m['main_metric'] for m in fold_metrics]
    avg_metrics['main_metric'] = {
        'mean': float(np.mean(main_values)),
        'std': float(np.std(main_values)),
        'values': main_values
    }

    for key, stats in avg_metrics.items():
        if key == 'main_metric':
            print(f"主指标          | 均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            print(f"{key.upper():15s} | 均值: {stats['mean']:.4f} ± {stats['std']:.4f}")

    results = {
        'config': config.to_dict() if hasattr(config, 'to_dict') else config.__dict__,
        'fold_metrics': fold_metrics,
        'average_metrics': avg_metrics,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(config.result_dir, exist_ok=True)
    results_path = os.path.join(config.result_dir, f'cross_validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n交叉验证结果已保存到: {results_path}")
    return avg_metrics


def hyperparameter_tuning(config: Config, param_grid: Dict[str, List],
                          n_folds: int = 3, verbose: bool = True) -> Tuple[Dict, float]:
    from sklearn.model_selection import ParameterGrid

    best_params = None
    best_score = float('inf') if getattr(config, 'task_type', 'classification') == 'regression' else 0.0
    param_combinations = list(ParameterGrid(param_grid))
    print(f"超参数调优: 共 {len(param_combinations)} 种组合")

    results = []

    for i, params in enumerate(param_combinations, 1):
        print(f"\n{'='*60}")
        print(f"组合 {i}/{len(param_combinations)}")
        print(f"参数: {params}")
        print('='*60)

        tuned_config = copy.deepcopy(config)
        for key, value in params.items():
            if hasattr(tuned_config, key):
                setattr(tuned_config, key, value)
            else:
                print(f"警告: 配置中没有属性 {key}，跳过")

        try:
            avg_metrics = cross_validation_training(tuned_config, n_folds=n_folds, verbose=False)
            main_metric = avg_metrics['main_metric']['mean']
            results.append({'params': params, 'score': main_metric, 'metrics': avg_metrics})

            task_type = getattr(config, 'task_type', 'classification')
            if task_type == 'regression':
                if main_metric < best_score:
                    best_score = main_metric
                    best_params = params
            else:
                if main_metric > best_score:
                    best_score = main_metric
                    best_params = params

            print(f"当前得分: {main_metric:.4f} | 最佳得分: {best_score:.4f}")
        except Exception as e:
            print(f"参数组合失败: {e}")
            continue

    tuning_results = {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results,
        'param_grid': param_grid,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(config.result_dir, exist_ok=True)
    tuning_path = os.path.join(config.result_dir, f'hyperparameter_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(tuning_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("超参数调优完成")
    print(f"最佳参数: {best_params}")
    print(f"最佳得分: {best_score:.4f}")
    print('='*60)
    return best_params, best_score


__all__ = ['FatigueTrainer', 'cross_validation_training', 'hyperparameter_tuning']