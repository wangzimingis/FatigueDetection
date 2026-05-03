# evaluator.py - 修复版本
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, matthews_corrcoef, cohen_kappa_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import pandas as pd
from datetime import datetime
import os
from config import Config

# 尝试导入中文字体设置（如果可用）
try:
    from src.utils import setup_chinese_font
    setup_chinese_font()
except ImportError:
    pass


class FatigueEvaluator:
    """疲劳检测模型评估器（修复版）"""

    def __init__(self, model_path, device='cuda'):
        """
        初始化评估器

        Args:
            model_path: 模型检查点路径（由 trainer.save_checkpoint 保存）
            device: 运行设备
        """
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        # 确保 config 对象有 task_type 属性
        if not hasattr(self.config, 'task_type'):
            self.config.task_type = getattr(self.config, 'classification_type', 'classification')
        self.device = device

        # 重建模型（避免循环导入，放在方法内部）
        from src.models import create_model
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # 加载训练历史（适配 trainer 保存的结构）
        self.history = checkpoint.get('history', {})
        if not isinstance(self.history, dict):
            self.history = {}  # 如果格式不对，置空

        # 样式设置
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_training_history(self, save_path=None):
        """
        绘制训练历史曲线（适配 trainer.py 的 history 结构）
        """
        if not self.history or not self.history.get('train_loss'):
            print("没有训练历史数据或数据格式不正确")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 提取损失
        train_losses = self.history['train_loss']
        val_losses = self.history['val_loss']

        # 提取主要指标（根据任务类型自动选择）
        if self.config.task_type == 'classification':
            train_metric = [m['f1'] for m in self.history['train_metrics']]
            val_metric = [m['f1'] for m in self.history['val_metrics']]
            metric_name = 'F1 Score'
            ch_metric_name = 'F1分数'
        else:
            train_metric = [m['rmse'] for m in self.history['train_metrics']]
            val_metric = [m['rmse'] for m in self.history['val_metrics']]
            metric_name = 'RMSE'
            ch_metric_name = 'RMSE'

        # 准确率（仅分类）
        if self.config.task_type == 'classification':
            train_acc = [m['accuracy'] for m in self.history['train_metrics']]
            val_acc = [m['accuracy'] for m in self.history['val_metrics']]

        # 学习率
        lrs = self.history.get('learning_rates', [])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 1. 损失曲线
        axes[0].plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练和验证损失', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 主指标曲线
        axes[1].plot(epochs, train_metric, 'b-', label='训练', linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, val_metric, 'r-', label='验证', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'训练和验证{ch_metric_name}', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 准确率（分类）或 相关系数（回归）
        if self.config.task_type == 'classification':
            axes[2].plot(epochs, train_acc, 'b-', label='训练', linewidth=2, marker='o', markersize=4)
            axes[2].plot(epochs, val_acc, 'r-', label='验证', linewidth=2, marker='s', markersize=4)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('准确率')
            axes[2].set_title('训练和验证准确率', fontweight='bold')
        else:
            train_corr = [m['correlation'] for m in self.history['train_metrics']]
            val_corr = [m['correlation'] for m in self.history['val_metrics']]
            axes[2].plot(epochs, train_corr, 'b-', label='训练', linewidth=2, marker='o', markersize=4)
            axes[2].plot(epochs, val_corr, 'r-', label='验证', linewidth=2, marker='s', markersize=4)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('皮尔逊相关系数')
            axes[2].set_title('训练和验证相关系数', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. 学习率曲线
        if lrs:
            axes[3].plot(epochs, lrs, 'g-', linewidth=2, marker='d', markersize=4)
            axes[3].set_yscale('log')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('学习率')
        axes[3].set_title('学习率变化', fontweight='bold')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes=None, save_path=None):
        """绘制混淆矩阵"""
        if classes is None:
            # 从 config 获取类别数，默认 3
            n_classes = getattr(self.config, 'num_classes', 3)
            if n_classes == 3:
                classes = ['清醒', '轻度疲劳', '重度疲劳']
            elif n_classes == 2:
                classes = ['非疲劳', '疲劳']
            else:
                classes = [f'Class {i}' for i in range(n_classes)]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': '样本数'})
        plt.title('疲劳状态混淆矩阵', fontsize=16, fontweight='bold')
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")

        plt.show()
        return cm

    def plot_roc_curves(self, y_true, y_probs, classes=None, save_path=None):
        """绘制多类 ROC 曲线（支持二分类和多分类）"""
        y_true = np.array(y_true)
        y_probs = np.array(y_probs)

        # 处理二分类情况
        if y_probs.ndim == 1:
            # 二分类，将概率转换为 (n_samples, 2) 格式
            n_classes = 2
            y_probs_bin = np.zeros((len(y_probs), 2))
            y_probs_bin[:, 1] = y_probs
            y_probs_bin[:, 0] = 1 - y_probs
            y_probs = y_probs_bin
        else:
            n_classes = y_probs.shape[1]

        if classes is None:
            if n_classes == 3:
                classes = ['清醒', '轻度疲劳', '重度疲劳']
            elif n_classes == 2:
                classes = ['非疲劳', '疲劳']
            else:
                classes = [f'Class {i}' for i in range(n_classes)]

        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        auc_values = {}

        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values[classes[i]] = roc_auc
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
        plt.title('多类ROC曲线', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")

        plt.show()
        return auc_values

    def plot_class_distribution(self, y_true, y_pred, classes=None, save_path=None):
        """绘制真实与预测标签分布对比图"""
        if classes is None:
            n_classes = getattr(self.config, 'num_classes', 3)
            if n_classes == 3:
                classes = ['清醒', '轻度疲劳', '重度疲劳']
            elif n_classes == 2:
                classes = ['非疲劳', '疲劳']
            else:
                classes = [f'Class {i}' for i in range(n_classes)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 真实标签分布
        true_counts = np.bincount(y_true, minlength=len(classes))
        axes[0].bar(range(len(classes)), true_counts, color='skyblue', edgecolor='black')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45)
        axes[0].set_ylabel('样本数')
        axes[0].set_title('真实标签分布', fontweight='bold')
        for i, count in enumerate(true_counts):
            axes[0].text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

        # 预测标签分布
        pred_counts = np.bincount(y_pred, minlength=len(classes))
        axes[1].bar(range(len(classes)), pred_counts, color='lightcoral', edgecolor='black')
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45)
        axes[1].set_ylabel('样本数')
        axes[1].set_title('预测标签分布', fontweight='bold')
        for i, count in enumerate(pred_counts):
            axes[1].text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图已保存到: {save_path}")

        plt.show()

    def generate_report(self, y_true, y_pred, y_probs=None, save_path=None):
        """
        生成详细评估报告（支持分类和回归）
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task_type': self.config.task_type,
            'dataset_info': {
                'total_samples': len(y_true),
            }
        }

        if self.config.task_type == 'classification':
            # 分类指标
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            kappa = cohen_kappa_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            auc_score = 0.0
            if y_probs is not None:
                try:
                    auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
                except ValueError as e:
                    print(f"AUC计算失败: {e}")
                    auc_score = 0.0

            report['dataset_info']['class_distribution'] = np.bincount(y_true).tolist()
            report['overall_metrics'] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score),
                'kappa': float(kappa),
                'mcc': float(mcc)
            }
            report['classwise_metrics'] = classification_report(y_true, y_pred, output_dict=True)
            report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

            # 打印摘要
            print("\n" + "=" * 60)
            print("模型评估报告摘要（分类）")
            print("=" * 60)
            print(f"总样本数: {len(y_true)}")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"AUC: {auc_score:.4f}")
            print(f"Kappa系数: {kappa:.4f}")
            print(f"马修斯相关系数: {mcc:.4f}")
            print("\n详细分类报告:")
            print(classification_report(y_true, y_pred))

        else:  # 回归任务
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

            report['overall_metrics'] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'correlation': float(corr)
            }

            print("\n" + "=" * 60)
            print("模型评估报告摘要（回归）")
            print("=" * 60)
            print(f"总样本数: {len(y_true)}")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"平均绝对误差 (MAE): {mae:.4f}")
            print(f"决定系数 (R²): {r2:.4f}")
            print(f"皮尔逊相关系数: {corr:.4f}")

        # 保存 JSON 报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n评估报告已保存到: {save_path}")

        return report

    def predict(self, eeg_data, eog_data=None):
        """
        预测EEG数据的疲劳状态（支持单样本和批量）

        Args:
            eeg_data: 形状为 (batch, channels, features) 或 (channels, features) 的 numpy 数组
            eog_data: 形状为 (batch, features) 或 (features,) 的 numpy 数组，可选

        Returns:
            preds: 预测类别 (分类) 或 预测值 (回归)
            probs: 分类概率 (分类) 或 None (回归)
        """
        self.model.eval()

        # 处理单个样本：增加 batch 维度
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, axis=0)
        if eog_data is not None and eog_data.ndim == 1:
            eog_data = np.expand_dims(eog_data, axis=0)

        # 转换为张量并移到设备
        eeg_tensor = torch.FloatTensor(eeg_data).to(self.device)
        if eog_data is not None:
            eog_tensor = torch.FloatTensor(eog_data).to(self.device)
        else:
            eog_tensor = None

        with torch.no_grad():
            if eog_tensor is not None and getattr(self.config, 'use_eog', False):
                outputs = self.model(eeg_tensor, eog_tensor)
            else:
                outputs = self.model(eeg_tensor)

        if self.config.task_type == 'classification':
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            return preds, probs
        else:
            preds = outputs.squeeze().cpu().numpy()
            return preds, None


# 单元测试（可选）
if __name__ == '__main__':
    # 简单测试：创建一个虚拟模型并保存检查点
    import torch.nn as nn
    from config import Config

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(25, 3)
        def forward(self, x, eog=None):
            return self.fc(x)

    config = Config()
    config.task_type = 'classification'
    config.num_classes = 3
    config.use_eog = False

    model = DummyModel()
    dummy_checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict(),
        'history': {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'train_metrics': [{'f1': 0.7}, {'f1': 0.75}, {'f1': 0.8}],
            'val_metrics': [{'f1': 0.65}, {'f1': 0.7}, {'f1': 0.75}],
            'learning_rates': [1e-3, 1e-3, 1e-4]
        }
    }
    os.makedirs('./test_checkpoints', exist_ok=True)
    torch.save(dummy_checkpoint, './test_checkpoints/dummy.pth')

    evaluator = FatigueEvaluator('./test_checkpoints/dummy.pth', device='cpu')
    evaluator.plot_training_history()
    print("测试完成，请手动关闭图形窗口。")