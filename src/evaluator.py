#evaluator.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, matthews_corrcoef, cohen_kappa_score
)
import pandas as pd
from datetime import datetime
import os


class FatigueEvaluator:
    """疲劳检测模型评估器"""

    def __init__(self, model_path, device='cuda'):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            device: 运行设备
        """
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']

        # 重建模型
        from models import create_model
        self.model = create_model(self.config)

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.metrics = checkpoint.get('metrics', {})
        self.history = checkpoint.get('val_history', [])

        # 设置样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_training_history(self, save_path=None):
        """
        绘制训练历史曲线

        Args:
            save_path: 保存路径
        """
        if not self.history:
            print("没有训练历史数据")
            return

        epochs = [h['epoch'] for h in self.history]

        # 提取指标
        train_losses = [h.get('loss', 0) for h in self.history]
        val_losses = [h.get('loss', 0) for h in self.history]
        train_accs = [h.get('accuracy', 0) for h in self.history]
        val_accs = [h.get('accuracy', 0) for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(epochs, train_losses, label='训练损失', linewidth=2, marker='o', markersize=4)
        axes[0, 0].plot(epochs, val_losses, label='验证损失', linewidth=2, marker='s', markersize=4)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[0, 1].plot(epochs, train_accs, label='训练准确率', linewidth=2, marker='o', markersize=4)
        axes[0, 1].plot(epochs, val_accs, label='验证准确率', linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('训练和验证准确率', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # F1分数曲线
        if 'f1' in self.history[0]:
            f1_scores = [h['f1'] for h in self.history]
            axes[1, 0].plot(epochs, f1_scores, label='F1分数', color='green',
                            linewidth=2, marker='^', markersize=4)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('F1 Score', fontsize=12)
            axes[1, 0].set_title('验证F1分数', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)

        # AUC曲线
        if 'auc' in self.history[0]:
            auc_scores = [h['auc'] for h in self.history]
            axes[1, 1].plot(epochs, auc_scores, label='AUC', color='purple',
                            linewidth=2, marker='d', markersize=4)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('AUC', fontsize=12)
            axes[1, 1].set_title('验证AUC', fontsize=14, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes=None, save_path=None):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            classes: 类别名称
            save_path: 保存路径

        Returns:
            混淆矩阵
        """
        if classes is None:
            classes = ['清醒', '轻度疲劳', '重度疲劳']

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
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_probs: 预测概率
            classes: 类别名称
            save_path: 保存路径

        Returns:
            ROC曲线AUC值
        """
        if classes is None:
            classes = ['清醒', '轻度疲劳', '重度疲劳']

        n_classes = y_probs.shape[1]

        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        auc_values = {}
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values[classes[i]] = roc_auc

            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{classes[i]} (AUC = {roc_auc:.3f})')

        # 绘制对角线
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
        """
        绘制类别分布图

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            classes: 类别名称
            save_path: 保存路径
        """
        if classes is None:
            classes = ['清醒', '轻度疲劳', '重度疲劳']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 真实标签分布
        true_counts = np.bincount(y_true, minlength=len(classes))
        axes[0].bar(range(len(classes)), true_counts, color='skyblue', edgecolor='black')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45)
        axes[0].set_ylabel('样本数', fontsize=12)
        axes[0].set_title('真实标签分布', fontsize=14, fontweight='bold')

        # 在柱状图上添加数值
        for i, count in enumerate(true_counts):
            axes[0].text(i, count + max(true_counts) * 0.01, str(count),
                         ha='center', va='bottom', fontsize=10)

        # 预测标签分布
        pred_counts = np.bincount(y_pred, minlength=len(classes))
        axes[1].bar(range(len(classes)), pred_counts, color='lightcoral', edgecolor='black')
        axes[1].set_xticks(range(len(classes)))
        axes[1].set_xticklabels(classes, rotation=45)
        axes[1].set_ylabel('样本数', fontsize=12)
        axes[1].set_title('预测标签分布', fontsize=14, fontweight='bold')

        # 在柱状图上添加数值
        for i, count in enumerate(pred_counts):
            axes[1].text(i, count + max(pred_counts) * 0.01, str(count),
                         ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图已保存到: {save_path}")

        plt.show()

    def generate_report(self, y_true, y_pred, y_probs, save_path=None):
        """
        生成详细评估报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率
            save_path: 报告保存路径

        Returns:
            评估报告字典
        """
        # 计算各项指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        try:
            auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        except:
            auc_score = 0.0

        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # 各类别指标
        class_report = classification_report(y_true, y_pred, output_dict=True)

        # 创建报告字典
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_samples': len(y_true),
                'class_distribution': np.bincount(y_true).tolist()
            },
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score),
                'kappa': float(kappa),
                'mcc': float(mcc)
            },
            'classwise_metrics': class_report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'model_config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }

        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"评估报告已保存到: {save_path}")

        # 打印报告摘要
        print("\n" + "=" * 60)
        print("模型评估报告摘要")
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

        return report

    def predict(self, eeg_data, eog_data=None):
        """
        预测EEG数据的疲劳状态

        Args:
            eeg_data: EEG数据，形状为 (n_samples, seq_len, channels, features)
            eog_data: EOG数据，形状为 (n_samples, features)

        Returns:
            预测结果和概率
        """
        self.model.eval()

        # 处理单个样本的情况
        if len(eeg_data.shape) == 3:  # (channels, features, seq_len) 或类似
            eeg_data = np.expand_dims(eeg_data, 0)

        # 确保形状正确
        if len(eeg_data.shape) == 4:  # (n_samples, channels, features, seq_len)
            # 转换为 (n_samples, seq_len, channels, features)
            eeg_data = np.transpose(eeg_data, (0, 3, 1, 2))

        # 转换为张量
        eeg_tensor = torch.FloatTensor(eeg_data).to(self.device)

        if eog_data is not None:
            eog_tensor = torch.FloatTensor(eog_data).to(self.device)
        else:
            eog_tensor = None

        with torch.no_grad():
            if eog_tensor is not None:
                logits = self.model(eeg_tensor, eog_tensor)
            else:
                logits = self.model(eeg_tensor)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()