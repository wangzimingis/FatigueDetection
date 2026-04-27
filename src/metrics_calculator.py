# metrics_calculator.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
import pandas as pd


class FatigueMetricsCalculator:
    """疲劳检测指标计算器"""

    def __init__(self):
        self.metrics_results = {}

    def calculate_all_metrics(self, y_true, y_pred, y_probs=None):
        """
        计算所有评价指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率

        Returns:
            包含所有指标的字典
        """
        # 基本分类指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # 各类别指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 其他指标
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # 计算AUC（如果提供了概率）
        auc = None
        if y_probs is not None:
            try:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
            except:
                auc = 0.0

        # 计算灵敏度（召回率）和特异性
        sensitivity = recall_score(y_true, y_pred, average=None)
        specificity = self.calculate_specificity(y_true, y_pred, cm)

        # 计算FAR和FRR
        far, frr = self.calculate_far_frr(cm)

        # 保存结果
        self.metrics_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc else None,
            'kappa': float(kappa),
            'mcc': float(mcc),
            'confusion_matrix': cm.tolist(),
            'class_wise_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist(),
                'sensitivity': sensitivity.tolist(),
                'specificity': specificity
            },
            'error_rates': {
                'far': far,
                'frr': frr
            }
        }

        return self.metrics_results

    def calculate_specificity(self, y_true, y_pred, cm):
        """
        计算各类别的特异性

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            cm: 混淆矩阵

        Returns:
            各类别的特异性
        """
        n_classes = cm.shape[0]
        specificity = []

        for i in range(n_classes):
            # 真阴性：所有不是类别i且正确预测为不是类别i的样本
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))

            # 假阳性：所有不是类别i但预测为类别i的样本
            fp = np.sum(np.delete(cm[i, :], i))

            # 特异性 = TN / (TN + FP)
            if tn + fp > 0:
                specificity.append(tn / (tn + fp))
            else:
                specificity.append(0.0)

        return specificity

    def calculate_far_frr(self, cm):
        """
        计算错误接受率(FAR)和错误拒绝率(FRR)

        Args:
            cm: 混淆矩阵

        Returns:
            FAR, FRR
        """
        # 对于三分类问题，我们可以分别计算各类别的错误率
        n_classes = cm.shape[0]

        far_per_class = []
        frr_per_class = []

        for i in range(n_classes):
            # 错误接受：将其他类预测为本类
            fp = np.sum(np.delete(cm[:, i], i))

            # 错误拒绝：将本类预测为其他类
            fn = np.sum(np.delete(cm[i, :], i))

            # 真实本类样本数
            actual_i = np.sum(cm[i, :])

            # 真实非本类样本数
            actual_not_i = np.sum(np.delete(cm, i, axis=0))

            if actual_not_i > 0:
                far = fp / actual_not_i
            else:
                far = 0.0

            if actual_i > 0:
                frr = fn / actual_i
            else:
                frr = 0.0

            far_per_class.append(far)
            frr_per_class.append(frr)

        # 计算平均错误率
        avg_far = np.mean(far_per_class)
        avg_frr = np.mean(frr_per_class)

        return avg_far, avg_frr

    def print_detailed_report(self):
        """打印详细指标报告"""
        if not self.metrics_results:
            print("未计算指标")
            return

        print("\n" + "=" * 70)
        print("驾驶疲劳检测模型详细评估报告")
        print("=" * 70)

        print(f"\n整体性能指标:")
        print(f"  准确率 (Accuracy): {self.metrics_results['accuracy']:.4f}")
        print(f"  精确率 (Precision): {self.metrics_results['precision']:.4f}")
        print(f"  召回率 (Recall): {self.metrics_results['recall']:.4f}")
        print(f"  F1分数: {self.metrics_results['f1_score']:.4f}")

        if self.metrics_results['auc']:
            print(f"  AUC: {self.metrics_results['auc']:.4f}")

        print(f"  Kappa系数: {self.metrics_results['kappa']:.4f}")
        print(f"  马修斯相关系数: {self.metrics_results['mcc']:.4f}")

        print(f"\n各类别性能指标:")
        print(f"  {'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'灵敏度':<10} {'特异性':<10}")
        print("-" * 70)

        classes = ['清醒', '轻度疲劳', '重度疲劳']
        for i, cls in enumerate(classes):
            print(f"  {cls:<10} "
                  f"{self.metrics_results['class_wise_metrics']['precision'][i]:<10.4f} "
                  f"{self.metrics_results['class_wise_metrics']['recall'][i]:<10.4f} "
                  f"{self.metrics_results['class_wise_metrics']['f1'][i]:<10.4f} "
                  f"{self.metrics_results['class_wise_metrics']['sensitivity'][i]:<10.4f} "
                  f"{self.metrics_results['class_wise_metrics']['specificity'][i]:<10.4f}")

        print(f"\n错误率指标:")
        print(f"  平均错误接受率 (FAR): {self.metrics_results['error_rates']['far']:.4f}")
        print(f"  平均错误拒绝率 (FRR): {self.metrics_results['error_rates']['frr']:.4f}")

        print(f"\n混淆矩阵:")
        cm = np.array(self.metrics_results['confusion_matrix'])
        print(f"  {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}  | 清醒")
        print(f"  {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}  | 轻度疲劳")
        print(f"  {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}  | 重度疲劳")

        print("=" * 70)

    def save_to_excel(self, filepath='./results/metrics_report.xlsx'):
        """
        保存指标到Excel文件

        Args:
            filepath: 文件路径
        """
        import pandas as pd

        # 创建DataFrame
        df_overall = pd.DataFrame([{
            '准确率': self.metrics_results['accuracy'],
            '精确率': self.metrics_results['precision'],
            '召回率': self.metrics_results['recall'],
            'F1分数': self.metrics_results['f1_score'],
            'AUC': self.metrics_results['auc'] if self.metrics_results['auc'] else 'N/A',
            'Kappa系数': self.metrics_results['kappa'],
            '马修斯相关系数': self.metrics_results['mcc'],
            '平均FAR': self.metrics_results['error_rates']['far'],
            '平均FRR': self.metrics_results['error_rates']['frr']
        }])

        # 类别指标DataFrame
        classes = ['清醒', '轻度疲劳', '重度疲劳']
        df_classwise = pd.DataFrame({
            '类别': classes,
            '精确率': self.metrics_results['class_wise_metrics']['precision'],
            '召回率': self.metrics_results['class_wise_metrics']['recall'],
            'F1分数': self.metrics_results['class_wise_metrics']['f1'],
            '灵敏度': self.metrics_results['class_wise_metrics']['sensitivity'],
            '特异性': self.metrics_results['class_wise_metrics']['specificity']
        })

        # 混淆矩阵DataFrame
        df_confusion = pd.DataFrame(
            np.array(self.metrics_results['confusion_matrix']),
            index=classes,
            columns=classes
        )

        # 保存到Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_overall.to_excel(writer, sheet_name='整体指标', index=False)
            df_classwise.to_excel(writer, sheet_name='类别指标', index=False)
            df_confusion.to_excel(writer, sheet_name='混淆矩阵')

        print(f"指标报告已保存到: {filepath}")


# 使用示例
def demonstrate_metrics():
    """演示指标计算方法"""
    # 模拟数据
    np.random.seed(42)

    # 生成模拟标签和预测
    n_samples = 300
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])
    y_pred = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    y_probs = np.random.rand(n_samples, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    # 计算指标
    calculator = FatigueMetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_probs)

    # 打印报告
    calculator.print_detailed_report()

    # 保存到Excel
    calculator.save_to_excel()

    return metrics


if __name__ == '__main__':
    demonstrate_metrics()