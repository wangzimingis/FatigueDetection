# metrics_calculator.py - 优化版
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
import pandas as pd


class FatigueMetricsCalculator:
    """疲劳检测指标计算器（支持分类任务）"""

    def __init__(self):
        self.metrics_results = {}

    def calculate_all_metrics(self, y_true, y_pred, y_probs=None):
        """
        计算所有评价指标
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_probs: 预测概率 (n_samples, n_classes) 可选
        Returns:
            包含所有指标的字典
        """
        # 基本分类指标（加权平均）
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 各类别指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # Kappa 和 MCC
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # AUC（如果提供了概率）
        auc = None
        if y_probs is not None:
            try:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"AUC 计算失败: {e}")
                auc = 0.0

        # 灵敏度（召回率）已包含在 recall_per_class
        sensitivity = recall_per_class.copy()
        specificity = self._calculate_specificity_from_cm(cm)

        # FAR / FRR
        far, frr = self._calculate_far_frr_from_cm(cm)

        # 保存结果
        self.metrics_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc is not None else None,
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

    def _calculate_specificity_from_cm(self, cm):
        """从混淆矩阵计算每个类别的特异性（真阴性率）"""
        n_classes = cm.shape[0]
        specificity = []
        for i in range(n_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm[i, :], i))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity.append(spec)
        return specificity

    def _calculate_far_frr_from_cm(self, cm):
        """从混淆矩阵计算平均 FAR 和 FRR"""
        n_classes = cm.shape[0]
        far_list, frr_list = [], []
        for i in range(n_classes):
            fp = np.sum(np.delete(cm[:, i], i))   # 其他类被预测为 i
            fn = np.sum(np.delete(cm[i, :], i))   # 类 i 被预测为其他
            actual_i = np.sum(cm[i, :])
            actual_not_i = np.sum(np.delete(cm, i, axis=0))
            far = fp / actual_not_i if actual_not_i > 0 else 0.0
            frr = fn / actual_i if actual_i > 0 else 0.0
            far_list.append(far)
            frr_list.append(frr)
        return np.mean(far_list), np.mean(frr_list)

    def print_detailed_report(self):
        """打印详细指标报告（控制台）"""
        if not self.metrics_results:
            print("未计算指标，请先调用 calculate_all_metrics 方法。")
            return

        res = self.metrics_results
        print("\n" + "=" * 70)
        print("驾驶疲劳检测模型详细评估报告")
        print("=" * 70)

        print(f"\n整体性能指标:")
        print(f"  准确率 (Accuracy): {res['accuracy']:.4f}")
        print(f"  精确率 (Precision): {res['precision']:.4f}")
        print(f"  召回率 (Recall): {res['recall']:.4f}")
        print(f"  F1分数: {res['f1_score']:.4f}")
        if res['auc'] is not None:
            print(f"  AUC: {res['auc']:.4f}")
        print(f"  Kappa系数: {res['kappa']:.4f}")
        print(f"  马修斯相关系数: {res['mcc']:.4f}")

        print(f"\n各类别性能指标:")
        print(f"  {'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'灵敏度':<10} {'特异性':<10}")
        print("-" * 70)
        classes = ['清醒', '轻度疲劳', '重度疲劳']
        cmw = res['class_wise_metrics']
        for i, cls in enumerate(classes):
            print(f"  {cls:<10} "
                  f"{cmw['precision'][i]:<10.4f} "
                  f"{cmw['recall'][i]:<10.4f} "
                  f"{cmw['f1'][i]:<10.4f} "
                  f"{cmw['sensitivity'][i]:<10.4f} "
                  f"{cmw['specificity'][i]:<10.4f}")

        print(f"\n错误率指标:")
        print(f"  平均错误接受率 (FAR): {res['error_rates']['far']:.4f}")
        print(f"  平均错误拒绝率 (FRR): {res['error_rates']['frr']:.4f}")

        print(f"\n混淆矩阵:")
        cm = np.array(res['confusion_matrix'])
        print(f"  {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}  | 清醒")
        print(f"  {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}  | 轻度疲劳")
        print(f"  {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}  | 重度疲劳")
        print("=" * 70)

    def save_to_excel(self, filepath='./results/metrics_report.xlsx'):
        """
        保存指标到 Excel 文件（修复版：自动创建目录、异常处理）
        Args:
            filepath: 保存路径（支持相对/绝对路径）
        Returns:
            bool: 保存成功返回 True，否则 False
        """
        if not self.metrics_results:
            print("错误：尚未计算任何指标，请先调用 calculate_all_metrics 方法。")
            return False

        try:
            # 确保目录存在
            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            res = self.metrics_results

            # 整体指标 DataFrame
            df_overall = pd.DataFrame([{
                '准确率': res['accuracy'],
                '精确率': res['precision'],
                '召回率': res['recall'],
                'F1分数': res['f1_score'],
                'AUC': res['auc'] if res['auc'] is not None else 'N/A',
                'Kappa系数': res['kappa'],
                '马修斯相关系数': res['mcc'],
                '平均FAR': res['error_rates']['far'],
                '平均FRR': res['error_rates']['frr']
            }])

            # 类别指标 DataFrame
            classes = ['清醒', '轻度疲劳', '重度疲劳']
            cmw = res['class_wise_metrics']
            df_classwise = pd.DataFrame({
                '类别': classes,
                '精确率': cmw['precision'],
                '召回率': cmw['recall'],
                'F1分数': cmw['f1'],
                '灵敏度': cmw['sensitivity'],
                '特异性': cmw['specificity']
            })

            # 混淆矩阵 DataFrame
            df_confusion = pd.DataFrame(
                np.array(res['confusion_matrix']),
                index=classes,
                columns=classes
            )

            # 写入 Excel（使用 openpyxl 引擎）
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df_overall.to_excel(writer, sheet_name='整体指标', index=False)
                df_classwise.to_excel(writer, sheet_name='类别指标', index=False)
                df_confusion.to_excel(writer, sheet_name='混淆矩阵')

            print(f"指标报告已保存到: {filepath}")
            return True

        except ImportError as e:
            print(f"缺少依赖库: {e}. 请安装 pandas 和 openpyxl → pip install pandas openpyxl")
            return False
        except PermissionError:
            print(f"权限不足，无法写入文件: {filepath}。请检查文件是否被占用或目录权限。")
            return False
        except Exception as e:
            print(f"保存 Excel 失败: {e}")
            return False


def demonstrate_metrics():
    """演示指标计算方法（优化版）"""
    np.random.seed(42)
    n_samples = 300
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])
    y_pred = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    y_probs = np.random.rand(n_samples, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    calculator = FatigueMetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_probs)
    calculator.print_detailed_report()

    # 保存 Excel（测试）
    success = calculator.save_to_excel('./results/metrics_report.xlsx')
    if not success:
        print("保存失败，请检查依赖或权限。")
    return metrics


if __name__ == '__main__':
    demonstrate_metrics()