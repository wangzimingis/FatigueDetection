# utils.py - 修复中文显示问题的完整版本
import numpy as np
import torch
import random
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import warnings
import sys

warnings.filterwarnings('ignore')


# ========================== 中文字体配置 ==========================
def setup_chinese_font():
    """设置Matplotlib中文字体"""
    try:
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'PingFang',  # 苹方（macOS）
            'Source Han Sans',  # 思源黑体
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'DejaVu Sans',  # 备选英文字体
            'Arial'  # 备选英文字体
        ]

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')

        # 验证字体是否设置成功
        test_font = matplotlib.font_manager.FontProperties(
            family=plt.rcParams['font.sans-serif'][0]
        )

        print(f"✅ 中文字体设置成功: {plt.rcParams['font.sans-serif'][0]}")
        return True

    except Exception as e:
        print(f"⚠️  中文字体设置失败: {e}")
        print("尝试备选方案...")

        # 备选方案：使用英文标签
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ 已设置英文字体")
            return False
        except Exception as e2:
            print(f"❌ 字体设置完全失败: {e2}")
            return False


# 初始化字体设置
_chinese_font_available = setup_chinese_font()


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")


def check_gpu():
    """检查GPU信息"""
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if gpu_info['available']:
        current_device = torch.cuda.current_device()
        gpu_info['device_name'] = torch.cuda.get_device_name(current_device)
        gpu_info['current_device'] = current_device

        # 获取GPU内存信息
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info['total_memory_gb'] = gpu.memoryTotal / 1024
                gpu_info['free_memory_gb'] = gpu.memoryFree / 1024
                gpu_info['used_memory_gb'] = gpu.memoryUsed / 1024
            else:
                gpu_info['total_memory_gb'] = 0
                gpu_info['free_memory_gb'] = 0
                gpu_info['used_memory_gb'] = 0
        except ImportError:
            gpu_info['total_memory_gb'] = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            gpu_info['free_memory_gb'] = 0
            gpu_info['used_memory_gb'] = 0
    else:
        gpu_info['device_name'] = 'CPU'
        gpu_info['current_device'] = None
        gpu_info['total_memory_gb'] = 0
        gpu_info['free_memory_gb'] = 0
        gpu_info['used_memory_gb'] = 0

    return gpu_info


def print_model_summary(model, config):
    """打印模型摘要（替代torchviz的可视化）"""
    print("\n" + "=" * 60)
    print("模型结构摘要")
    print("=" * 60)

    total_params = 0
    trainable_params = 0

    print("模型层详细信息:")
    print("-" * 80)
    print(f"{'层名称':<30} {'输出形状':<25} {'参数数量':<15} {'可训练'}")
    print("-" * 80)

    for name, module in model.named_modules():
        if not name:  # 跳过根模块
            continue

        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += num_params
        trainable_params += num_trainable

        # 简化输出形状
        output_shape = 'N/A'

        print(f"{name:<30} {str(output_shape):<25} {num_params:<15,} {'是' if num_trainable > 0 else '否'}")

    print("-" * 80)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print("=" * 60)


def plot_model_architecture(model, config, input_shape=None, save_path=None):
    """可视化模型架构 - 修复版，避免复杂的torchviz依赖"""
    try:
        # 保存模型当前模式，并设置为评估模式
        original_mode = model.training
        model.eval()

        # 创建虚拟输入
        if config.feature_type == '2Hz':
            feature_dim = config.frequency_bands
        else:
            feature_dim = config.five_bands

        # 创建虚拟输入（使用batch size=2）
        batch_size = 2
        dummy_eeg = torch.randn(batch_size, config.eeg_channels, feature_dim)

        if config.use_eog and config.use_multimodal:
            dummy_eog = torch.randn(batch_size, config.eog_features)

            # 简单测试前向传播
            with torch.no_grad():
                output = model(dummy_eeg, dummy_eog)
        else:
            with torch.no_grad():
                output = model(dummy_eeg)

        print(f"✅ 模型前向传播测试成功！")
        print(f"   输入形状: EEG={dummy_eeg.shape}, " +
              (f"EOG={dummy_eog.shape}" if config.use_eog and config.use_multimodal else "无EOG"))
        print(f"   输出形状: {output.shape}")

        # 使用简单的文本摘要代替复杂的可视化
        print_model_summary(model, config)

        # 保存模型架构信息到文件
        if save_path:
            model_info = {
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'layers': []
            }

            for name, module in model.named_modules():
                if not name:
                    continue
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'trainable': any(p.requires_grad for p in module.parameters())
                }
                model_info['layers'].append(layer_info)

            # 保存到JSON文件
            json_path = save_path.replace('.png', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            print(f"模型架构信息已保存到: {json_path}")

            # 创建简单的文本图示
            create_text_model_diagram(model, save_path.replace('.png', '_diagram.txt'))

        # 恢复模型原来的模式
        model.train(original_mode)

        return True

    except Exception as e:
        print(f"⚠️  模型可视化失败: {e}")
        print("将使用简单的模型摘要代替...")

        # 即使失败也打印简单摘要
        print_model_summary(model, config)

        return False


def create_text_model_diagram(model, save_path):
    """创建文本模型结构图"""
    diagram_lines = []
    diagram_lines.append("=" * 80)
    diagram_lines.append("深度学习模型结构图")
    diagram_lines.append("=" * 80)

    # 模型基本信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    diagram_lines.append(f"\n📊 模型参数统计:")
    diagram_lines.append(f"   总参数量: {total_params:,}")
    diagram_lines.append(f"   可训练参数: {trainable_params:,}")
    diagram_lines.append(f"   不可训练参数: {total_params - trainable_params:,}")

    # 层级结构
    diagram_lines.append(f"\n🏗️  模型层级结构:")
    diagram_lines.append("-" * 80)

    indent_level = 0
    for name, module in model.named_modules():
        if not name:
            continue

        # 计算缩进
        parts = name.split('.')
        current_indent = len(parts) - 1

        # 模块类型
        module_type = module.__class__.__name__

        # 参数统计
        param_count = sum(p.numel() for p in module.parameters())

        # 构建层级显示
        indent_str = "  " * current_indent

        if current_indent == 0:
            prefix = "├─ "
        elif current_indent == 1:
            prefix = "│  ├─ "
        else:
            prefix = "│  " + "   " * (current_indent - 1) + "└─ "

        line = f"{prefix}{name} ({module_type})"
        if param_count > 0:
            line += f" [参数: {param_count:,}]"

        diagram_lines.append(line)

    diagram_lines.append("-" * 80)

    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(diagram_lines))

    print(f"文本模型结构图已保存到: {save_path}")


def save_experiment_config(config, save_path):
    """保存实验配置"""
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"实验配置已保存到: {save_path}")


def load_experiment_config(config_path):
    """加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    from config import Config
    config = Config.from_dict(config_dict)
    return config


def monitor_resources():
    """监控系统资源"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()

    # 内存使用
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)
    memory_used_gb = memory.used / (1024 ** 3)
    memory_percent = memory.percent

    # GPU信息
    gpu_info = check_gpu()

    print(f"\n系统资源监控:")
    print(f"  CPU: {cpu_percent}% 使用率 ({cpu_count} 核心)")
    print(f"  内存: {memory_used_gb:.1f} GB / {memory_total_gb:.1f} GB ({memory_percent}%)")

    if gpu_info['available']:
        print(f"  GPU: {gpu_info['device_name']}")
        print(f"  GPU内存: {gpu_info['used_memory_gb']:.1f} GB / {gpu_info['total_memory_gb']:.1f} GB")

    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'gpu_info': gpu_info
    }


def plot_attention_weights(attention_weights, channel_names=None, save_path=None):
    """可视化注意力权重"""
    if attention_weights is None:
        return

    # 设置中文字体
    if _chinese_font_available:
        title = "注意力权重矩阵"
        xlabel = "Key"
        ylabel = "Query"
    else:
        title = "Attention Weights Matrix"
        xlabel = "Key"
        ylabel = "Query"

    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))

    if attention_weights.ndim == 2:
        # 2D注意力矩阵
        plt.imshow(attention_weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight' if not _chinese_font_available else '注意力权重')

        if channel_names is not None:
            plt.xticks(range(len(channel_names)), channel_names, rotation=45, ha='right')
            plt.yticks(range(len(channel_names)), channel_names)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, fontweight='bold')

    elif attention_weights.ndim == 3:
        # 3D注意力（多头）
        num_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))

        if num_heads == 1:
            axes = [axes]

        for i in range(num_heads):
            ax = axes[i]
            im = ax.imshow(attention_weights[i], cmap='viridis', aspect='auto')
            ax.set_title(f'Head {i + 1}', fontweight='bold')

            if channel_names is not None:
                ax.set_xticks(range(len(channel_names)))
                ax.set_xticklabels(channel_names, rotation=45, ha='right')
                ax.set_yticks(range(len(channel_names)))
                ax.set_yticklabels(channel_names)

        if _chinese_font_available:
            plt.suptitle('多头注意力权重', fontweight='bold')
        else:
            plt.suptitle('Multi-head Attention Weights', fontweight='bold')
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_grad_flow(named_parameters, save_path=None):
    """可视化梯度流"""
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())

    if not layers:
        print("没有可用的梯度数据")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))

    # 动态设置ylim
    if ave_grads:
        max_val = max(max_grads + [0.02])
        plt.ylim(bottom=-0.001, top=max_val * 1.1)

    if _chinese_font_available:
        plt.xlabel("网络层")
        plt.ylabel("梯度值")
        plt.title("梯度流可视化", fontweight='bold')
        plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                    plt.Line2D([0], [0], color="b", lw=4),
                    plt.Line2D([0], [0], color="k", lw=4)],
                   ['最大梯度', '平均梯度', '零梯度线'])
    else:
        plt.xlabel("Layers")
        plt.ylabel("Gradient Value")
        plt.title("Gradient Flow Visualization", fontweight='bold')
        plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                    plt.Line2D([0], [0], color="b", lw=4),
                    plt.Line2D([0], [0], color="k", lw=4)],
                   ['Max Gradient', 'Mean Gradient', 'Zero Gradient'])

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒" if _chinese_font_available else f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        if _chinese_font_available:
            return f"{int(minutes)}分{int(seconds)}秒"
        else:
            return f"{int(minutes)}m{int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if _chinese_font_available:
            return f"{int(hours)}时{int(minutes)}分{int(seconds)}秒"
        else:
            return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"


def create_experiment_report(config, metrics, training_time, save_path=None):
    """创建实验报告"""
    report = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_time': format_time(training_time),
            'device': config.device
        },
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        'metrics': metrics,
        'summary': {
            'task_type': config.task_type
        }
    }

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"实验报告已保存到: {save_path}")

    return report


def compare_models(model_results, metric='main_metric', save_path=None):
    """比较不同模型的结果"""
    model_names = list(model_results.keys())
    metric_values = [results[metric] for results in model_results.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(model_names)), metric_values)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

    if metric == 'main_metric':
        ylabel = '主要指标' if _chinese_font_available else 'Main Metric'
    else:
        ylabel = metric.upper()

    plt.ylabel(ylabel)

    if _chinese_font_available:
        plt.title('模型性能比较', fontweight='bold')
    else:
        plt.title('Model Performance Comparison', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def save_model_summary(model, config, save_path=None):
    """保存模型摘要到文件"""
    summary_lines = []

    # 基本信息
    summary_lines.append("=" * 60)
    if _chinese_font_available:
        summary_lines.append("模型结构摘要")
    else:
        summary_lines.append("Model Architecture Summary")
    summary_lines.append("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if _chinese_font_available:
        summary_lines.append(f"总参数量: {total_params:,}")
        summary_lines.append(f"可训练参数量: {trainable_params:,}")
        summary_lines.append(f"不可训练参数量: {total_params - trainable_params:,}")

        summary_lines.append("\n模型层详细信息:")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'层名称':<30} {'参数数量':<15} {'可训练'}")
    else:
        summary_lines.append(f"Total Parameters: {total_params:,}")
        summary_lines.append(f"Trainable Parameters: {trainable_params:,}")
        summary_lines.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")

        summary_lines.append("\nLayer Details:")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'Layer Name':<30} {'Parameters':<15} {'Trainable'}")

    summary_lines.append("-" * 80)

    for name, module in model.named_modules():
        if not name:
            continue

        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        if num_params > 0:  # 只显示有参数的层
            summary_lines.append(f"{name:<30} {num_params:<15,} {'Yes' if num_trainable > 0 else 'No'}")

    summary_lines.append("-" * 80)

    # 配置信息
    if _chinese_font_available:
        summary_lines.append("\n配置信息:")
    else:
        summary_lines.append("\nConfiguration:")
    summary_lines.append("-" * 80)
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            summary_lines.append(f"{key}: {value}")

    # 保存到文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print(f"模型摘要已保存到: {save_path}")

    # 打印到控制台
    print('\n'.join(summary_lines))

    return summary_lines


def plot_training_history_simple(train_losses, val_losses, train_metrics, val_metrics,
                                 config, save_path=None):
    """简单的训练历史可视化（确保中英文兼容）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation')
    ax.set_xlabel('Epoch')

    if _chinese_font_available:
        ax.set_ylabel('损失值')
        ax.set_title('训练和验证损失', fontweight='bold')
    else:
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss', fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # 主要指标
    ax = axes[1]
    if config.task_type == 'classification':
        train_main = [m['f1'] for m in train_metrics] if train_metrics else [0] * len(epochs)
        val_main = [m['f1'] for m in val_metrics] if val_metrics else [0] * len(epochs)
        metric_name = 'F1 Score' if not _chinese_font_available else 'F1分数'
    else:
        train_main = [m['rmse'] for m in train_metrics] if train_metrics else [0] * len(epochs)
        val_main = [m['rmse'] for m in val_metrics] if val_metrics else [0] * len(epochs)
        metric_name = 'RMSE'

    ax.plot(epochs, train_main, 'b-', linewidth=2, label='Train')
    ax.plot(epochs, val_main, 'r-', linewidth=2, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)

    if _chinese_font_available:
        ax.set_title(f'训练和验证{metric_name}', fontweight='bold')
    else:
        ax.set_title(f'Training and Validation {metric_name}', fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率（分类任务）
    ax = axes[2]
    if config.task_type == 'classification':
        train_acc = [m['accuracy'] for m in train_metrics] if train_metrics else [0] * len(epochs)
        val_acc = [m['accuracy'] for m in val_metrics] if val_metrics else [0] * len(epochs)

        ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train')
        ax.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')
        ax.set_xlabel('Epoch')

        if _chinese_font_available:
            ax.set_ylabel('准确率')
            ax.set_title('训练和验证准确率', fontweight='bold')
        else:
            ax.set_ylabel('Accuracy')
            ax.set_title('Training and Validation Accuracy', fontweight='bold')
    else:
        # 回归任务显示相关系数
        train_corr = [m['correlation'] for m in train_metrics] if train_metrics else [0] * len(epochs)
        val_corr = [m['correlation'] for m in val_metrics] if val_metrics else [0] * len(epochs)

        ax.plot(epochs, train_corr, 'b-', linewidth=2, label='Train')
        ax.plot(epochs, val_corr, 'r-', linewidth=2, label='Validation')
        ax.set_xlabel('Epoch')

        if _chinese_font_available:
            ax.set_ylabel('相关系数')
            ax.set_title('训练和验证相关系数', fontweight='bold')
        else:
            ax.set_ylabel('Correlation')
            ax.set_title('Training and Validation Correlation', fontweight='bold')

    ax.legend()
    ax.grid(True, alpha=0.3)

    # 学习率
    ax = axes[3]
    # 这里需要传入学习率历史，如果没有就留空
    ax.set_xlabel('Epoch')

    if _chinese_font_available:
        ax.set_ylabel('学习率')
        ax.set_title('学习率变化', fontweight='bold')
    else:
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule', fontweight='bold')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")

    plt.show()


# 测试函数
def test_utils():
    """测试工具函数"""
    print("=" * 60)
    print("测试工具函数")
    print("=" * 60)

    # 测试随机种子设置
    print("1. 测试随机种子设置...")
    set_seed(42)
    print("✅ 通过")

    # 测试GPU检查
    print("2. 测试GPU检查...")
    gpu_info = check_gpu()
    print(f"GPU可用: {gpu_info['available']}")
    print("✅ 通过")

    # 测试时间格式化
    print("3. 测试时间格式化...")
    print(f"3600秒: {format_time(3600)}")
    print(f"3661秒: {format_time(3661)}")
    print("✅ 通过")

    # 测试中文字体
    print("4. 测试中文字体...")
    if _chinese_font_available:
        print("✅ 中文字体可用")
    else:
        print("⚠️  中文字体不可用，使用英文字体")

    print("\n所有测试完成！")


if __name__ == '__main__':
    test_utils()