# main.py
import argparse
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import datetime

from config import Config

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    SEEDVIGDataset,
    DataAugmentation
)
from src.feature_analyzer import (
    EEGFeatureExtractor,
    FeatureVisualizer,
    analyze_dataset_statistics
)
from src.models import create_model
from src.trainer import FatigueTrainer, cross_validation_training, hyperparameter_tuning
from src.utils import set_seed, check_gpu, plot_model_architecture, setup_chinese_font

# 初始化中文字体
setup_chinese_font()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='基于深度学习的驾驶疲劳评估系统（SEED-VIG数据集）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./data/SEED-VIG/',
                        help='数据集根目录')
    parser.add_argument('--subject_ids', type=int, nargs='+', default=None,
                        help='被试者ID列表，None表示使用所有被试者')
    parser.add_argument('--feature_type', type=str, default='2Hz',
                        choices=['2Hz', '5Bands'],
                        help='特征类型')
    parser.add_argument('--use_eog', action='store_true', default=True,
                        help='使用EOG特征')
    parser.add_argument('--use_multimodal', action='store_true', default=True,
                        help='使用多模态融合')
    parser.add_argument('--no-use_eog', dest='use_eog', action='store_false', help='禁用EOG特征')
    parser.add_argument('--no-use_multimodal', dest='use_multimodal', action='store_false', help='禁用多模态融合')

    # 任务类型
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='任务类型：分类或回归')

    # 特征分析
    parser.add_argument('--analyze_features', action='store_true',
                        help='进行特征分析')
    parser.add_argument('--visualize_features', action='store_true',
                        help='可视化特征')

    # 模型参数
    # 模型参数
    parser.add_argument('--model_type', type=str, default='multimodal_transformer',
                        choices=['cnn', 'multimodal_cnn_lstm', 'multimodal_transformer',
                                 'hyperlstm', 'macnn', 'mlp', 'lightcnn'],
                        help='模型类型')
    parser.add_argument('--fusion_method', type=str, default='cross_attention',
                        choices=['concatenate', 'attention', 'cross_attention'],
                        help='多模态融合方法')

    # 训练参数
    parser.add_argument('--train', action='store_true',
                        help='训练模型')
    parser.add_argument('--cross_validation', action='store_true',
                        help='使用交叉验证')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                        help='超参数调优')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')

    # 新增：关键正则化参数
    parser.add_argument('--dropout_rate', type=float, default=None,
                        help='Dropout 比率 (None 时使用模型默认值)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='权重衰减 (None 时使用模型默认值)')
    parser.add_argument('--patience', type=int, default=None,
                        help='早停耐心 (None 时使用模型默认值)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='学习率调度器类型')

    # 评估参数
    parser.add_argument('--evaluate', action='store_true',
                        help='评估模型')
    parser.add_argument('--model_path', type=str,
                        help='模型路径')

    # 实验设置
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='实验名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 其他参数
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--visualize_training', action='store_true', default=True,
                        help='可视化训练过程')

    return parser.parse_args()


def setup_experiment(args):
    """设置实验"""
    print("=" * 70)
    print("驾驶疲劳评估系统 - 实验设置")
    print("=" * 70)

    config = Config()

    config.data_root = args.data_root
    config.feature_type = args.feature_type
    config.use_eog = args.use_eog
    config.use_multimodal = args.use_multimodal
    config.classification_type = args.task
    config.model_type = args.model_type
    config.fusion_method = args.fusion_method
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.seed = args.seed

    # 覆盖可能由命令行指定的正则化参数
    if args.dropout_rate is not None:
        config.dropout_rate = args.dropout_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.patience is not None:
        config.patience = args.patience

    # 若使用 MACNN，自动设置推荐的超参数（除非用户明确指定）
    if config.model_type == 'macnn':
        # 推荐配置
        if args.dropout_rate is None:
            config.dropout_rate = 0.5
        if args.weight_decay is None:
            config.weight_decay = 1e-3
        if args.patience is None:
            config.patience = 30
        if args.epochs == 150:   # 默认值150，可以适当延长
            config.epochs = 200
        if args.learning_rate == 1e-3:
            config.learning_rate = 5e-4
        # 如果未指定调度器，则使用余弦退火
        if args.scheduler == 'plateau':
            config.scheduler_type = 'cosine'

    # 保存调度器类型到 config 对象（需在 Config 类中添加属性，如不存在则动态添加）
    if not hasattr(config, 'scheduler_type'):
        setattr(config, 'scheduler_type', args.scheduler)

    if args.debug:
        config.epochs = 5
        config.batch_size = 16
        print("调试模式：减少训练轮数和批大小")

    set_seed(config.seed)

    gpu_info = check_gpu()
    print(f"设备: {config.device}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']} (内存: {gpu_info['total_memory_gb']:.1f} GB)")
    else:
        print("警告：未检测到GPU，将使用CPU训练")

    experiment_dir = f"./experiments/{args.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.save_dir = os.path.join(experiment_dir, 'checkpoints')
    config.log_dir = os.path.join(experiment_dir, 'logs')
    config.result_dir = os.path.join(experiment_dir, 'results')
    config.figure_dir = os.path.join(experiment_dir, 'figures')

    for dir_path in [config.save_dir, config.log_dir, config.result_dir, config.figure_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"实验目录: {experiment_dir}")
    print(f"任务类型: {'分类' if config.classification_type == 'classification' else '回归'}")
    print(f"模型类型: {config.model_type}")
    print(f"特征类型: {config.feature_type}")
    print(f"使用EOG: {config.use_eog}")
    print(f"多模态融合: {config.use_multimodal}")
    print(f"Dropout: {config.dropout_rate}  Weight Decay: {config.weight_decay}  Patience: {config.patience}")
    print(f"调度器: {config.scheduler_type}")

    return config


def main():
    args = parse_arguments()
    config = setup_experiment(args)

    # 特征分析
    if args.analyze_features or args.visualize_features:
        print("\n" + "=" * 70)
        print("特征分析")
        print("=" * 70)
        dataset = SEEDVIGDataset(config, subject_ids=[1], mode='train')
        if args.analyze_features:
            features, labels = analyze_dataset_statistics(dataset, config)
        if args.visualize_features:
            visualizer = FeatureVisualizer()
            sample_indices = np.random.choice(len(dataset), min(50, len(dataset)), replace=False)
            sample_data = []
            sample_labels = []
            for idx in sample_indices:
                sample = dataset[idx]
                sample_data.append(sample['eeg'].numpy())
                sample_labels.append(sample['label'].numpy())
            sample_data = np.array(sample_data)
            sample_labels = np.concatenate(sample_labels)
            visualizer.plot_time_domain_features(sample_data, sample_labels,
                                                 save_path=f"{config.figure_dir}/time_domain_features.png")
            visualizer.plot_frequency_domain_features(sample_data, sample_labels,
                                                      save_path=f"{config.figure_dir}/frequency_domain_features.png")
            visualizer.plot_spatial_features(sample_data[0],
                                             save_path=f"{config.figure_dir}/spatial_features.png")
            visualizer.plot_fatigue_analysis(sample_data, sample_labels,
                                             save_path=f"{config.figure_dir}/fatigue_analysis.png")
            print(f"特征可视化已保存到: {config.figure_dir}")

    # 模型训练
    if args.train:
        print("\n" + "=" * 70)
        print("模型训练")
        print("=" * 70)

        if args.cross_validation:
            print("使用交叉验证训练...")
            avg_metrics = cross_validation_training(config, subject_ids=args.subject_ids, n_folds=config.n_folds)
            print(f"\n交叉验证完成！平均主要指标: {avg_metrics['main_metric']['mean']:.4f} ± {avg_metrics['main_metric']['std']:.4f}")
        else:
            # ========== 固定被试划分方案 ==========
            train_ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20]  # 17个实验，训练
            val_ids = [5, 16, 19]   # 3个实验，验证
            test_ids = [21, 22, 23]  # 3个实验，测试

            # 关键：禁用归一化，否则验证集/测试集会独立计算 min-max，破坏分布一致性
            config.normalization = False

            # 数据增强（仅训练集）
            train_transform = DataAugmentation(noise_std=0.01, scale_range=(0.9, 1.1), drop_prob=0.1)

            # 创建训练集（自己将会拟合 scaler）
            train_dataset = SEEDVIGDataset(config, subject_ids=train_ids, mode='train',
                                           transform=train_transform)

            # 显式取出训练集拟合好的 scaler
            train_eeg_scaler = train_dataset.eeg_scaler
            train_eog_scaler = train_dataset.eog_scaler
            if train_eeg_scaler is None:
                raise RuntimeError("训练集 EEG scaler 为 None，标准化未生效！")
            print(f"训练集 EEG scaler 已捕获，均值前3维示例: {train_eeg_scaler.mean_[:3]}")

            # 创建验证集和测试集，传入训练集的 scaler
            val_dataset = SEEDVIGDataset(config, subject_ids=val_ids, mode='val', transform=None,
                                         scaler_eeg=train_eeg_scaler,
                                         scaler_eog=train_eog_scaler)
            test_dataset = SEEDVIGDataset(config, subject_ids=test_ids, mode='val', transform=None,
                                          scaler_eeg=train_eeg_scaler,
                                          scaler_eog=train_eog_scaler)

            # ---------- 类别平衡采样 ----------
            sampler = None
            if config.task_type == 'classification':
                train_labels = train_dataset.data['labels'].astype(int)
                unique_labels = np.unique(train_labels)
                if len(unique_labels) == config.num_classes:
                    class_counts = np.bincount(train_labels, minlength=config.num_classes)
                    if class_counts.min() > 0:
                        class_weights = 1.0 / class_counts
                        sample_weights = class_weights[train_labels]
                        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
                        print(f"  类别平衡采样: 类别分布={class_counts.tolist()}")
                    else:
                        print(f"  警告: 某些类别样本数为0")
                else:
                    print(f"  警告: 训练集类别数({len(unique_labels)})不等于配置类别数({config.num_classes})")

            # 创建 DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                shuffle=sampler is None,
                num_workers=0,
                pin_memory=config.device == 'cuda',
                drop_last=True if len(train_dataset) > config.batch_size else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=config.device == 'cuda'
            )

            print(f"训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
            print(f"验证集: {len(val_dataset)} 个样本, {len(val_loader)} 个批次")
            print(f"测试集 (实验IDs: {test_ids}) 共 {len(test_dataset)} 个样本，将在训练完成后评估")

            # 创建模型
            model = create_model(config)
            plot_model_architecture(model, config, save_path=f"{config.figure_dir}/model_architecture.png")

            # 训练（传递调度器类型）
            trainer = FatigueTrainer(model, config)
            best_metric = trainer.train(train_loader, val_loader)
            print(f"\n训练完成！最佳验证指标: {best_metric:.4f}")

            # 保存最终模型及 scaler
            trainer.save_checkpoint('final_model.pth')
            torch.save({'eeg_scaler': train_eeg_scaler, 'eog_scaler': train_eog_scaler},
                       os.path.join(config.save_dir, 'scalers.pth'))

            # ========== 在独立测试集上最终评估 ==========
            print("\n" + "=" * 70)
            print("在独立测试集上最终评估")
            print("=" * 70)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     num_workers=0, pin_memory=config.device == 'cuda')
            test_metrics = trainer.evaluate(test_loader, save_figures=True)
            print(f"\n测试集最终性能摘要:")
            for key, value in test_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

    # 超参数调优
    if args.hyperparameter_tuning:
        print("\n" + "=" * 70)
        print("超参数调优")
        print("=" * 70)
        param_grid = {
            'learning_rate': [1e-3, 1e-4, 5e-4],
            'batch_size': [16, 32, 64],
            'dropout_rate': [0.3, 0.5, 0.7],
            'cnn_channels': [[32, 64, 128], [64, 128, 256]],
        }
        best_params, best_score = hyperparameter_tuning(config, param_grid)
        print(f"最佳超参数: {best_params}")
        print(f"最佳得分: {best_score}")

    # 模型评估（单独评估模式）
    if args.evaluate and args.model_path:
        print("\n" + "=" * 70)
        print("模型评估")
        print("=" * 70)
        checkpoint = torch.load(args.model_path, map_location=config.device, weights_only=False)
        eval_config = checkpoint['config']
        eval_config.device = config.device
        model = create_model(eval_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(eval_config.device)
        trainer = FatigueTrainer(model, eval_config)
        trainer.history = checkpoint.get('history', trainer.history)
        trainer.best_metric = checkpoint.get('best_metric', trainer.best_metric)

        test_ids = args.subject_ids if args.subject_ids else [21, 22, 23]
        test_dataset = SEEDVIGDataset(eval_config, subject_ids=test_ids, mode='val', transform=None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_config.batch_size,
                                                  shuffle=False, num_workers=0)
        test_metrics = trainer.evaluate(test_loader)
        print(f"\n测试集性能:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("程序执行完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()