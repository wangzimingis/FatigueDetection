#main.py
import argparse
import sys
import os
import numpy as np
import torch
from config import Config
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# main.py 开头添加


from src.data_loader import (
    SEEDVIGDataset,
    create_dataloaders,
    create_cross_validation_dataloaders,
    DataAugmentation
)
from src.feature_analyzer import (
    EEGFeatureExtractor,
    FeatureVisualizer,
    analyze_dataset_statistics
)
from src.models import create_model, MultiModalFatigueModel
from src.trainer import FatigueTrainer, cross_validation_training, hyperparameter_tuning
from src.utils import set_seed, check_gpu, plot_model_architecture

# main.py 文件开头添加以下代码
import matplotlib
import matplotlib.font_manager as fm
import os
import platform

# 根据操作系统设置中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统：使用微软雅黑
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
        'C:/Windows/Fonts/msyhbd.ttc',  # 微软雅黑粗体
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
    ]
elif system == 'Darwin':  # macOS
    # macOS系统：使用苹方
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Microsoft/msyh.ttf',
    ]
else:  # Linux
    # Linux系统
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]

# 尝试添加中文字体
font_added = False
for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            # 添加字体到Matplotlib
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已设置中文字体: {font_name}")
            font_added = True
            break
        except Exception as e:
            print(f"⚠️  加载字体失败 {font_path}: {e}")

# 如果没找到系统字体，使用下载的字体
if not font_added:
    try:
        # 下载思源黑体作为备选
        import urllib.request
        import tempfile

        print("正在下载中文字体...")
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
        temp_dir = tempfile.gettempdir()
        local_font_path = os.path.join(temp_dir, "SourceHanSansSC-Regular.otf")

        if not os.path.exists(local_font_path):
            urllib.request.urlretrieve(font_url, local_font_path)

        fm.fontManager.addfont(local_font_path)
        font_name = fm.FontProperties(fname=local_font_path).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"✅ 已使用下载字体: {font_name}")
    except Exception as e:
        print(f"❌ 无法设置中文字体: {e}")
        print("将使用默认英文字体，中文显示为方框")

# 设置Matplotlib默认参数
matplotlib.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


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
    parser.add_argument('--model_type', type=str, default='multimodal_transformer',
                        choices=['cnn', 'multimodal_cnn_lstm', 'multimodal_transformer'],
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

    # 创建配置
    config = Config()

    # 更新配置
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

    # 调试模式
    if args.debug:
        config.epochs = 5
        config.batch_size = 16
        print("调试模式：减少训练轮数和批大小")

    # 设置随机种子
    set_seed(config.seed)

    # 检查GPU
    gpu_info = check_gpu()
    print(f"设备: {config.device}")
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']} (内存: {gpu_info['total_memory_gb']:.1f} GB)")
    else:
        print("警告：未检测到GPU，将使用CPU训练")

    # 创建实验目录
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

    return config


def main():
    """主函数"""
    args = parse_arguments()

    # 设置实验
    config = setup_experiment(args)

    # 特征分析
    if args.analyze_features or args.visualize_features:
        print("\n" + "=" * 70)
        print("特征分析")
        print("=" * 70)

        # 创建数据集
        dataset = SEEDVIGDataset(
            config=config,
            subject_ids=[1],  # 使用第一个被试者进行分析
            mode='train'
        )

        # 特征提取和分析
        if args.analyze_features:
            features, labels = analyze_dataset_statistics(dataset, config)

        # 可视化特征
        if args.visualize_features:
            visualizer = FeatureVisualizer()

            # 随机选择一些样本进行可视化
            sample_indices = np.random.choice(len(dataset), min(50, len(dataset)), replace=False)
            sample_data = []
            sample_labels = []

            for idx in sample_indices:
                sample = dataset[idx]
                sample_data.append(sample['eeg'].numpy())
                sample_labels.append(sample['label'].numpy())

            sample_data = np.array(sample_data)
            sample_labels = np.concatenate(sample_labels)

            # 时域特征可视化
            visualizer.plot_time_domain_features(
                sample_data,
                sample_labels,
                save_path=f"{config.figure_dir}/time_domain_features.png"
            )

            # 频域特征可视化
            visualizer.plot_frequency_domain_features(
                sample_data,
                sample_labels,
                save_path=f"{config.figure_dir}/frequency_domain_features.png"
            )

            # 空域特征可视化
            visualizer.plot_spatial_features(
                sample_data[0],  # 第一个样本
                save_path=f"{config.figure_dir}/spatial_features.png"
            )

            # 疲劳相关分析
            visualizer.plot_fatigue_analysis(
                sample_data,
                sample_labels,
                save_path=f"{config.figure_dir}/fatigue_analysis.png"
            )

            print(f"特征可视化已保存到: {config.figure_dir}")

    # 模型训练
    if args.train:
        print("\n" + "=" * 70)
        print("模型训练")
        print("=" * 70)

        if args.cross_validation:
            # 交叉验证训练
            print("使用交叉验证训练...")
            avg_metrics = cross_validation_training(
                config=config,
                subject_ids=args.subject_ids,
                n_folds=config.n_folds
            )

            print(
                f"\n交叉验证完成！平均主要指标: {avg_metrics['main_metric']['mean']:.4f} ± {avg_metrics['main_metric']['std']:.4f}")

        else:
            # 常规训练
            # 创建数据加载器
            train_loader, val_loader = create_dataloaders(
                config=config,
                subject_ids=args.subject_ids,
                batch_size=config.batch_size
            )

            print(f"训练集: {len(train_loader.dataset)} 个样本")
            print(f"验证集: {len(val_loader.dataset)} 个样本")

            # 创建模型
            model = create_model(config)

            # 可视化模型结构
            plot_model_architecture(model, config, save_path=f"{config.figure_dir}/model_architecture.png")

            # 训练模型
            trainer = FatigueTrainer(model, config)
            best_metric = trainer.train(train_loader, val_loader)

            print(f"\n训练完成！最佳指标: {best_metric:.4f}")

            # 评估模型
            val_metrics = trainer.evaluate(val_loader)

            # 保存最终模型
            trainer.save_checkpoint('final_model.pth')

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

    # 模型评估
    if args.evaluate and args.model_path:
        print("\n" + "=" * 70)
        print("模型评估")
        print("=" * 70)

        # 加载模型
        checkpoint = torch.load(args.model_path, map_location=config.device)

        # 重建配置
        eval_config = Config.from_dict(checkpoint['config'])
        eval_config.device = config.device

        # 创建模型
        model = create_model(eval_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(eval_config.device)

        # 创建训练器
        trainer = FatigueTrainer(model, eval_config)
        trainer.history = checkpoint.get('history', trainer.history)
        trainer.best_metric = checkpoint.get('best_metric', trainer.best_metric)

        # 创建测试数据加载器
        test_dataset = SEEDVIGDataset(
            config=eval_config,
            subject_ids=args.subject_ids,
            mode='val'
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=eval_config.batch_size,
            shuffle=False,
            num_workers=4
        )

        # 评估模型
        test_metrics = trainer.evaluate(test_loader)

        print(f"\n测试集性能:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("程序执行完成！")
    print("=" * 70)


if __name__ == '__main__':
    # 将这里原来的 `import datetime` 修改为以下代码
    import datetime
    main()