# config.py - 最终修复版本
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import os


@dataclass
class Config:
    """训练和模型配置"""

    # 数据集配置
    dataset_name: str = 'SEED-VIG'
    sampling_rate: int = 200
    num_subjects: int = 22  # 根据实际数据修正

    # 数据路径配置
    data_root: str = './data/SEED-VIG/'

    # 特征配置
    eeg_channels: int = 17
    eeg_forehead_channels: int = 4
    frequency_bands: int = 25  # 2Hz特征
    five_bands: int = 5
    eog_features: int = 36

    # 数据使用配置
    use_eeg: bool = True
    use_eog: bool = True
    use_multimodal: bool = True
    feature_type: str = '2Hz'  # '2Hz' or '5Bands'

    # 预处理配置
    normalization: bool = True
    standardization: bool = True
    apply_baseline_correction: bool = False

    # 分类配置
    task_type: str = 'classification'  # 'regression' or 'classification'
    num_classes: int = 3

    # 阈值配置（用于将回归转为分类）
    regression_thresholds: Dict[str, float] = field(
        default_factory=lambda: {'alert': 0.3, 'mild': 0.6, 'severe': 1.0}
    )

    # 深度学习模型配置
    model_type: str = 'lstm'  # 'cnn', 'lstm', 'transformer', 'simple_cnn', 'multimodal_cnn_lstm'

    # CNN配置
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    dropout_rate: float = 0.3

    # LSTM配置
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_bidirectional: bool = True

    # Transformer配置
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_ff_dim: int = 512

    # HyperLSTM 配置
    hyper_hidden_size: int = 64  # 超网络隐藏层维度
    hyper_embedding_size: int = 16  # 超网络嵌入维度
    use_layer_norm: bool = True  # 是否在 HyperLSTM 中使用层归一化

    # 多模态融合
    fusion_method: str = 'concatenate'  # 'concatenate', 'attention', 'cross_attention'

    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    early_stopping: bool = True
    gradient_clip: float = 1.0

    # 交叉验证
    n_folds: int = 5
    seed: int = 42

    # 评估指标
    metrics: List[str] = field(
        default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa', 'mcc']
    )

    # 可视化配置
    plot_training_history: bool = True
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = False
    plot_attention_maps: bool = False
    plot_t_sne: bool = False

    # 路径配置
    save_dir: str = './checkpoints/'
    log_dir: str = './logs/'
    result_dir: str = './results/'
    figure_dir: str = './figures/'

    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练过程配置
    use_mixed_precision: bool = True
    checkpoint_freq: int = 10
    plot_freq: int = 10

    def __post_init__(self):
        """初始化后处理"""
        # 根据特征类型设置特征维度
        if self.feature_type == '2Hz':
            self.feature_dim = self.frequency_bands
        else:
            self.feature_dim = self.five_bands

        # 根据任务类型设置输出维度
        if self.task_type == 'regression':
            self.num_classes = 1
            self.regression_thresholds = {'alert': 0.3, 'mild': 0.6, 'severe': 1.0}#清醒、轻度疲劳、重度疲劳
        else:
            self.num_classes = 3

        if hasattr(self, 'task_type'):
            self.classification_type = self.task_type
        elif hasattr(self, 'classification_type'):
            self.task_type = self.classification_type

        print(f"配置初始化完成: 特征维度={self.feature_dim}, 任务类型={self.task_type}, 设备={self.device}")

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self):
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @property
    def classification_type(self):
        """向后兼容属性"""
        return self.task_type

    @classification_type.setter
    def classification_type(self, value):
        self.task_type = value


class TrainingConfig:
    """训练特定配置"""

    def __init__(self):
        self.checkpoint_freq = 10
        self.log_freq = 10
        self.plot_freq = 10
        self.save_best = True
        self.use_mixed_precision = True
        self.accumulation_steps = 1