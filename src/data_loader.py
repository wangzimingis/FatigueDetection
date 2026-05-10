# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import scipy.io as sio
import h5py
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


class SEEDVIGDataset(Dataset):
    """SEED-VIG数据集类 - 基于文件索引的精确加载版本"""

    def __init__(self, config, subject_ids=None, mode='train', transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform

        # 1. 构建实验文件索引（按文件名排序后，每个实验包含EEG/EOG/标签路径）
        self.experiment_index = self._build_experiment_index()

        # 2. 确定要加载的实验序号
        if subject_ids is None:
            # 默认加载所有实验
            subject_ids = list(range(1, len(self.experiment_index) + 1))
        else:
            # 过滤掉超出范围的序号
            max_id = len(self.experiment_index)
            valid_ids = [sid for sid in subject_ids if 1 <= sid <= max_id]
            if len(valid_ids) != len(subject_ids):
                print(f"警告: 部分subject_id超出范围[1, {max_id}]，已忽略无效ID")
            subject_ids = valid_ids

        # 3. 加载指定实验的数据
        self.data = self._load_experiments_by_ids(subject_ids)

        # 4. 预处理
        self._preprocess()

        print(f"数据集初始化完成: {mode}模式，{len(self)}个样本")

    def _build_experiment_index(self):
        """扫描数据集目录，返回按数字编号排序的实验索引列表"""
        base_path = Path(self.config.data_root)
        feature_type = self.config.feature_type
        use_eog = self.config.use_eog

        eeg_dir = base_path / f"EEG_Feature_{feature_type}"
        eog_dir = base_path / "EOG_Feature" if use_eog else None
        label_dir = base_path / "perclos_labels"

        # 收集所有非Forehead的EEG .mat文件
        eeg_files = []
        for f in eeg_dir.glob("*.mat"):
            if f.parent.name.startswith("Forehead"):
                continue
            eeg_files.append(f)

        # ---- 自然排序：按文件名开头的数字，数字相同时按完整文件名排序 ----
        def sort_key(filepath):
            stem = filepath.stem
            # 提取开头的数字（例如 "4_20151107_noon" -> 4, "10_20151125_noon" -> 10）
            num_str = stem.split('_')[0]
            try:
                number = int(num_str)
            except ValueError:
                number = 0  # 容错
            return (number, stem)

        eeg_files.sort(key=sort_key)

        experiments = []
        for eeg_file in eeg_files:
            stem = eeg_file.stem
            eog_file = eog_dir / f"{stem}.mat" if eog_dir else None
            label_file = label_dir / f"{stem}.mat"

            if use_eog and (eog_file is None or not eog_file.exists()):
                print(f"  跳过实验 {stem}: 缺少EOG文件")
                continue
            if not label_file.exists():
                print(f"  跳过实验 {stem}: 缺少标签文件")
                continue

            experiments.append({
                'eeg': eeg_file,
                'eog': eog_file if use_eog else None,
                'label': label_file
            })

        print(f"构建实验索引完成，共发现 {len(experiments)} 个有效实验")
        return experiments

    def _load_experiments_by_ids(self, exp_ids):
        """通过实验索引加载数据"""
        all_eeg = []
        all_eog = []
        all_labels = []

        for exp_id in exp_ids:
            print(f"  加载实验 {exp_id}...")
            exp_info = self.experiment_index[exp_id - 1]  # 索引从0开始
            data = self._load_single_experiment(exp_info)
            if data is not None:
                eeg, eog, labels = data
                all_eeg.append(eeg)
                if eog is not None:
                    all_eog.append(eog)
                all_labels.append(labels)
            else:
                print(f"  警告: 实验 {exp_id} 数据加载失败")

        if not all_eeg:
            raise ValueError("未加载到任何有效数据")

        combined_eeg = np.concatenate(all_eeg, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        combined_eog = None
        if all_eog:
            combined_eog = np.concatenate(all_eog, axis=0)

        print(f"  EEG形状: {combined_eeg.shape}")
        print(f"  标签形状: {combined_labels.shape}")
        if combined_eog is not None:
            print(f"  EOG形状: {combined_eog.shape}")

        return {
            'eeg': combined_eeg,
            'eog': combined_eog,
            'labels': combined_labels
        }

    def _load_single_experiment(self, file_dict):
        """加载单个实验的文件，file_dict包含'eeg','eog','label'的Path对象"""
        eeg_file = file_dict['eeg']
        eog_file = file_dict.get('eog')
        label_file = file_dict['label']

        # 加载EEG
        eeg_data = self._load_mat_file(eeg_file)
        if eeg_data is None:
            return None
        print(f"    找到EEG文件: {eeg_file.name}")

        # 维度转换 (channels, samples, features) -> (samples, channels, features)
        if eeg_data.ndim == 3:
            if eeg_data.shape[0] in [17, 4]:  # 通道优先
                eeg_data = np.transpose(eeg_data, (1, 0, 2))
            elif eeg_data.shape[2] in [17, 4]:  # 通道在后
                eeg_data = np.transpose(eeg_data, (0, 2, 1))

        # 加载EOG
        eog_data = None
        if eog_file is not None:
            eog_data = self._load_mat_file(eog_file)
            if isinstance(eog_data, dict):
                # 尝试常见的键
                for key in ['features_table_ica', 'features_table_minus', 'eog_features']:
                    if key in eog_data:
                        eog_data = eog_data[key]
                        break
            if eog_data is not None:
                if eog_data.ndim == 1:
                    eog_data = eog_data.reshape(-1, 1)
                elif eog_data.ndim == 2 and eog_data.shape[0] != eeg_data.shape[0]:
                    eog_data = eog_data.T
            print(f"    找到EOG文件: {eog_file.name}")

        # 加载标签
        label_data = self._load_mat_file(label_file)
        if isinstance(label_data, dict):
            for key in ['perclos_labels', 'perclos', 'label', 'y']:
                if key in label_data:
                    labels = label_data[key].flatten()
                    break
            else:
                # 取第一个非系统键
                for key in label_data:
                    if not key.startswith('__'):
                        labels = label_data[key].flatten()
                        break
        else:
            labels = label_data.flatten()
        print(f"    找到标签文件: {label_file.name}")

        # 数据对齐（取最小长度）
        min_len = min(
            eeg_data.shape[0],
            len(labels),
            eog_data.shape[0] if eog_data is not None else float('inf')
        )
        eeg_data = eeg_data[:min_len]
        labels = labels[:min_len]
        if eog_data is not None:
            eog_data = eog_data[:min_len]

        print(f"    数据对齐: EEG={eeg_data.shape}, EOG={eog_data.shape if eog_data is not None else 'None'}, Labels={len(labels)}")
        return eeg_data, eog_data, labels

    def _load_mat_file(self, filepath):
        """加载.mat文件"""
        try:
            data = sio.loadmat(str(filepath))
            for key in data:
                if not key.startswith('__'):
                    return data[key]
        except Exception as e:
            try:
                with h5py.File(str(filepath), 'r') as f:
                    keys = list(f.keys())
                    if keys:
                        return np.array(f[keys[0]])
            except Exception as e2:
                print(f"    加载文件失败: {filepath.name}, 错误: {e2}")
                return None
        return None

    # 对数据进行预处理，标准化、归一化、标签处理
    def _preprocess(self):
        """数据预处理"""
        # 标准化
        if self.config.standardization:
            print("  标准化数据...")
            self.eeg_scaler = StandardScaler()
            original_shape = self.data['eeg'].shape
            eeg_2d = self.data['eeg'].reshape(-1, original_shape[-1])
            self.data['eeg'] = self.eeg_scaler.fit_transform(eeg_2d).reshape(original_shape)

            if self.data['eog'] is not None:
                self.eog_scaler = StandardScaler()
                self.data['eog'] = self.eog_scaler.fit_transform(self.data['eog'])

        # 归一化
        if self.config.normalization:
            print("  归一化数据...")
            eeg_min = self.data['eeg'].min(axis=(0, 1), keepdims=True)
            eeg_max = self.data['eeg'].max(axis=(0, 1), keepdims=True)
            if (eeg_max - eeg_min).max() > 1e-8:
                self.data['eeg'] = (self.data['eeg'] - eeg_min) / (eeg_max - eeg_min + 1e-8)

            if self.data['eog'] is not None:
                eog_min = self.data['eog'].min(axis=0, keepdims=True)
                eog_max = self.data['eog'].max(axis=0, keepdims=True)
                if (eog_max - eog_min).max() > 1e-8:
                    self.data['eog'] = (self.data['eog'] - eog_min) / (eog_max - eog_min + 1e-8)

        # 标签处理
        if self.config.task_type == 'classification':
            print("  将回归标签转换为分类标签...")
            self.data['labels'] = self._regression_to_classification(self.data['labels'])

        # 打印标签分布
        unique_labels, counts = np.unique(self.data['labels'], return_counts=True)
        print(f"  标签分布: {dict(zip(unique_labels, counts))}")

    def _regression_to_classification(self, regression_labels):
        """将回归标签转换为分类标签"""
        thresholds = self.config.regression_thresholds
        class_labels = np.zeros_like(regression_labels, dtype=np.int64)

        # 根据PERCLOS定义：值越大越疲劳
        class_labels[regression_labels < thresholds['alert']] = 0  # 清醒
        class_labels[(regression_labels >= thresholds['alert']) &
                     (regression_labels < thresholds['mild'])] = 1  # 轻度疲劳
        class_labels[regression_labels >= thresholds['mild']] = 2  # 重度疲劳

        return class_labels

    def __len__(self):
        return self.data['eeg'].shape[0]

    def __getitem__(self, idx):
        eeg = self.data['eeg'][idx].astype(np.float32)
        label = self.data['labels'][idx]

        sample = {
            'eeg': torch.FloatTensor(eeg),
            'label': torch.tensor(label,
                                  dtype=torch.long if self.config.task_type == 'classification' else torch.float32)
        }

        if self.data['eog'] is not None and self.config.use_eog:
            eog = self.data['eog'][idx].astype(np.float32)
            sample['eog'] = torch.FloatTensor(eog)

        if self.mode == 'train' and self.transform is not None:
            sample = self.transform(sample)

        return sample


class DataAugmentation:
    """数据增强类"""

    def __init__(self, noise_std=0.01, scale_range=(0.9, 1.1), drop_prob=0.1):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.drop_prob = drop_prob

    def __call__(self, sample):
        augmented = {}
        if 'eeg' in sample:
            eeg_np = sample['eeg'].numpy()

            # 高斯噪声
            if np.random.random() > 0.5:
                noise = np.random.normal(0, self.noise_std, eeg_np.shape).astype(np.float32)
                eeg_np = eeg_np + noise

            # 随机通道丢弃
            if np.random.random() > 0.5 and eeg_np.shape[0] > 1:
                drop_mask = np.random.choice([0, 1], size=eeg_np.shape[0],
                                             p=[self.drop_prob, 1 - self.drop_prob])
                eeg_np = eeg_np * drop_mask[:, np.newaxis]

            # 随机缩放
            if np.random.random() > 0.5:
                scale = np.random.uniform(*self.scale_range)
                eeg_np = eeg_np * scale

            augmented['eeg'] = torch.FloatTensor(eeg_np)

        for key in sample:
            if key not in augmented:
                augmented[key] = sample[key]
        return augmented


def create_dataloaders(config, subject_ids=None, batch_size=None):
    """创建数据加载器（划分训练/验证集）"""
    if batch_size is None:
        batch_size = config.batch_size

    print("创建数据加载器...")

    dataset = SEEDVIGDataset(
        config=config,
        subject_ids=subject_ids,
        mode='train',
        transform=None
    )

    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    indices = list(range(total_samples))
    np.random.seed(config.seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"  数据集划分: 训练集={train_size}, 验证集={val_size}")

    train_transform = DataAugmentation(
        noise_std=0.01,
        scale_range=(0.9, 1.1),
        drop_prob=0.1
    )

    train_dataset = SubsetWithTransform(dataset, train_indices, transform=train_transform)
    val_dataset = Subset(dataset, val_indices)

    sampler = None
    if config.task_type == 'classification':
        train_labels = [dataset.data['labels'][i] for i in train_indices]
        train_labels = [int(label) for label in train_labels]

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        pin_memory=config.device == 'cuda',
        drop_last=True if len(train_dataset) > batch_size else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == 'cuda'
    )

    print(f"  训练集: {len(train_dataset)} 个样本，{len(train_loader)} 个批次")
    print(f"  验证集: {len(val_dataset)} 个样本，{len(val_loader)} 个批次")

    return train_loader, val_loader


class SubsetWithTransform(Subset):
    """带数据增强的子集类"""

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def create_cross_validation_dataloaders(config, subject_id, fold_idx):
    """创建交叉验证数据加载器（单个实验内按样本5折）"""
    dataset = SEEDVIGDataset(
        config=config,
        subject_ids=[subject_id],
        mode='train',
        transform=None
    )

    n_samples = len(dataset)
    fold_size = n_samples // config.n_folds

    val_start = fold_idx * fold_size
    val_end = (fold_idx + 1) * fold_size if fold_idx < config.n_folds - 1 else n_samples

    val_indices = list(range(val_start, val_end))
    train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))

    train_transform = DataAugmentation(
        noise_std=0.01,
        scale_range=(0.9, 1.1),
        drop_prob=0.1
    )

    train_dataset = SubsetWithTransform(dataset, train_indices, transform=train_transform)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.device == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.device == 'cuda'
    )

    return train_loader, val_loader


def test_data_loader():
    """测试数据加载器"""
    from config import Config

    print("=" * 60)
    print("测试数据加载器")
    print("=" * 60)

    config = Config()
    config.data_root = './data/SEED-VIG/'
    config.feature_type = '2Hz'
    config.use_eog = True
    config.task_type = 'classification'
    # 注意：现在的 subject_id 代表实验序号，而不是原始的“被试者号”
    config.num_subjects = 5  # 测试用

    try:
        print("1. 测试数据集创建（实验1）...")
        dataset = SEEDVIGDataset(config, subject_ids=[1])
        sample = dataset[0]
        print(f"  样本形状 - EEG: {sample['eeg'].shape}, Label: {sample['label']}")

        print("2. 测试数据加载器...")
        train_loader, val_loader = create_dataloaders(config, subject_ids=[1,2], batch_size=4)
        batch = next(iter(train_loader))
        print(f"  批次数据 - EEG: {batch['eeg'].shape}, Label: {batch['label'].shape}")

        print("3. 测试数据增强...")
        transform = DataAugmentation()
        augmented_sample = transform(sample)
        print(f"  增强后形状 - EEG: {augmented_sample['eeg'].shape}")

        print("✅ 所有测试通过!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_data_loader()