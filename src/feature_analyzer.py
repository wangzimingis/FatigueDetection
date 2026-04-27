#feature_analyzer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
import pywt
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import umap
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EEGFeatureExtractor:
    """EEG特征提取器 - 实现时域、频域、空域特征提取"""

    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def extract_time_domain_features(self, eeg_data):
        """提取时域特征"""
        features = {}

        # 基本统计特征
        features['mean'] = np.mean(eeg_data, axis=-1)
        features['std'] = np.std(eeg_data, axis=-1)
        features['variance'] = np.var(eeg_data, axis=-1)
        features['skewness'] = skew(eeg_data, axis=-1)
        features['kurtosis'] = kurtosis(eeg_data, axis=-1)

        # Hjorth参数
        features['activity'] = np.var(eeg_data, axis=-1)
        features['mobility'] = np.sqrt(np.var(np.diff(eeg_data, axis=-1), axis=-1) /
                                       (np.var(eeg_data, axis=-1) + 1e-10))
        features['complexity'] = np.sqrt(np.var(np.diff(np.diff(eeg_data, axis=-1), axis=-1), axis=-1) /
                                         (np.var(np.diff(eeg_data, axis=-1), axis=-1) + 1e-10))

        # 峰峰值
        features['peak_to_peak'] = np.ptp(eeg_data, axis=-1)

        # 过零率
        zero_crossings = np.diff(np.sign(eeg_data), axis=-1)
        features['zero_crossing_rate'] = np.sum(zero_crossings != 0, axis=-1) / eeg_data.shape[-1]

        return features

    def extract_frequency_domain_features(self, eeg_data):
        """提取频域特征"""
        features = {}

        # 计算功率谱密度
        freqs, psd = signal.welch(eeg_data, fs=self.sampling_rate,
                                  nperseg=min(256, eeg_data.shape[-1]), axis=-1)

        # 各频带功率
        band_powers = {}
        relative_powers = {}
        band_ratios = {}

        total_power = np.sum(psd, axis=-1)

        for band_name, (low, high) in self.frequency_bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(psd[..., idx_band], axis=-1)
            band_powers[band_name] = band_power
            relative_powers[f'rel_{band_name}'] = band_power / (total_power + 1e-10)

        # 频带比率（疲劳相关）
        if 'alpha' in band_powers and 'theta' in band_powers:
            band_ratios['alpha_theta'] = band_powers['alpha'] / (band_powers['theta'] + 1e-10)
        if 'beta' in band_powers and 'alpha' in band_powers:
            band_ratios['beta_alpha'] = band_powers['beta'] / (band_powers['alpha'] + 1e-10)
        if 'theta' in band_powers and 'beta' in band_powers:
            band_ratios['theta_beta'] = band_powers['theta'] / (band_powers['beta'] + 1e-10)

        # 谱熵
        spectral_entropy = entropy(psd, axis=-1)

        # 谱质心
        spectral_centroid = np.sum(freqs * psd, axis=-1) / (np.sum(psd, axis=-1) + 1e-10)

        features.update({
            'band_powers': band_powers,
            'relative_powers': relative_powers,
            'band_ratios': band_ratios,
            'spectral_entropy': spectral_entropy,
            'spectral_centroid': spectral_centroid,
            'total_power': total_power
        })

        return features

    def extract_time_frequency_features(self, eeg_data):
        """提取时频特征（小波变换）"""
        features = {}

        # 小波变换
        wavelet = 'db4'
        max_level = pywt.dwt_max_level(eeg_data.shape[-1], pywt.Wavelet(wavelet).dec_len)
        level = min(5, max_level)

        # 对每个通道进行小波变换
        wavelet_coeffs = []
        for i in range(eeg_data.shape[0] if eeg_data.ndim == 2 else 1):
            if eeg_data.ndim == 2:
                data = eeg_data[i]
            else:
                data = eeg_data

            coeffs = pywt.wavedec(data, wavelet, level=level)
            wavelet_coeffs.append(coeffs)

            # 小波能量
            for j, coeff in enumerate(coeffs):
                features[f'wavelet_energy_level_{j}'] = np.sum(coeff ** 2)

            # 小波熵
            wavelet_entropy = []
            for coeff in coeffs:
                energy = np.sum(coeff ** 2)
                wavelet_entropy.append(-energy * np.log(energy + 1e-10))
            features['wavelet_entropy'] = np.sum(wavelet_entropy)

        return features

    def extract_spatial_features(self, eeg_data):
        """提取空域特征"""
        features = {}

        if eeg_data.ndim == 2:  # (channels, features)
            n_channels = eeg_data.shape[0]

            # 通道间相关性
            correlation_matrix = np.corrcoef(eeg_data)
            features['correlation_matrix'] = correlation_matrix

            # 脑区划分
            t_channels = list(range(6))  # 颞叶
            p_channels = list(range(6, min(17, n_channels)))  # 枕叶

            # 脑区内和脑区间相关性
            if len(t_channels) > 1:
                intra_t_corr = np.mean(correlation_matrix[np.ix_(t_channels, t_channels)])
                features['intra_temporal_correlation'] = intra_t_corr

            if len(p_channels) > 1:
                intra_p_corr = np.mean(correlation_matrix[np.ix_(p_channels, p_channels)])
                features['intra_parietal_correlation'] = intra_p_corr

            if len(t_channels) > 0 and len(p_channels) > 0:
                inter_tp_corr = np.mean(correlation_matrix[np.ix_(t_channels, p_channels)])
                features['inter_temporal_parietal_correlation'] = inter_tp_corr

            # 全局场功率 (GFP)
            gfp = np.std(eeg_data, axis=0)
            features['global_field_power_mean'] = np.mean(gfp)
            features['global_field_power_std'] = np.std(gfp)

        return features

    def extract_all_features(self, eeg_data):
        """提取所有特征"""
        features = {}

        # 时域特征
        time_features = self.extract_time_domain_features(eeg_data)
        features.update(time_features)

        # 频域特征
        freq_features = self.extract_frequency_domain_features(eeg_data)
        features.update(freq_features)

        # 时频特征
        tf_features = self.extract_time_frequency_features(eeg_data)
        features.update(tf_features)

        # 空域特征
        spatial_features = self.extract_spatial_features(eeg_data)
        features.update(spatial_features)

        return features


class FeatureVisualizer:
    """特征可视化"""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_time_domain_features(self, eeg_data, labels=None, save_path=None):
        """绘制时域特征"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        extractor = EEGFeatureExtractor()
        features = extractor.extract_time_domain_features(eeg_data)

        # 绘制各个特征
        feature_names = ['mean', 'std', 'skewness', 'kurtosis', 'zero_crossing_rate', 'activity']
        titles = ['均值', '标准差', '偏度', '峰度', '过零率', 'Hjorth活动度']

        for i, (feature, title) in enumerate(zip(feature_names, titles)):
            if feature in features:
                ax = axes[i]
                data = features[feature].flatten()

                if labels is not None:
                    # 按类别绘制
                    unique_labels = np.unique(labels)
                    for label in unique_labels:
                        idx = labels == label
                        ax.hist(data[idx], alpha=0.5, label=f'Class {label}', bins=30)
                    ax.legend()
                else:
                    ax.hist(data, bins=50, alpha=0.7)

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        plt.suptitle('时域特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_frequency_domain_features(self, eeg_data, labels=None, save_path=None):
        """绘制频域特征"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        extractor = EEGFeatureExtractor()
        features = extractor.extract_frequency_domain_features(eeg_data)

        # 频带功率
        if 'band_powers' in features:
            band_names = list(features['band_powers'].keys())

            for i, band in enumerate(band_names[:6]):  # 最多显示6个频带
                if i < len(axes):
                    ax = axes[i]
                    data = features['band_powers'][band].flatten()

                    if labels is not None:
                        unique_labels = np.unique(labels)
                        for label in unique_labels:
                            idx = labels == label
                            ax.hist(data[idx], alpha=0.5, label=f'Class {label}', bins=30)
                        ax.legend()
                    else:
                        ax.hist(data, bins=50, alpha=0.7)

                    ax.set_title(f'{band}频带功率', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Power')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)

        plt.suptitle('频域特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_spatial_features(self, eeg_data, save_path=None):
        """绘制空域特征"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        extractor = EEGFeatureExtractor()
        features = extractor.extract_spatial_features(eeg_data)

        # 相关性矩阵
        if 'correlation_matrix' in features:
            ax = axes[0]
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(features['correlation_matrix'],
                        cmap=cmap, center=0,
                        square=True, linewidths=.5,
                        cbar_kws={"shrink": .5}, ax=ax)
            ax.set_title('通道间相关性矩阵', fontsize=12, fontweight='bold')
            ax.set_xlabel('通道')
            ax.set_ylabel('通道')

        # 脑区相关性
        ax = axes[1]
        brain_region_features = []
        feature_names = []

        for key in ['intra_temporal_correlation', 'intra_parietal_correlation',
                    'inter_temporal_parietal_correlation']:
            if key in features:
                brain_region_features.append(features[key])
                feature_names.append(key.replace('_', ' ').title())

        if brain_region_features:
            bars = ax.bar(range(len(brain_region_features)), brain_region_features)
            ax.set_xticks(range(len(brain_region_features)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_ylabel('相关性系数')
            ax.set_title('脑区间相关性', fontsize=12, fontweight='bold')

            # 添加数值标签
            for bar, value in zip(bars, brain_region_features):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{value:.3f}', ha='center', va='bottom')

        # GFP分布
        if 'global_field_power_mean' in features and 'global_field_power_std' in features:
            ax = axes[2]
            x = np.linspace(0, features['global_field_power_mean'] + 3 * features['global_field_power_std'], 100)
            from scipy.stats import norm
            y = norm.pdf(x, features['global_field_power_mean'], features['global_field_power_std'])
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, 0, y, alpha=0.3)
            ax.set_xlabel('Global Field Power')
            ax.set_ylabel('Probability Density')
            ax.set_title('全局场功率分布', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle('空域特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tsne_visualization(self, features, labels, save_path=None):
        """t-SNE降维可视化"""
        # 确保特征为2D数组
        if isinstance(features, dict):
            # 将特征字典转换为数组
            feature_list = []
            for key in features:
                if isinstance(features[key], np.ndarray):
                    feature_list.append(features[key].flatten())
            features_array = np.column_stack(feature_list)
        else:
            features_array = features

        # 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_array)

        plt.figure(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                idx = labels == label
                plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                            c=[color], label=f'Class {label}', alpha=0.6, s=50)

            plt.legend()
        else:
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)

        plt.title('t-SNE特征降维可视化', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fatigue_analysis(self, eeg_data, labels, save_path=None):
        """疲劳相关分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        extractor = EEGFeatureExtractor()
        features = extractor.extract_all_features(eeg_data)

        # 提取疲劳相关特征
        fatigue_features = {}

        # Alpha/Theta比率
        if 'band_ratios' in features and 'alpha_theta' in features['band_ratios']:
            fatigue_features['Alpha/Theta'] = features['band_ratios']['alpha_theta']

        # Beta/Alpha比率
        if 'band_ratios' in features and 'beta_alpha' in features['band_ratios']:
            fatigue_features['Beta/Alpha'] = features['band_ratios']['beta_alpha']

        # Theta/Beta比率
        if 'band_ratios' in features and 'theta_beta' in features['band_ratios']:
            fatigue_features['Theta/Beta'] = features['band_ratios']['theta_beta']

        # Alpha相对功率
        if 'relative_powers' in features and 'rel_alpha' in features['relative_powers']:
            fatigue_features['Alpha相对功率'] = features['relative_powers']['rel_alpha']

        # 按疲劳状态绘制特征分布
        if labels is not None:
            unique_labels = np.unique(labels)
            label_names = ['清醒', '轻度疲劳', '重度疲劳']
            colors = ['green', 'orange', 'red']

            for i, (feat_name, feat_data) in enumerate(list(fatigue_features.items())[:4]):
                if i < len(axes):
                    ax = axes[i]

                    # 准备箱线图数据
                    box_data = []
                    for label in unique_labels:
                        idx = labels == label
                        box_data.append(feat_data.flatten()[idx])

                    bp = ax.boxplot(box_data, labels=[label_names[l] for l in unique_labels],
                                    patch_artist=True)

                    # 设置颜色
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)

                    ax.set_title(feat_name, fontsize=12, fontweight='bold')
                    ax.set_ylabel('值')
                    ax.grid(True, alpha=0.3)

                    # 添加统计显著性标记（简化版）
                    if len(box_data) > 1:
                        # 这里可以添加统计检验
                        pass

        plt.suptitle('疲劳相关特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_dataset_statistics(dataset, config):
    """分析数据集统计特征"""
    print("=" * 60)
    print("数据集统计分析")
    print("=" * 60)

    # 提取特征
    extractor = EEGFeatureExtractor(sampling_rate=config.sampling_rate)

    # 随机选择一些样本进行分析
    sample_indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)

    all_features = []
    all_labels = []

    for idx in sample_indices:
        sample = dataset[idx]
        eeg_data = sample['eeg'].numpy()
        label = sample['label'].numpy()

        features = extractor.extract_all_features(eeg_data)

        # 简化特征：只取均值
        flat_features = []
        for key in features:
            if isinstance(features[key], np.ndarray):
                flat_features.append(np.mean(features[key]))

        all_features.append(flat_features)
        all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.concatenate(all_labels)

    print(f"特征数量: {all_features.shape[1]}")
    print(f"样本数量: {all_features.shape[0]}")
    print(f"标签分布: {np.bincount(all_labels.flatten().astype(int))}")

    # 特征重要性分析（使用PCA）
    pca = PCA(n_components=min(10, all_features.shape[1]))
    pca.fit(all_features)

    print(f"\nPCA分析:")
    print(f"  累计解释方差比: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"  前3个主成分解释方差: {np.sum(pca.explained_variance_ratio_[:3]):.3f}")

    # 可视化
    visualizer = FeatureVisualizer()

    # 绘制PCA可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('PCA主成分解释方差比', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比')
    plt.title('PCA累计解释方差比', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{config.figure_dir}/pca_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    return all_features, all_labels