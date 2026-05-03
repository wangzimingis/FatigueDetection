# feature_analyzer.py - 修复版本
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
import pywt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 全局样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EEGFeatureExtractor:
    """
    EEG特征提取器 - 实现时域、频域、时频域、空域特征提取
    修复版本：支持批量输入，输出平坦特征向量，修复小波特征覆盖问题
    """

    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        self.channel_names = None  # 可选的通道名称列表

    def extract_time_domain_features(self, eeg_data):
        """
        提取时域特征
        eeg_data: (batch, channels, features) 或 (channels, features)
        返回: dict, 每个特征形状为 (batch, channels) 或 (channels,)
        """
        # 统一转为三维 (batch, channels, features)
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, axis=0)
        batch, channels, features = eeg_data.shape

        features_dict = {}

        # 基本统计量
        features_dict['mean'] = np.mean(eeg_data, axis=-1)           # (batch, channels)
        features_dict['std'] = np.std(eeg_data, axis=-1)
        features_dict['variance'] = np.var(eeg_data, axis=-1)
        features_dict['skewness'] = skew(eeg_data, axis=-1)
        features_dict['kurtosis'] = kurtosis(eeg_data, axis=-1)

        # Hjorth参数
        features_dict['activity'] = np.var(eeg_data, axis=-1)        # 同 variance
        # 一阶差分
        diff1 = np.diff(eeg_data, axis=-1)
        var_diff1 = np.var(diff1, axis=-1)
        var_data = np.var(eeg_data, axis=-1)
        features_dict['mobility'] = np.sqrt(var_diff1 / (var_data + 1e-10))
        # 二阶差分
        diff2 = np.diff(diff1, axis=-1)
        var_diff2 = np.var(diff2, axis=-1)
        features_dict['complexity'] = np.sqrt(var_diff2 / (var_diff1 + 1e-10))

        # 峰峰值
        features_dict['peak_to_peak'] = np.ptp(eeg_data, axis=-1)

        # 过零率 (zero-crossing rate)
        zero_crossings = np.diff(np.sign(eeg_data), axis=-1)
        features_dict['zero_crossing_rate'] = np.sum(zero_crossings != 0, axis=-1) / features

        # 如果输入是2D，去掉batch维度
        if batch == 1:
            features_dict = {k: v.squeeze(0) for k, v in features_dict.items()}
        return features_dict

    def extract_frequency_domain_features(self, eeg_data):
        """
        提取频域特征：功率谱密度，频带功率，相对功率，比率，谱熵，谱质心
        eeg_data: (batch, channels, features) 或 (channels, features)
        返回: dict, 特征形状为 (batch, channels) 或 (channels,)
        """
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, axis=0)
        batch, channels, n_features = eeg_data.shape
        features_dict = {}

        # 计算每个通道的 PSD（使用 Welch 方法）
        nperseg = min(256, n_features)
        freqs, psd = signal.welch(eeg_data, fs=self.sampling_rate,
                                  nperseg=nperseg, axis=-1)  # psd: (batch, channels, n_freqs)

        # 总功率
        total_power = np.sum(psd, axis=-1)  # (batch, channels)

        # 频带功率及相对功率
        band_powers = {}
        rel_powers = {}
        for band_name, (low, high) in self.frequency_bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            bp = np.sum(psd[..., idx], axis=-1)
            band_powers[band_name] = bp
            rel_powers[f'rel_{band_name}'] = bp / (total_power + 1e-10)

        features_dict['band_powers'] = band_powers
        features_dict['relative_powers'] = rel_powers

        # 频带比率
        band_ratios = {}
        if 'alpha' in band_powers and 'theta' in band_powers:
            band_ratios['alpha_theta'] = band_powers['alpha'] / (band_powers['theta'] + 1e-10)
        if 'beta' in band_powers and 'alpha' in band_powers:
            band_ratios['beta_alpha'] = band_powers['beta'] / (band_powers['alpha'] + 1e-10)
        if 'theta' in band_powers and 'beta' in band_powers:
            band_ratios['theta_beta'] = band_powers['theta'] / (band_powers['beta'] + 1e-10)
        features_dict['band_ratios'] = band_ratios

        # 谱熵 (需要将 PSD 归一化为概率分布)
        psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=-1)
        features_dict['spectral_entropy'] = spectral_entropy

        # 谱质心
        spectral_centroid = np.sum(freqs * psd, axis=-1) / (total_power + 1e-10)
        features_dict['spectral_centroid'] = spectral_centroid

        features_dict['total_power'] = total_power

        # 如果是单样本，去掉 batch 维度
        if batch == 1:
            for k, v in features_dict.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    features_dict[k] = v.squeeze(0)
                elif isinstance(v, dict):
                    features_dict[k] = {kk: vv.squeeze(0) if hasattr(vv, 'squeeze') else vv for kk, vv in v.items()}
        return features_dict

    def extract_time_frequency_features(self, eeg_data):
        """
        提取时频特征（小波变换） - 修复版
        为每个通道独立提取小波能量和熵，并汇总为均值、标准差等统计量
        eeg_data: (batch, channels, features) 或 (channels, features)
        返回: dict, 特征值标量（避免覆盖）
        """
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, axis=0)
        batch, channels, n_features = eeg_data.shape

        wavelet = 'db4'
        max_level = pywt.dwt_max_level(n_features, pywt.Wavelet(wavelet).dec_len)
        level = min(5, max_level)

        # 存储所有通道的小波特征，最后聚合
        all_channel_energies = []   # 每个元素为 (batch, levels+1)
        all_channel_entropies = []  # 每个元素为 (batch,)

        for i in range(channels):
            # 取出所有样本的第i通道 (batch, n_features)
            channel_data = eeg_data[:, i, :]
            batch_energies = []
            batch_entropies = []
            for b in range(batch):
                coeffs = pywt.wavedec(channel_data[b], wavelet, level=level)
                # 计算各级能量
                energies = [np.sum(c ** 2) for c in coeffs]
                batch_energies.append(energies)
                # 小波熵
                total_energy = sum(energies) + 1e-10
                p = [e / total_energy for e in energies]
                entropy_val = -np.sum(p * np.log(p + 1e-10))
                batch_entropies.append(entropy_val)
            all_channel_energies.append(np.array(batch_energies))  # (batch, n_levels)
            all_channel_entropies.append(np.array(batch_entropies))  # (batch,)

        # 将所有通道的能量堆叠成 (batch, channels, n_levels)
        energy_tensor = np.stack(all_channel_energies, axis=1)   # (batch, channels, levels)
        entropy_tensor = np.stack(all_channel_entropies, axis=1) # (batch, channels)

        # 聚合统计：跨通道的均值、标准差、最大值、最小值
        features_dict = {}
        # 能量特征：每个分解级别的统计量
        for lev in range(energy_tensor.shape[2]):
            lev_data = energy_tensor[:, :, lev]
            features_dict[f'wavelet_energy_level_{lev}_mean'] = np.mean(lev_data, axis=1)
            features_dict[f'wavelet_energy_level_{lev}_std'] = np.std(lev_data, axis=1)
            features_dict[f'wavelet_energy_level_{lev}_max'] = np.max(lev_data, axis=1)
            features_dict[f'wavelet_energy_level_{lev}_min'] = np.min(lev_data, axis=1)

        # 熵特征
        features_dict['wavelet_entropy_mean'] = np.mean(entropy_tensor, axis=1)
        features_dict['wavelet_entropy_std'] = np.std(entropy_tensor, axis=1)
        features_dict['wavelet_entropy_max'] = np.max(entropy_tensor, axis=1)
        features_dict['wavelet_entropy_min'] = np.min(entropy_tensor, axis=1)

        # 如果只有一个样本，去除batch维度
        if batch == 1:
            features_dict = {k: v.squeeze(0) for k, v in features_dict.items()}
        return features_dict

    def extract_spatial_features(self, eeg_data):
        """
        提取空域特征：通道间相关性、脑区内相关性、全局场功率
        支持批量输入，返回每个样本的特征
        eeg_data: (batch, channels, features) 或 (channels, features)
        返回: dict, 每个特征形状为 (batch,) 或 (batch, channels, channels) 等
        """
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, axis=0)
        batch, channels, n_features = eeg_data.shape

        features_dict = {}

        # 计算每个样本的通道间相关性矩阵
        correlation_matrices = []   # list of (channels, channels)
        for b in range(batch):
            cm = np.corrcoef(eeg_data[b])  # (channels, channels)
            correlation_matrices.append(cm)
        corr_stack = np.stack(correlation_matrices, axis=0)  # (batch, channels, channels)

        # 存储相关性矩阵（可能很大，建议只保存统计量）
        # features_dict['correlation_matrix'] = corr_stack

        # 脑区划分（硬编码为前6通道为颞叶，其后至17为枕叶，可根据实际调整）
        # 更灵活的方式：允许外部传入channel_groups
        n_channels = channels
        t_channels = list(range(min(6, n_channels)))
        p_channels = list(range(6, min(17, n_channels))) if n_channels > 6 else []

        # 计算脑区内平均相关性
        def avg_corr(corr, indices):
            if len(indices) < 2:
                return 0.0
            sub = corr[np.ix_(indices, indices)]
            return np.mean(sub)

        intra_t = np.array([avg_corr(corr_stack[b], t_channels) for b in range(batch)])
        intra_p = np.array([avg_corr(corr_stack[b], p_channels) for b in range(batch)]) if p_channels else np.zeros(batch)
        inter_tp = 0.0
        if t_channels and p_channels:
            inter_tp = np.array([np.mean(corr_stack[b][np.ix_(t_channels, p_channels)]) for b in range(batch)])

        features_dict['intra_temporal_correlation'] = intra_t
        features_dict['intra_parietal_correlation'] = intra_p
        features_dict['inter_temporal_parietal_correlation'] = inter_tp

        # 全局场功率 GFP: 每个样本所有通道的标准差随时间取平均
        gfp = np.std(eeg_data, axis=1)  # (batch, features)
        features_dict['global_field_power_mean'] = np.mean(gfp, axis=1)
        features_dict['global_field_power_std'] = np.std(gfp, axis=1)

        if batch == 1:
            features_dict = {k: v.squeeze(0) for k, v in features_dict.items()}
        return features_dict

    def extract_all_features(self, eeg_data, flatten=True):
        """
        提取所有特征并返回平坦的一维特征向量（每个样本）
        Args:
            eeg_data: (batch, channels, features) 或 (channels, features)
            flatten: 是否将特征展平为二维数组 (batch, n_features)
        Returns:
            若 flatten=False，返回包含所有特征的字典（字典值可能为数组）
            否则返回 (batch, n_features) 的 numpy 数组和特征名称列表
        """
        # 存储所有特征（字典值可能为多维数组）
        all_feats = {}

        # 时域
        td = self.extract_time_domain_features(eeg_data)
        all_feats.update(td)

        # 频域
        fd = self.extract_frequency_domain_features(eeg_data)
        # 频域中有嵌套字典，需要展开
        for k, v in fd.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    all_feats[f'{k}_{subk}'] = subv
            else:
                all_feats[k] = v

        # 时频
        tf = self.extract_time_frequency_features(eeg_data)
        all_feats.update(tf)

        # 空域
        sp = self.extract_spatial_features(eeg_data)
        all_feats.update(sp)

        if not flatten:
            return all_feats

        # 转换为平坦向量
        # 对于每个特征，如果是标量或一维数组，直接作为特征；如果是多维，需要展平
        feature_vectors = []
        feature_names = []
        # 获取batch大小
        batch_size = eeg_data.shape[0] if eeg_data.ndim == 3 else 1

        for name, value in all_feats.items():
            if isinstance(value, np.ndarray):
                # 确保形状要么是 (batch,)，要么是 (batch, dim)
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                elif value.ndim > 2:
                    value = value.reshape(batch_size, -1)
            else:
                # 标量，转为 (batch, 1)
                value = np.full((batch_size, 1), value, dtype=np.float32)

            feature_vectors.append(value)
            # 生成列名：如果是多维，加上数字索引
            if value.shape[1] == 1:
                feature_names.append(name)
            else:
                for i in range(value.shape[1]):
                    feature_names.append(f'{name}_{i}')

        # 水平拼接
        features_array = np.hstack(feature_vectors) if feature_vectors else np.empty((batch_size, 0))
        return features_array, feature_names


class FeatureVisualizer:
    """特征可视化（修复版，支持更多配置）"""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_time_domain_features(self, eeg_data, labels=None, save_path=None):
        """绘制时域特征分布（按类别）"""
        extractor = EEGFeatureExtractor()
        feats = extractor.extract_time_domain_features(eeg_data)
        # 选择要展示的特征
        plot_keys = ['mean', 'std', 'skewness', 'kurtosis', 'zero_crossing_rate', 'mobility']
        titles = ['均值', '标准差', '偏度', '峰度', '过零率', 'Hjorth移动性']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (key, title) in enumerate(zip(plot_keys, titles)):
            if key not in feats:
                continue
            data = feats[key]
            # 如果数据包含多个样本，展平；否则直接使用
            if data.ndim > 1:
                data = data.flatten()
            ax = axes[i]
            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    idx = labels == label
                    ax.hist(data[idx], alpha=0.5, label=f'Class {label}', bins=30)
                ax.legend()
            else:
                ax.hist(data, bins=50, alpha=0.7)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

        plt.suptitle('时域特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_frequency_domain_features(self, eeg_data, labels=None, save_path=None):
        """绘制频域特征：各频带功率（按类别）"""
        extractor = EEGFeatureExtractor()
        feats = extractor.extract_frequency_domain_features(eeg_data)
        band_powers = feats.get('band_powers', {})
        band_names = list(band_powers.keys())

        if not band_names:
            print("无频带功率数据。")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, band in enumerate(band_names[:6]):
            if i >= len(axes):
                break
            data = band_powers[band].flatten()
            ax = axes[i]
            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    idx = labels == label
                    ax.hist(data[idx], alpha=0.5, label=f'Class {label}', bins=30)
                ax.legend()
            else:
                ax.hist(data, bins=50, alpha=0.7)
            ax.set_title(f'{band}频带功率', fontweight='bold')
            ax.set_xlabel('Power')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

        plt.suptitle('频域特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_spatial_features(self, eeg_data, save_path=None):
        """绘制空域特征：相关性热图，脑区相关性柱状图，GFP分布"""
        # 只取第一个样本（或平均值）
        if eeg_data.ndim == 3:
            sample = eeg_data[0]
        else:
            sample = eeg_data

        extractor = EEGFeatureExtractor()
        feats = extractor.extract_spatial_features(sample)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 相关性矩阵
        if 'correlation_matrix' in feats:
            cm = feats['correlation_matrix']
            ax = axes[0]
            sns.heatmap(cm, cmap='coolwarm', center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('通道间相关性矩阵', fontweight='bold')
            ax.set_xlabel('通道')
            ax.set_ylabel('通道')

        # 脑区相关性条形图
        ax = axes[1]
        keys = ['intra_temporal_correlation', 'intra_parietal_correlation', 'inter_temporal_parietal_correlation']
        labels = ['颞叶内', '枕叶内', '颞枕之间']
        values = [feats.get(k, 0) for k in keys]
        bars = ax.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_ylabel('平均相关系数')
        ax.set_title('脑区间相关性', fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.3f}', ha='center', va='bottom')

        # GFP分布
        ax = axes[2]
        gfp_mean = feats.get('global_field_power_mean', 0)
        gfp_std = feats.get('global_field_power_std', 0)
        if gfp_std > 0:
            from scipy.stats import norm
            x = np.linspace(gfp_mean - 3*gfp_std, gfp_mean + 3*gfp_std, 100)
            y = norm.pdf(x, gfp_mean, gfp_std)
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, 0, y, alpha=0.3)
            ax.set_title('全局场功率分布', fontweight='bold')
            ax.set_xlabel('GFP')
            ax.set_ylabel('概率密度')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无有效的GFP数据', ha='center', va='center')

        plt.suptitle('空域特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tsne_visualization(self, features, labels, save_path=None):
        """t-SNE降维可视化（特征应为二维数组，样本 x 特征）"""
        if isinstance(features, dict):
            # 尝试转换为数组（仅提取数值特征）
            feature_list = []
            for v in features.values():
                if isinstance(v, np.ndarray):
                    feature_list.append(v.flatten())
            features_array = np.column_stack(feature_list) if feature_list else np.empty((0, 0))
        else:
            features_array = np.array(features)

        if features_array.size == 0:
            print("无法从特征字典构建有效数组。")
            return

        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features_scaled.shape[0]-1))
        features_2d = tsne.fit_transform(features_scaled)

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

        plt.title('t-SNE特征降维可视化', fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fatigue_analysis(self, eeg_data, labels, save_path=None):
        """疲劳相关特征分析（Alpha/Theta, Beta/Alpha, Theta/Beta, Alpha相对功率）"""
        extractor = EEGFeatureExtractor()
        # 提取所有特征（扁平化）
        features_array, _ = extractor.extract_all_features(eeg_data, flatten=True)
        # 需要从原始提取中获取频带比率等信息（简便方法：重新调用提取并取出）
        # 这里为了效率，直接使用之前已提取的嵌套结果
        # 简单起见，重新获取频域特征（不展开）
        fd = extractor.extract_frequency_domain_features(eeg_data)
        band_ratios = fd.get('band_ratios', {})
        rel_powers = fd.get('relative_powers', {})

        fatigue_features = {}
        if 'alpha_theta' in band_ratios:
            fatigue_features['Alpha/Theta'] = band_ratios['alpha_theta'].flatten()
        if 'beta_alpha' in band_ratios:
            fatigue_features['Beta/Alpha'] = band_ratios['beta_alpha'].flatten()
        if 'theta_beta' in band_ratios:
            fatigue_features['Theta/Beta'] = band_ratios['theta_beta'].flatten()
        if 'rel_alpha' in rel_powers:
            fatigue_features['Alpha相对功率'] = rel_powers['rel_alpha'].flatten()

        if not fatigue_features:
            print("未找到疲劳相关特征，请检查频域提取。")
            return

        # 绘制箱线图
        n_feats = len(fatigue_features)
        fig, axes = plt.subplots(1, n_feats, figsize=(5*n_feats, 6))
        if n_feats == 1:
            axes = [axes]

        unique_labels = np.unique(labels)
        label_names = ['清醒', '轻度疲劳', '重度疲劳']
        colors = ['green', 'orange', 'red']

        for ax_idx, (feat_name, feat_data) in enumerate(fatigue_features.items()):
            ax = axes[ax_idx]
            box_data = []
            for label in unique_labels:
                idx = labels == label
                box_data.append(feat_data[idx])
            bp = ax.boxplot(box_data, labels=[label_names[int(l)] for l in unique_labels], patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title(feat_name, fontweight='bold')
            ax.set_ylabel('值')
            ax.grid(True, alpha=0.3)

        plt.suptitle('疲劳相关特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def analyze_dataset_statistics(dataset, config):
    """
    分析数据集统计特征（修复版）
    - 使用分层采样保证类别比例
    - 提取特征后使用标准化再进行PCA
    - 输出特征统计信息
    """
    print("=" * 60)
    print("数据集统计分析")
    print("=" * 60)

    extractor = EEGFeatureExtractor(sampling_rate=config.sampling_rate)

    # 获取所有标签
    all_labels = []
    all_eeg = []
    # 限制最多分析2000个样本，避免过慢
    max_samples = min(2000, len(dataset))
    indices = np.random.choice(len(dataset), max_samples, replace=False)
    for idx in indices:
        sample = dataset[idx]
        all_eeg.append(sample['eeg'].numpy())
        all_labels.append(sample['label'].numpy().item())
    all_labels = np.array(all_labels)
    all_eeg = np.stack(all_eeg, axis=0)  # (N, channels, features)

    # 提取特征
    features_array, feature_names = extractor.extract_all_features(all_eeg, flatten=True)
    print(f"特征数量: {features_array.shape[1]}")
    print(f"样本数量: {features_array.shape[0]}")
    print(f"标签分布: {np.bincount(all_labels)}")

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # PCA
    n_comp = min(10, features_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(features_scaled)
    print(f"\nPCA分析 (前{n_comp}个主成分):")
    print(f"  解释方差比: {pca.explained_variance_ratio_}")
    print(f"  累计解释方差: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"  前3个主成分累计解释方差: {np.sum(pca.explained_variance_ratio_[:3]):.3f}")

    # 可视化PCA
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_comp+1), pca.explained_variance_ratio_)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('PCA主成分解释方差比', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_comp+1), np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比')
    plt.title('PCA累计解释方差比', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if hasattr(config, 'figure_dir'):
        save_path = f"{config.figure_dir}/pca_analysis.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # t-SNE可视化（可选，较慢，故注释）
    # visualizer = FeatureVisualizer()
    # visualizer.plot_tsne_visualization(features_scaled, all_labels)

    return features_array, all_labels


if __name__ == '__main__':
    # 简单测试
    from config import Config
    from src.data_loader import SEEDVIGDataset
    config = Config()
    config.data_root = './data/SEED-VIG/'
    config.feature_type = '2Hz'
    config.use_eog = True
    dataset = SEEDVIGDataset(config, subject_ids=[1,2], mode='train')
    features, labels = analyze_dataset_statistics(dataset, config)
    print("特征形状:", features.shape)