import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

# 归一化和反归一化辅助函数(Normalized and inverse normalized auxiliary functions)
def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

# 时间序列切片函数(Time Series Slicing function)
def slice_time_series(time_series, slide_win, slide_stride):
    """使用滑动窗口对时间序列进行切片"""
    slices = []
    total_length = len(time_series)
    for i in range(0, total_length - slide_win + 1, slide_stride):
        slices.append(time_series[i:i + slide_win])
    return slices

# 数据标准化函数(Data normalization function)
def normalize_data(data, feature_range=(-1, 1)):
    """将数据进行Min-Max标准化"""
    scaler = MinMaxScaler()
    all_data = np.concatenate(data)
    scaler.fit(all_data)
    normalized_time_series = []
    min_range, max_range = feature_range
    for ts in data:
        normalized = scaler.transform(ts)
        if min_range == -1 and max_range == 1:
            normalized = normalize_to_neg_one_to_one(normalized)
        normalized_time_series.append(normalized)
    return normalized_time_series, scaler

# 数据反标准化函数(Data denormalization function)
def denormalize_data(normalized, scaler, feature_range=(-1, 1)):
    """将标准化的数据反标准化回原始范围"""
    min_range, max_range = feature_range
    if isinstance(normalized, list):
        denormalized_time_series = []
        for ts in normalized:
            if min_range == -1 and max_range == 1:
                ts = unnormalize_to_zero_to_one(ts)
            ts_denormalized = scaler.inverse_transform(ts)
            denormalized_time_series.append(ts_denormalized)
        return denormalized_time_series
    else:
        if min_range == -1 and max_range == 1:
            normalized = unnormalize_to_zero_to_one(normalized)
        denormalized = scaler.inverse_transform(normalized)
        return denormalized

# 处理多维度时间序列切片函数 - 联合处理所有维度(Processing multidimensional time series slicing function - jointly processing all dimensions)
def process_multi_dim_time_series_slice(time_series, slice_idx, slide_win, config):
    """处理多维度时间序列切片并进行峰值检测和傅里叶变换
    现在对所有维度的数据进行联合处理，而不是单独处理每个维度
    """
    n = len(time_series)
    fs = 1.0  # 假设采样频率为 1Hz
    # 对每个维度进行傅里叶变换
    fft_result = np.fft.fft(time_series, axis=0)
    # 计算每个维度的振幅谱
    amplitude = np.abs(fft_result)
    # 获取频率轴
    freq = np.fft.fftfreq(n, d=1 / fs)
    positive_mask = freq > 0
    freq_positive = freq[positive_mask]
    # 计算所有维度的平均振幅谱 - 这是关键变化点
    amplitude_positive = np.mean(amplitude[positive_mask], axis=1)
    # 计算频率分辨率
    freq_resolution = fs / n
    # 计算振幅的标准差（用于高度阈值）
    amp_std = np.std(amplitude_positive)
    peak_height_threshold = config.peak_height_coeff * amp_std
    # 计算最小峰值距离（基于频率分辨率系数）
    peak_distance = int(np.ceil(config.peak_dist_coeff * freq_resolution * n))  # 转换为采样点间隔
    # 过滤掉接近零的频率（可能是噪声或趋势）
    valid_freqs_mask = np.abs(freq_positive) >= config.near_zero_threshold
    valid_freq_positive = freq_positive[valid_freqs_mask]
    valid_amplitude_positive = amplitude_positive[valid_freqs_mask]
    # 执行峰值检测（使用动态阈值和距离）
    peaks, _ = find_peaks(valid_amplitude_positive, height=peak_height_threshold,
                          distance=peak_distance)
    # 提取主要周期
    peak_amplitudes = valid_amplitude_positive[peaks]
    if len(peaks) >= 3:
        # 找到振幅最大的三个峰值
        top3_indices = np.argsort(peak_amplitudes)[-3:][::-1]
        top3_freqs = valid_freq_positive[peaks][top3_indices]
    elif len(peaks) > 0:
        print(
            f"警告: 在切片{slice_idx}中找到的峰值不足3个，使用所有可用的{len(peaks)}个峰值，并以slide_win={slide_win}对应的频率填充剩余位置")
        top3_freqs = valid_freq_positive[peaks]
        fill_freq = 1.0 / slide_win
        while len(top3_freqs) < 3:
            top3_freqs = np.append(top3_freqs, fill_freq)
    else:
        print(f"错误: 在切片{slice_idx}的有效频率范围内未找到峰值，使用slide_win={slide_win}对应的频率")
        top3_freqs = np.array([1.0 / slide_win] * 3)
    # 计算周期（频率的倒数）并取整
    top3_periods = np.round(1.0 / np.array(top3_freqs)).astype(int) if top3_freqs.size > 0 else []
    top3_periods = sorted(top3_periods)
    print(f"\n切片{slice_idx}的联合周期（取整后，从小到大排序）：")
    for period in top3_periods:
        print(f"{period}")
    return freq_positive, amplitude_positive, top3_freqs, top3_periods

# 优化后的边生成函数(Optimized edge generating function)
def generate_edges_efficiently(num_steps, periods, use_period_index, k):
    """高效生成满足周期条件的边并去重"""
    edges = set()
    for i in range(num_steps - 1):
        edges.add((i, i + 1))  # 时间序列中的相邻节点连接
    if use_period_index is not None:
        if use_period_index < len(periods):
            selected_period = periods[use_period_index]
            for start in range(num_steps):
                max_k = (num_steps - start - 1) // selected_period
                if max_k >= 1:
                    ks = np.arange(1, max_k + 1)
                    js = start + ks * selected_period
                    new_edges = [(start, j) for j in js]
                    edges.update(new_edges)
            # 添加剩下两个周期的边，考虑延伸次数k
            remaining_periods = [p for idx, p in enumerate(periods) if idx != use_period_index]
            for period in remaining_periods:
                for start in range(num_steps):
                    for multiplier in range(1, k + 1):
                        end = start + multiplier * period
                        if end < num_steps:
                            edges.add((start, end))
        else:
            print(f"警告: 指定的周期索引 {use_period_index} 超出范围，使用所有周期。")
    return list(edges)

# 生成邻接矩阵和节点特征矩阵(Generating adjacency matrix and node characteristic matrix)
def generate_graph_matrices(all_slices, all_slices_periods, config):
    """生成邻接矩阵和节点特征矩阵，每个切片对应一张图"""
    all_adj_matrices = []
    all_node_features = []
    for var_idx, (slices, periods_list) in enumerate(zip(all_slices, all_slices_periods)):
        for slice_idx, (slice_data, periods) in enumerate(zip(slices, periods_list)):
            num_nodes = len(slice_data)
            # 生成邻接矩阵
            edges = generate_edges_efficiently(num_nodes, periods, config.use_period_index, config.k)
            adj_matrix = np.zeros((num_nodes, num_nodes))
            for i, j in edges:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # 无向图
            # 生成节点特征矩阵 (一维特征，节点对应的数据值)
            node_features = slice_data  # [num_nodes, dim]
            all_adj_matrices.append(adj_matrix)
            all_node_features.append(node_features)
    # 转换为numpy数组
    all_adj_matrices = np.array(all_adj_matrices)  # [num_graphs, num_nodes, num_nodes]
    all_node_features = np.array(all_node_features)  # [num_graphs, num_nodes, dim]
    return all_adj_matrices, all_node_features

# 自定义数据集类(Custom dataset class)
from torch.utils.data import Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self, adj_matrices, node_features):
        self.adj_matrices = adj_matrices
        self.node_features = node_features

    def __len__(self):
        return len(self.adj_matrices)

    def __getitem__(self, idx):
        adj_matrix = torch.FloatTensor(self.adj_matrices[idx])
        node_feature = torch.FloatTensor(self.node_features[idx])
        # 检查并处理NaN值
        if torch.isnan(adj_matrix).any():
            print(f"警告: 邻接矩阵样本 {idx} 包含NaN值，将其替换为0")
            adj_matrix = torch.nan_to_num(adj_matrix)
        if torch.isnan(node_feature).any():
            print(f"警告: 节点特征样本 {idx} 包含NaN值，将其替换为0")
            node_feature = torch.nan_to_num(node_feature)
        return adj_matrix, node_feature