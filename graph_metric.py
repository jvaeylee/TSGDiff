import numpy as np
from scipy.signal import find_peaks
import torch
import os
import warnings
from scipy.linalg import eigh
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')

# 设置设备(Setup device)
device = "cuda" if torch.cuda.is_available() else "cpu"


# 集中管理所有输入参数(Centralized management of all input parameters)
class Config:
    def __init__(self):
        # 文件路径配置
        self.original_data_file = r'./original_date.npy'
        self.generated_data_file = r'./generated_date.npy'
        self.output_dir = './output/evaluation'  # 输出目录

        # 峰值检测参数
        self.peak_height_coeff = 0.01  # 峰值高度系数（相对于标准差）
        self.peak_dist_coeff = 0.01  # 峰值最小距离系数（相对于频率分辨率）
        self.near_zero_threshold = 0  # 频率接近零的阈值

        # 使用第几个周期生成边（用于图构建）
        self.use_period_index = 0  # 使用第几个周期生成边

        # 新增参数k
        self.k = 0


# 处理多维度时间序列切片函数（直接处理已切片的数据）(Processing multidimensional time series slicing functions)
def process_multi_dim_time_series_slice(time_series, slice_idx, config):
    """处理多维度时间序列切片并进行峰值检测和傅里叶变换"""
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
    # 计算所有维度的平均振幅谱
    amplitude_positive = np.mean(amplitude[positive_mask], axis=1)
    # 计算频率分辨率
    freq_resolution = fs / n
    # 计算振幅的标准差（用于高度阈值）
    amp_std = np.std(amplitude_positive)
    peak_height_threshold = config.peak_height_coeff * amp_std
    # 计算最小峰值距离（基于频率分辨率系数）
    peak_distance = int(np.ceil(config.peak_dist_coeff * freq_resolution * n))
    # 过滤掉接近零的频率
    valid_freqs_mask = np.abs(freq_positive) >= config.near_zero_threshold
    valid_freq_positive = freq_positive[valid_freqs_mask]
    valid_amplitude_positive = amplitude_positive[valid_freqs_mask]
    # 执行峰值检测
    peaks, _ = find_peaks(valid_amplitude_positive, height=peak_height_threshold,
                          distance=peak_distance)
    # 提取主要周期
    peak_amplitudes = valid_amplitude_positive[peaks]
    if len(peaks) >= 3:
        top3_indices = np.argsort(peak_amplitudes)[-3:][::-1]
        top3_freqs = valid_freq_positive[peaks][top3_indices]
    elif len(peaks) > 0:
        print(
            f"警告: 在切片{slice_idx}中找到的峰值不足3个，使用所有可用的{len(peaks)}个峰值")
        top3_freqs = valid_freq_positive[peaks]
        fill_freq = 1.0 / n  # 使用当前切片长度对应的频率
        while len(top3_freqs) < 3:
            top3_freqs = np.append(top3_freqs, fill_freq)
    else:
        print(f"错误: 在切片{slice_idx}的有效频率范围内未找到峰值，使用切片长度对应的频率")
        top3_freqs = np.array([1.0 / n] * 3)  # 使用当前切片长度对应的频率
    # 计算周期（频率的倒数）并取整
    top3_periods = np.round(1.0 / np.array(top3_freqs)).astype(int) if top3_freqs.size > 0 else []
    top3_periods = sorted(top3_periods)
    return freq_positive, amplitude_positive, top3_freqs, top3_periods


# 生成边函数(Generating edge functions)
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
            # 生成节点特征矩阵
            node_features = slice_data  # [num_nodes, dim]
            all_adj_matrices.append(adj_matrix)
            all_node_features.append(node_features)
    # 转换为numpy数组
    all_adj_matrices = np.array(all_adj_matrices)  # [num_graphs, num_nodes, num_nodes]
    all_node_features = np.array(all_node_features)  # [num_graphs, num_nodes, dim]
    return all_adj_matrices, all_node_features


# 图编辑距离(Graph Edit Similarity)
def calculate_graph_edit_distance(original_adj, generated_adj):
    """计算两个图之间的编辑距离（简化版）"""
    # 计算邻接矩阵差异
    diff = np.abs(original_adj - generated_adj)
    # 编辑距离为差异的总和
    edit_distance = np.sum(diff)
    # 归一化到 [0, 1] 范围
    max_possible_edges = original_adj.shape[0] * original_adj.shape[1]
    normalized_distance = edit_distance / max_possible_edges
    # 转换为相似度 (值越大越相似)
    similarity = 1 - normalized_distance
    return similarity


# 结构熵相似度(Structural Entropy Similarity)
def calculate_entropy_similarity(original_adj, generated_adj):
    """计算两个图的结构熵相似度，基于节点度分布的熵"""

    def compute_degree_entropy(adj_matrix):
        """计算图的度分布熵"""
        # 计算每个节点的度
        degrees = np.sum(adj_matrix, axis=1)
        # 计算度分布
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        # 计算概率分布
        probs = counts / np.sum(counts)
        # 计算熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    # 计算结构熵
    original_entropy = compute_degree_entropy(original_adj)
    generated_entropy = compute_degree_entropy(generated_adj)

    # 计算差异（越小越相似）
    diff = abs(original_entropy - generated_entropy)

    # 转换为相似度
    similarity = 1 / (1 + diff)

    return similarity


# 组合相似度（加权平均）(Combinatorial similarity)
def calculate_combined_similarity(original_adj, generated_adj):
    """计算多种相似度指标的加权组合"""
    # 计算各项相似度
    edit_sim = calculate_graph_edit_distance(original_adj, generated_adj)
    entropy_sim = calculate_entropy_similarity(original_adj, generated_adj)

    # 加权平均（权重相等）
    combined_sim = (edit_sim + entropy_sim) / 2

    return {
        'edit_similarity': edit_sim,
        'entropy_similarity': entropy_sim,
        'combined_similarity': combined_sim
    }


# 主函数(main)
def main():
    # 创建配置实例
    config = Config()

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 读取数据
    try:
        print(f"读取原始数据: {config.original_data_file}")
        original_data = np.load(config.original_data_file)
        print(f"读取生成数据: {config.generated_data_file}")
        generated_data = np.load(config.generated_data_file)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        return
    except Exception as e:
        print(f"错误: 读取数据时出错 - {e}")
        return

    print(f"原始数据形状: {original_data.shape}")
    print(f"生成数据形状: {generated_data.shape}")

    # 验证数据维度
    if len(original_data.shape) != 3 or len(generated_data.shape) != 3:
        print("错误: 数据格式不正确。期望形状为 (样本数, 时间步长, 特征数)")
        return

    # 处理数据并提取周期信息
    print("\n处理数据并提取周期信息...")
    original_slices = original_data  # 直接使用加载的数据，无需切片
    generated_slices = generated_data  # 直接使用加载的数据，无需切片

    original_slices_periods = []
    generated_slices_periods = []

    # 提取原始数据的周期信息
    for i in range(original_slices.shape[0]):
        sample = original_slices[i]
        _, _, _, top3_periods = process_multi_dim_time_series_slice(sample, i, config)
        original_slices_periods.append(top3_periods)

    # 提取生成数据的周期信息
    for i in range(generated_slices.shape[0]):
        sample = generated_slices[i]
        _, _, _, top3_periods = process_multi_dim_time_series_slice(sample, i, config)
        generated_slices_periods.append(top3_periods)

    print(f"原始数据样本数量: {len(original_slices)}")
    print(f"生成数据样本数量: {len(generated_slices)}")

    # 生成图结构
    print("\n生成图结构...")
    print("生成原始数据的图结构...")
    original_adj_matrices, _ = generate_graph_matrices([original_slices], [original_slices_periods], config)
    print("生成生成数据的图结构...")
    generated_adj_matrices, _ = generate_graph_matrices([generated_slices], [generated_slices_periods], config)

    print(f"原始数据的图数量: {len(original_adj_matrices)}")
    print(f"生成数据的图数量: {len(generated_adj_matrices)}")

    # 确保两种数据生成的图数量相同（取较小值）
    num_graphs = min(len(original_adj_matrices), len(generated_adj_matrices))
    print(f"\n将比较前 {num_graphs} 个图...")

    # 计算图结构相似度
    print("\n计算图结构相似度指标...")
    edit_similarities = []
    entropy_similarities = []
    combined_similarities = []

    # 对每对图计算相似度
    for i in range(num_graphs):
        if i % 100 == 0:
            print(f"处理图对 {i}/{num_graphs}...")

        original_adj = original_adj_matrices[i]
        generated_adj = generated_adj_matrices[i]

        # 计算相似度
        similarities = calculate_combined_similarity(original_adj, generated_adj)

        # 收集结果
        edit_similarities.append(similarities['edit_similarity'])
        entropy_similarities.append(similarities['entropy_similarity'])
        combined_similarities.append(similarities['combined_similarity'])

    # 计算平均相似度
    avg_edit_similarity = np.mean(edit_similarities)
    avg_entropy_similarity = np.mean(entropy_similarities)
    avg_combined_similarity = np.mean(combined_similarities)

    # 打印结果
    print("\n=== 图结构相似度结果 ===")
    print(f"图编辑相似度: {avg_edit_similarity:.6f}")
    print(f"结构熵相似度: {avg_entropy_similarity:.6f}")
    print(f"综合相似度: {avg_combined_similarity:.6f}")

    # 保存结果到文件
    results_file = os.path.join(config.output_dir, "graph_similarity_results.txt")
    with open(results_file, 'w') as f:
        f.write("=== 图结构相似度结果 ===\n\n")
        f.write(f"图编辑相似度: {avg_edit_similarity:.6f}\n")
        f.write(f"结构熵相似度: {avg_entropy_similarity:.6f}\n")
        f.write(f"综合相似度: {avg_combined_similarity:.6f}\n\n")
        f.write("所有相似度指标范围均为 [0, 1]，值越大表示越相似\n")

    print(f"\n结果已保存至: {results_file}")


if __name__ == "__main__":
    main()