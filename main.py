import pandas as pd
import numpy as np
import os
import yaml
from config import Config
from data_utils import (slice_time_series, normalize_data,
                        process_multi_dim_time_series_slice,
                        generate_graph_matrices, GraphDataset)
from model import GraphVAE
from train_utils import train, sample
from visualization_utils import visualization
from evaluation_utils import (calculate_discriminative_score,
                              calculate_predictive_score,
                              calculate_correlational_score)
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # 创建配置实例(Create configuration instance)
    config = Config()
    try:
        # 读取 YAML 配置文件(Read yaml configuration file)
        with open('./config/etth.yaml', 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            # 根据需要更新 config 对象
            for key, value in yaml_config.items():
                setattr(config, key, value)
    except FileNotFoundError:
        print("警告: 未找到 YAML 配置文件，使用默认配置")
    except Exception as e:
        print(f"错误: 读取 YAML 配置文件时出错 - {e}")

    # 读取数据
    try:
        data = pd.read_csv(config.data_file)
    except FileNotFoundError:
        print(f"错误: 文件 {config.data_file} 不存在")
        return
    # 检查数据是否包含NaN
    if data.isna().any().any():
        print("警告: 输入数据包含NaN值，在处理前将其填充为0")
        data = data.fillna(0)
    # 排除 date 列，提取数值列作为时间序列数据
    time_series_list = [data[col].values for col in data.columns if col != 'date']
    # 将所有变量的数据合并成一个多维度时间序列
    multi_dim_time_series = np.stack(time_series_list, axis=1)
    # 数据标准化
    normalized_time_series, scaler = normalize_data([multi_dim_time_series], config.feature_range)
    normalized_time_series = normalized_time_series[0]
    normalization_params = scaler
    # 切片处理
    slices = slice_time_series(normalized_time_series, config.slide_win, config.slide_stride)
    # 存储所有切片的周期
    all_slices_periods = []
    # 处理所有切片 - 进行傅里叶变换
    for slice_idx, slice_data in enumerate(slices):
        _, _, _, top3_periods = process_multi_dim_time_series_slice(
            slice_data, slice_idx, config.slide_win, config)
        all_slices_periods.append(top3_periods)
    # 生成邻接矩阵和节点特征矩阵
    print("\n生成图结构数据...")
    all_adj_matrices, all_node_features = generate_graph_matrices([slices], [all_slices_periods], config)
    print(f"生成了 {len(all_adj_matrices)} 个图结构数据")
    # 打印邻接矩阵和特征矩阵的维度
    print(f"邻接矩阵的维度: {all_adj_matrices.shape}")
    print(f"特征矩阵的维度: {all_node_features.shape}")
    # 创建数据集和数据加载器
    dataset = GraphDataset(all_adj_matrices, all_node_features)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # 创建模型
    input_dim = all_node_features[0].shape[1]  # 特征维度
    num_nodes = all_node_features[0].shape[0]  # 节点数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphVAE(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        num_nodes=num_nodes,
        num_timesteps=config.num_timesteps,
        nblocks=config.nblocks,
        nunits=config.nunits
    ).to(device)
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    # 训练模型
    print("\n开始训练模型...")
    train(model, dataloader, optimizer, config.epochs, config, scheduler)
    # 生成样本
    print("\n生成合成数据...")
    num_samples = len(dataset)
    generated_samples = sample(model, num_samples, config.embed_dim, num_nodes, input_dim, config)
    generated_samples = generated_samples.cpu().numpy()

    # 反归一化原始数据和生成数据
    original_data = np.array([sample.cpu().numpy() for _, sample in dataset])
    original_data = (original_data + 1) * 0.5
    generated_samples = (generated_samples + 1) * 0.5

    # 保存反归一化后的原始数据和生成数据
    original_data_path = os.path.join(config.output_dir, "original_data.npy")
    generated_data_path = os.path.join(config.output_dir, "generated_data.npy")
    np.save(original_data_path, original_data)
    np.save(generated_data_path, generated_samples)
    print(f"反归一化后的原始数据已保存至: {original_data_path}")
    print(f"反归一化后的生成数据已保存至: {generated_data_path}")

    # 可视化
    print("\n生成可视化图表...")
    visualization_dir = os.path.join(config.output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    visualization(original_data, generated_samples, 'pca', save_path=visualization_dir)
    visualization(original_data, generated_samples, 'tsne', save_path=visualization_dir)
    visualization(original_data, generated_samples, 'kernel', save_path=visualization_dir)

    # === 计算评价指标 ===
    print("\n计算评价指标...")
    # 确保数据格式正确
    original_data_2d = original_data.reshape(original_data.shape[0], -1)
    generated_data_2d = generated_samples.reshape(generated_samples.shape[0], -1)
    # 1. Discriminative Score
    try:
        discriminative_score = calculate_discriminative_score(original_data, generated_samples)
        print(f"Discriminative Score: {discriminative_score:.6f}")
    except Exception as e:
        print(f"计算Discriminative Score时出错: {e}")
        discriminative_score = float('inf')
    # 2. Predictive Score
    try:
        predictive_score = calculate_predictive_score(original_data, generated_samples)
        print(f"Predictive Score (MAE): {predictive_score:.6f}")
    except Exception as e:
        print(f"计算Predictive Score时出错: {e}")
        predictive_score = float('inf')
    # 3. Correlational Score
    try:
        correlational_score = calculate_correlational_score(original_data, generated_samples)
        print(f"Correlational Score: {correlational_score:.6f}")
    except Exception as e:
        print(f"计算Correlational Score时出错: {e}")
        correlational_score = float('inf')
    # 保存指标结果
    metrics_path = os.path.join(config.output_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Discriminative Score: {discriminative_score:.6f}\n")
        f.write(f"Predictive Score (MAE): {predictive_score:.6f}\n")
        f.write(f"Correlational Score: {correlational_score:.6f}\n")

    print(f"\n所有处理完成!")
    print(f"结果保存在: {config.output_dir}")
    print(f"评价指标已保存至: {metrics_path}")

if __name__ == "__main__":
    main()