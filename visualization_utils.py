import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 可视化函数（使用归一化数据）(Visualization function)
def visualization(ori_data, generated_data, analysis, compare=3000, save_path=None, point_size=20):
    """Using PCA or tSNE for normalized data visualization"""
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]
    # 数据预处理（均为归一化后的数据）
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    no, seq_len, dim = ori_data.shape
    # 均值聚合
    prep_data = []
    prep_data_hat = []
    for i in range(anal_sample_no):
        # 沿节点维度取均值，保留时间序列特征
        prep_data.append(np.mean(ori_data[i, :, :], axis=1).reshape(1, seq_len))
        prep_data_hat.append(np.mean(generated_data[i, :, :], axis=1).reshape(1, seq_len))
    prep_data = np.concatenate(prep_data, axis=0)
    prep_data_hat = np.concatenate(prep_data_hat, axis=0)
    # 可视化参数
    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)]
    # PCA可视化
    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        f, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:anal_sample_no],
                    alpha=0.6, label="Original (Normalized)", s=point_size)
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[anal_sample_no:],
                    alpha=0.6, label="Synthetic (Normalized)", s=point_size)
        ax.legend(fontsize=12)
        plt.title('PCA Visualization of Normalized Data', fontsize=14)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        if save_path:
            plt.savefig(f"{save_path}/pca_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
    # t-SNE可视化
    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(prep_data_final)
        f, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.6, label="Original (Normalized)", s=point_size)
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.6, label="Synthetic (Normalized)", s=point_size)
        ax.legend(fontsize=12)
        plt.title('t-SNE Visualization of Normalized Data', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        if save_path:
            plt.savefig(f"{save_path}/tsne_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()
    # 核密度估计可视化
    elif analysis == 'kernel':
        f, ax = plt.subplots(figsize=(12, 6))
        # 展平数据以绘制分布
        prep_data_flat = prep_data.reshape(-1)
        prep_data_hat_flat = prep_data_hat.reshape(-1)
        sns.kdeplot(prep_data_flat, label="Original (Normalized)", color="red",
                    linewidth=2, alpha=0.8)
        sns.kdeplot(prep_data_hat_flat, label="Synthetic (Normalized)", color="blue",
                    linewidth=2, alpha=0.8, linestyle='--')
        plt.legend(fontsize=12)
        plt.title('Kernel Density Estimation of Normalized Data', fontsize=14)
        plt.xlabel('Data Value', fontsize=12)
        plt.ylabel('Density Estimate', fontsize=12)
        if save_path:
            plt.savefig(f"{save_path}/kernel_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()