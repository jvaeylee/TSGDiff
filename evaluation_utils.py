import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 计算Discriminative Score
def calculate_discriminative_score(original_data, generated_data, test_size=0.2, random_state=42):
    """
    计算Discriminative Score，使用逻辑回归模型区分原始数据和生成数据
    分数计算为|accuracy - 0.5|，值越低越好
    """
    # 展平数据
    n_original = original_data.shape[0]
    n_generated = generated_data.shape[0]
    original_flat = original_data.reshape(n_original, -1)
    generated_flat = generated_data.reshape(n_generated, -1)
    # 合并数据并创建标签
    X = np.vstack([original_flat, generated_flat])
    y = np.concatenate([np.zeros(n_original), np.ones(n_generated)])
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # 评估模型
    accuracy = model.score(X_test, y_test)
    discriminative_score = abs(accuracy - 0.5)
    return discriminative_score

# 计算Predictive Score
def calculate_predictive_score(original_data, generated_data, horizon=1, random_state=42):
    """
    计算Predictive Score，使用GRU模型预测下一时间步
    分数为平均绝对误差(MAE)，值越低越好
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置超参数
    batch_size = 64
    epochs = 20
    learning_rate = 0.001

    # 定义GRU模型
    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
            super(GRUModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    # 准备原始数据的训练集和测试集
    n_samples, seq_len, n_features = original_data.shape
    X_original = original_data[:, :-1, :]
    y_original = original_data[:, -1, :]
    # 准备生成数据的测试集
    X_generated = generated_data[:, :-1, :]
    y_generated = generated_data[:, -1, :]
    # 合并原始数据的训练集和测试集
    X_combined = np.vstack([X_original, X_generated])
    y_combined = np.vstack([y_original, y_generated])
    # 划分训练集和测试集
    indices = np.arange(X_combined.shape[0])
    np.random.seed(random_state)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:int(0.8 * len(indices))], indices[int(0.8 * len(indices)):]
    X_train, y_train = X_combined[train_indices], y_combined[train_indices]
    X_test, y_test = X_combined[test_indices], y_combined[test_indices]
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 初始化模型
    input_dim = n_features
    hidden_dim = 64
    output_dim = n_features
    model = GRUModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mae = mean_absolute_error(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy())
    return mae

# 计算Correlational Score
def calculate_correlational_score(original_data, generated_data):
    """
    计算Correlational Score，评估时间序列的时间依赖性
    分数为原始数据和生成数据的互相关矩阵差异，值越低越好
    """

    def compute_correlation_matrix(data):
        """计算时间序列的互相关矩阵"""
        n_samples, seq_len, n_features = data.shape
        corr_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                # 计算特征i和j的协方差
                cov = np.cov(data[:, :, i].flatten(), data[:, :, j].flatten())[0, 1]
                # 计算标准差
                std_i = np.std(data[:, :, i].flatten())
                std_j = np.std(data[:, :, j].flatten())
                # 计算相关系数
                if std_i * std_j > 0:
                    corr_matrix[i, j] = cov / (std_i * std_j)
                else:
                    corr_matrix[i, j] = 0
        return corr_matrix

    # 计算原始数据和生成数据的相关矩阵
    original_corr = compute_correlation_matrix(original_data)
    generated_corr = compute_correlation_matrix(generated_data)
    # 计算差异
    diff = np.abs(original_corr - generated_corr)
    # 计算平均差异作为分数
    correlational_score = np.mean(diff)
    return correlational_score