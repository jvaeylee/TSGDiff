import torch
import torch.nn as nn
import torch.nn.init as init

# Mish激活函数(Mish activation function)
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

# 改进的带残差连接的图卷积块(Improved graph convolution block with residual connection)
class ImprovedGraphConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedGraphConvBlock, self).__init__()
        # 第一个图卷积层
        self.conv1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.activation1 = Mish()
        # 第二个图卷积层
        self.conv2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.activation2 = Mish()
        # 捷径连接
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, adj):
        # 第一个图卷积层
        y = torch.matmul(adj, x)
        y = self.conv1(y)
        y = self.bn1(y.transpose(1, 2)).transpose(1, 2)
        y = self.activation1(y)
        # 第二个图卷积层
        y = torch.matmul(adj, y)
        y = self.conv2(y)
        y = self.bn2(y.transpose(1, 2)).transpose(1, 2)
        y = self.activation2(y)
        # 捷径连接
        shortcut = self.shortcut(x)
        # 残差连接
        return y + shortcut

# 图编码器(Graph Encoder)
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = ImprovedGraphConvBlock(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = ImprovedGraphConvBlock(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = ImprovedGraphConvBlock(input_dim=hidden_dim, output_dim=embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.linear_mu = nn.Linear(embed_dim, embed_dim)
        self.linear_logstd = nn.Linear(embed_dim, embed_dim)  # 输出对数标准差
        self.mish = Mish()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.mish(x)
        x = self.conv2(x, adj)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.mish(x)
        x = self.conv3(x, adj)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        # 使用均值池化聚合节点信息
        x = torch.mean(x, 1)  # [batch_size, embed_dim]
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)  # 输出对数标准差
        logstd = torch.clamp(logstd, min=-10, max=2)
        return mu, logstd

# 图解码器(Graph Decoder)
class GraphDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, num_nodes):
        super(GraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_nodes * output_dim)
        self.mish = Mish()
        self.tanh = nn.Tanh()  # 用于将输出限制在[-1, 1]范围内

    def forward(self, z):
        z = self.linear1(z)
        z = self.bn1(z)
        z = self.mish(z)
        z = self.linear2(z)
        z = self.bn2(z)
        z = self.mish(z)
        z = self.linear3(z)
        z = self.bn3(z)
        z = self.mish(z)
        z = self.linear4(z)
        # 重塑为 [batch_size, num_nodes, output_dim]
        z = z.view(z.size(0), self.num_nodes, self.output_dim)
        z = self.tanh(z)  # 添加激活函数，确保输出在合理范围内
        return z

# Diffusion Block
class DiffusionBlock(nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x

# Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nblocks: int = 2, nunits: int = 64):
        super(DiffusionModel, self).__init__()
        self.inblock = nn.Linear(nfeatures + 1, nunits)  # 输入层：拼接特征+时间步
        self.midblocks = nn.ModuleList([DiffusionBlock(nunits) for _ in range(nblocks)])  # 中间块
        self.outblock = nn.Linear(nunits, nfeatures)  # 输出层

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 将时间步t转换为与x同形状的列向量并拼接
        t = t.unsqueeze(1)  # 从[batch_size]变为[batch_size, 1]
        val = torch.hstack([x, t])  # 拼接特征和时间步
        val = self.inblock(val)  # 输入层处理
        for midblock in self.midblocks:  # 中间块处理
            val = midblock(val)
        val = self.outblock(val)  # 输出层处理
        return val

# GraphVAE模型(GraphVAE Model)
class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_nodes, num_timesteps, nblocks, nunits):
        super(GraphVAE, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, embed_dim)
        self.diffusion_model = DiffusionModel(
            nfeatures=embed_dim,
            nblocks=nblocks,
            nunits=nunits
        )
        self.decoder = GraphDecoder(embed_dim, hidden_dim, input_dim, num_nodes)
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)  # 从对数标准差恢复标准差
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, adj):
        mu, logstd = self.encoder(x, adj)
        z = self.reparameterize(mu, logstd)
        # Diffusion过程
        t = torch.randint(0, self.num_timesteps, (z.shape[0],))
        # 将 t 移动到和 z 相同的设备上
        t = t.to(z.device)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t]).unsqueeze(-1)
        # 将 sqrt_alphas_cumprod_t 和 sqrt_one_minus_alphas_cumprod_t 移动到和 z 相同的设备上
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(z.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(z.device)
        eps = torch.randn_like(z)
        z_t = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * eps
        # 预测原始数据
        predicted_z = self.diffusion_model(z_t, t)
        # 重构z
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_t_cumprod = self.alphas_cumprod[t].unsqueeze(-1)
        # 将 alpha_t 和 alpha_t_cumprod 移动到和 z 相同的设备上
        alpha_t = alpha_t.to(z.device)
        alpha_t_cumprod = alpha_t_cumprod.to(z.device)
        z_recon = (z_t - torch.sqrt(1 - alpha_t_cumprod) * eps) / torch.sqrt(alpha_t_cumprod)
        recon_x = self.decoder(z_recon)
        return recon_x, mu, logstd, predicted_z, z