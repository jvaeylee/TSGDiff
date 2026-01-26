import math

class Config:
    def __init__(self):
        # 文件路径配置(File path configuration)
        self.data_file = './datasets/ETTh.csv'
        self.output_dir = './output'  # 输出目录(Output directory)
        # 切片参数(Patch parameters)
        self.slide_win = 48  # 滑动窗口大小(Sliding window size)
        self.slide_stride = 1  # 滑动步长(Sliding step)
        # 峰值检测参数(Peak detection parameters)
        self.peak_height_coeff = 0.01  # 峰值高度系数（相对于标准差）(Peak height factor)
        self.peak_dist_coeff = 0.01  # 峰值最小距离系数（相对于频率分辨率）(Peak minimum distance coefficient)
        self.near_zero_threshold = 0  # 频率接近零的阈值(Threshold value of frequency close to zero)
        # 图神经网络参数(GNN parameters)
        self.hidden_dim = 1600  # 编码器隐藏层维度(Encoder hidden layer dimension)
        self.embed_dim = 1600  # 编码器嵌入维度(Encoder embedding dimension)
        # 使用第几个周期生成边（Generate edges using the number of cycles）
        self.use_period_index = 2
        # GraphVAE 参数(GraphVAE parameters)
        self.batch_size = 128
        self.epochs = 500
        self.learning_rate = 0.01

        # 数据标准化参数(Data standardization parameters)
        self.normalize_data = True
        self.feature_range = (-1, 1)  # 标准化范围
        # 固定KL权重(KL weight)
        self.kl_weight = 0.2
        # Diffusion损失权重(Denoising loss weight)
        self.diffusion_weight = 1.0  # 扩散模型损失权重
        # Diffusion Model 参数(Diffusion Model parameters)
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        # Diffusion Block参数(Diffusion Block parameters)
        self.nblocks = 3  # 中间块数量(Number of intermediate blocks)
        self.nunits = 64  # 每个块的单元数量(Number of cells per block)
        # 新增参数k(New parameter k)
        self.k = 2
        # Fourier损失权重(Fourier loss weight)
        self.ff_weight = math.sqrt(self.slide_win) / 5