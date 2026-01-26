import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 优化后的训练函数(Optimized training function)
def train(model, dataloader, optimizer, epochs, config, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_diffusion_loss = 0
        total_fourier_loss = 0
        batch_count = 0
        # 使用固定的KL权重
        kl_weight = config.kl_weight
        # 使用扩散损失权重
        diffusion_weight = config.diffusion_weight
        # 使用Fourier损失权重
        ff_weight = config.ff_weight
        for adj_matrix, node_feature in dataloader:
            adj_matrix = adj_matrix.to(device)
            node_feature = node_feature.to(device)
            optimizer.zero_grad()
            recon_x, mu, logstd, predicted_z, z = model(node_feature, adj_matrix)
            # 计算重构损失
            recon_loss = torch.nn.MSELoss()(recon_x, node_feature)
            # 计算KL散度损失（使用均值而非求和）
            kl_loss = -0.5 * torch.mean(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd))
            # 计算Diffusion损失
            diffusion_loss = torch.nn.MSELoss()(predicted_z, z)
            # 计算Fourier损失
            fft1 = torch.fft.fft(recon_x.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(node_feature.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = torch.nn.MSELoss()(torch.real(fft1), torch.real(fft2)) + torch.nn.MSELoss()(torch.imag(fft1),
                                                                                           torch.imag(fft2))
            # 总损失（使用权重调整各部分贡献）
            loss = recon_loss + kl_weight * kl_loss + diffusion_weight * diffusion_loss + ff_weight * fourier_loss
            # 检查损失是否为NaN
            if torch.isnan(loss).any():
                print("警告: 损失计算结果为NaN，跳过此次更新")
                continue
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_fourier_loss += fourier_loss.item()
            batch_count += 1
        # 更新学习率调度器
        if scheduler and batch_count > 0:
            scheduler.step(total_loss / batch_count)
        # 打印详细损失信息
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_recon_loss = total_recon_loss / batch_count
            avg_kl_loss = total_kl_loss / batch_count
            avg_diffusion_loss = total_diffusion_loss / batch_count
            avg_fourier_loss = total_fourier_loss / batch_count
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs}, '
                  f'Loss: {avg_loss:.6f}, '
                  f'Recon: {avg_recon_loss:.6f}, '
                  f'KL: {avg_kl_loss:.6f}, '
                  f'Diffusion: {avg_diffusion_loss:.6f}, '
                  f'Fourier: {avg_fourier_loss:.6f}, '
                  f'KLW: {kl_weight:.6f}, '
                  f'DiffW: {diffusion_weight:.6f}, '
                  f'FFW: {ff_weight:.6f}, '
                  f'LR: {current_lr:.8f}')
        else:
            print(f'Epoch {epoch + 1}/{epochs}, 所有批次均因NaN损失被跳过')

# 采样函数(Sampling function)
def sample(model, num_samples, embed_dim, num_nodes, output_dim, config):
    """生成归一化的样本（不进行反归一化）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, embed_dim).to(device)  # 标准正态分布采样
        # 反向Diffusion过程
        for t in reversed(range(0, config.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_z = model.diffusion_model(z, t_batch)
            alpha_t = model.alphas[t]
            alpha_t_cumprod = model.alphas_cumprod[t]
            alpha_t_prev = model.alphas_cumprod[t - 1] if t > 0 else 1.0
            beta_t = model.betas[t]
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_t_cumprod)
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
            posterior_variance = beta_t * (1. - alpha_t_prev) / (1. - alpha_t_cumprod)
            z = sqrt_recip_alpha_t * (z - (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t * (z - predicted_z))
            if t > 0:
                noise = torch.randn_like(z)
                z += torch.sqrt(posterior_variance) * noise
        samples = model.decoder(z)  # 直接输出归一化结果
    return samples