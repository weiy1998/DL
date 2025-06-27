import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import math

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数设置
IMAGE_SIZE = 16
BATCH_SIZE = 128
T = 100  # 扩散步数
LEARNING_RATE = 2e-4
EPOCHS = 10
T_START_VLB = 10  # 开始计算VLB损失的时间步
LAMBDA_VLB = 0.001  # VLB损失的权重

# 创建保存图像的文件夹
os.makedirs('diffusion_images', exist_ok=True)

# ========== iddpm使用余弦调度代替ddpm线性调度 ==========
def cosine_beta_schedule(timesteps, s=0.008):
    """ 余弦beta调度
        timesteps表示扩散过程步数T
        s是一个微小的偏移量，防止数据不稳定
        返回一个长度为timestep的tensor,表示每一步的噪声强度beta,
        好处：前期和后期变化较慢中间变化较快，有助于模型生成更高质量的样本
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化，使第一个时间步=1，保证初始时刻无噪声
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

# 计算扩散过程的参数
betas = cosine_beta_schedule(T).to(device)  # 使用余弦调度

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])  # 用于VLB损失
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 加载并预处理MNIST数据集
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到[-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

# 使用单进程数据加载器避免Windows问题
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ========== 改进的噪声预测模型 - 输出两个通道（噪声和方差） ==========
class ImprovedNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 时间嵌入层 - 输出通道数匹配特征图通道数
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),  # 输出128通道，匹配特征图
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 卷积层 - 保持特征图尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 中间层
        self.mid_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # 输出层：输出两个通道（噪声预测和方差预测）
        self.output = nn.Conv2d(16, 2, kernel_size=1)
        
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embed(t.view(-1, 1))  # [B, 128]
        
        # 下采样路径
        x1 = self.conv1(x)  # [B, 32, 16, 16]
        x2 = self.conv2(x1) # [B, 64, 16, 16]
        x3 = self.conv3(x2) # [B, 128, 16, 16]
        
        # 中间层
        x_mid = self.mid_layer(x3) # [B, 128, 16, 16]
        
        # 添加时间嵌入 - 确保通道数匹配
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1)  # [B, 128, 1, 1]
        t_emb = t_emb.expand(-1, -1, x_mid.size(2), x_mid.size(3))  # [B, 128, 16, 16]
        x_mid = x_mid + t_emb  # 现在通道数都是128，可以相加
        
        # 上采样路径
        x = torch.cat([x_mid, x3], dim=1)  # [B, 256, 16, 16]
        x = self.up1(x)  # [B, 64, 16, 16]
        
        x = torch.cat([x, x2], dim=1)  # [B, 128, 16, 16]
        x = self.up2(x)  # [B, 32, 16, 16]
        
        x = torch.cat([x, x1], dim=1)  # [B, 64, 16, 16]
        x = self.up3(x)  # [B, 16, 16, 16]
        
        return self.output(x)  # [B, 2, 16, 16]

# 实例化模型 - 使用改进的模型
model = ImprovedNoisePredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 打印模型结构
print("Model architecture:")
print(model)

# 验证模型输入输出
test_input = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
test_t = torch.tensor([0.5, 0.3], device=device)
test_output = model(test_input, test_t)
print(f"Test input shape: {test_input.shape}")
print(f"Test output shape: {test_output.shape}")

# 前向扩散过程 (添加噪声)
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return xt, noise

# 可视化前向扩散过程 - 保持不变
def visualize_forward_diffusion():
    # 获取一批数据
    single_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=5, shuffle=True, num_workers=0)
    
    data, labels = next(iter(single_loader))
    x0 = data.to(device)
    
    plt.figure(figsize=(15, 3))
    plt.suptitle("Forward Diffusion Process (Adding Noise)", fontsize=16)
    
    # 展示不同时间步的图像
    time_steps = [0, T//4, T//2, 3*T//4, T-1]
    for i, t in enumerate(time_steps):
        t_tensor = torch.tensor([t] * x0.size(0), device=device)
        xt, _ = forward_diffusion(x0, t_tensor)
        
        np_imgs = xt.detach().cpu().numpy().squeeze()
        for j in range(5):
            plt.subplot(5, len(time_steps), j * len(time_steps) + i + 1)
            plt.imshow(np_imgs[j], cmap='gray', vmin=-1, vmax=1)
            plt.axis('off')
            if j == 0:
                plt.title(f"t={t}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('diffusion_images/forward_diffusion.png')
    plt.show()

# ========== 混合损失函数（简单损失 + VLB损失） ==========
def hybrid_loss(pred, target, x_t, x_0, t, alphas_cumprod, alphas_cumprod_prev, betas, lambda_vlb=LAMBDA_VLB):
    """
    IDDPM混合损失函数
    pred: 模型输出 [B, 2, H, W]
    target: 真实噪声 [B, 1, H, W]
    x_t: 噪声图像 [B, C, H, W]
    x_0: 原始图像 [B, C, H, W]
    t: 时间步 [B]
    """
    # 分解模型输出
    pred_noise = pred[:, 0:1, :, :]  # 噪声预测
    pred_v = pred[:, 1:2, :, :]      # 方差预测
    
    # 简单损失 (噪声预测)
    loss_simple = nn.functional.mse_loss(pred_noise, target)
    
    # 变分下界损失 (仅当 t < T_START_VLB 时计算)
    if t.max() < T_START_VLB:
        # 计算真实后验分布参数
        sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
        
        # 后验均值
        posterior_mean = (1.0 / torch.sqrt(alphas[t]).view(-1, 1, 1, 1)) * (
            x_t - (betas[t].view(-1, 1, 1, 1) / sqrt_one_minus_alpha_cumprod_t) * target
        )
        
        # 后验方差
        posterior_variance = (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t]) * betas[t]
        posterior_variance = posterior_variance.view(-1, 1, 1, 1)
        
        # 模型预测的均值
        pred_mean = (1.0 / torch.sqrt(alphas[t]).view(-1, 1, 1, 1)) * (
            x_t - (betas[t].view(-1, 1, 1, 1) / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )
        
        # 模型预测的方差
        log_sigma = pred_v * torch.log(betas[t]).view(-1, 1, 1, 1) + (1 - pred_v) * torch.log(posterior_variance)
        pred_sigma = torch.exp(log_sigma)
        
        # 计算KL散度
        kl = (torch.log(posterior_variance) - log_sigma + 
              (pred_sigma**2 + (pred_mean - posterior_mean)**2) / 
              (2 * posterior_variance) - 0.5)
        
        loss_vlb = kl.mean()
    else:
        loss_vlb = 0.0
    
    return loss_simple + lambda_vlb * loss_vlb

# ========== 训练模型 - 使用混合损失函数 ==========
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, _ in progress_bar:
            images = images.to(device)
            batch_size = images.size(0)
            
            # 随机采样时间步
            t = torch.randint(0, T, (batch_size,), device=device)
            
            # 前向扩散添加噪声
            noisy_images, true_noise = forward_diffusion(images, t)
            
            # 预测噪声和方差
            pred_output = model(noisy_images, t.float() / T)
            
            # 计算混合损失
            loss = hybrid_loss(
                pred_output, 
                true_noise, 
                noisy_images, 
                images, 
                t, 
                alphas_cumprod, 
                alphas_cumprod_prev, 
                betas
            )
            
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        # 每个epoch结束后保存模型
        torch.save(model.state_dict(), f'diffusion_images/iddpm_model_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'diffusion_images/iddpm_model_final.pth')
    print("Model saved")

# ========== 反向扩散过程 - 使用学习到的方差 ==========
def reverse_diffusion(model, num_images=8):
    model.eval()
    with torch.no_grad():
        # 从随机噪声开始
        x = torch.randn(num_images, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        
        # 存储生成过程
        gen_steps = []
        
        # 逐步去噪
        for t in tqdm(range(T-1, -1, -1), desc="Generating"):
            t_tensor = torch.tensor([t] * num_images, device=device)
            
            # 预测噪声和方差
            pred_output = model(x, t_tensor.float() / T)
            predicted_noise = pred_output[:, 0:1, :, :]  # 噪声预测
            pred_v = pred_output[:, 1:2, :, :]          # 方差预测
            
            # 计算系数
            alpha_t = alphas[t].view(-1, 1, 1, 1)
            beta_t = betas[t].view(-1, 1, 1, 1)
            alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_cumprod_t_prev = alphas_cumprod_prev[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            
            # 计算预测的原始图像x0
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # 确保值在合理范围内
            
            # 计算混合方差
            beta_t_hat = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * beta_t
            log_sigma = pred_v * torch.log(beta_t) + (1 - pred_v) * torch.log(beta_t_hat)
            sigma = torch.exp(log_sigma)
            
            # 计算均值
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            ## 去噪核心 更新公式 - 使用学习到的方差
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
            ) + sigma * z
            
            # 存储中间结果
            if t % (T//5) == 0 or t == 0:
                gen_steps.append(x.cpu().numpy())
        
        return gen_steps

# 可视化反向生成过程
def visualize_reverse_diffusion(gen_steps):
    plt.figure(figsize=(15, 8))
    plt.suptitle("Reverse Diffusion Process (Denoising)", fontsize=16)
    
    num_steps = len(gen_steps)
    num_images = gen_steps[0].shape[0]
    
    for i in range(num_images):
        for j, step in enumerate(gen_steps):
            plt_idx = i * num_steps + j + 1
            plt.subplot(num_images, num_steps, plt_idx)
            img = step[i].squeeze()
            plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
            plt.axis('off')
            if i == 0:
                t_val = T - (T//5) * j if j < num_steps-1 else 0
                plt.title(f"t={t_val}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('diffusion_images/reverse_iddpm.png')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 可视化前向扩散过程
    print("Visualizing forward diffusion process...")
    visualize_forward_diffusion()
    
    # 训练模型
    print("\nTraining model...")
    train_model()
    
    # 加载训练好的模型
    model.load_state_dict(torch.load('diffusion_images/iddpm_model_final.pth', map_location=device))
    
    # 生成图像并可视化反向过程
    print("\nGenerating images with reverse diffusion...")
    gen_steps = reverse_diffusion(model, num_images=8)
    visualize_reverse_diffusion(gen_steps)
    
    # 保存最终生成的图像
    final_images = gen_steps[-1]
    plt.figure(figsize=(10, 2))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(final_images[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
        plt.axis('off')
    plt.suptitle("Final Generated Images", fontsize=16)
    plt.tight_layout()
    plt.savefig('diffusion_images/final_generated.png')
    plt.show()
    
    print("All images saved in 'diffusion_images' folder")
    print("Improved DDPM (IDDPM) process demonstration complete!")