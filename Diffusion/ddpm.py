import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数设置
IMAGE_SIZE = 16
BATCH_SIZE = 128
T = 100  # 扩散步数
BETA_MIN = 0.0001
BETA_MAX = 0.02
LEARNING_RATE = 2e-4
EPOCHS = 10

# 创建保存图像的文件夹
os.makedirs('diffusion_images', exist_ok=True)

# 计算扩散过程的参数，alpha是每一步保留图像信息的比例，beta是每一步添加噪声的比例
betas = torch.linspace(BETA_MIN, BETA_MAX, T, device=device)  # 构建长度为T的等差数列
alphas = 1. - betas  # 计算每一步alpha值
alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算每一步alpha的累计连乘积
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 对累计连乘积开方
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)  # 对1-累计连乘积开方

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

# 修复的噪声预测模型 - 确保通道数匹配
class FixedNoisePredictor(nn.Module):
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
        
        # 输出层
        self.output = nn.Conv2d(16, 1, kernel_size=1)
        
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
        
        return self.output(x)  # [B, 1, 16, 16]

# 实例化模型
model = FixedNoisePredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

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
    noise = torch.randn_like(x0)  # 生成与x0形状相同的随机噪声
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # 根据时间步长t,从预先计算好的累计连乘积开方中取出对应的值，并reshap成[b,1,1,1]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)  # 从(1-累计连乘积开方)中取出对应的值
    xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise  # 生成加噪后的图像xt
    return xt, noise  

# 可视化前向扩散过程
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

# 训练模型
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
            
            # 预测噪声
            predicted_noise = model(noisy_images, t.float() / T)
            
            # 计算损失
            loss = criterion(predicted_noise, true_noise)
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
        torch.save(model.state_dict(), f'diffusion_images/diffusion_model_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'diffusion_images/diffusion_model_final.pth')
    print("Model saved")

# 反向扩散过程 (去噪生成)
def reverse_diffusion(model, num_images=8):
    model.eval()  # 评估模式，关闭dropout、BN等训练特性
    with torch.no_grad():  # 该代码块内不计算梯度
        # 从随机噪声开始
        x = torch.randn(num_images, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)  # 初始化，生成num_images张形状为[1,image_size,image_size]的高斯噪声图像，作为扩散的起点
        
        # 存储生成过程
        gen_steps = []
        
        # 逐步去噪
        for t in tqdm(range(T-1, -1, -1), desc="Generating"):  # 反向去噪，从T~1步
            t_tensor = torch.tensor([t] * num_images, device=device)  # 生成[b,1,1,1]的t_tensor用于输入模型
            
            # 预测噪声
            predicted_noise = model(x, t_tensor.float() / T)  # 用模式预测当前图像x在第t步的噪声（时间归一化到[0,1]）
            
            # 计算系数
            alpha_t = alphas[t].view(-1, 1, 1, 1)
            beta_t = betas[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            
            # 噪声项（高斯噪声），保证采样的多样性，最后一步(t=0)不加噪声
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
                
            ## 反向去噪的核心公式，逐步去噪
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * predicted_noise  # 去除模型预测的噪声成分
            ) + torch.sqrt(beta_t) * z  # 加上采样噪声，保证多样性
            
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
    plt.savefig('diffusion_images/reverse_diffusion.png')
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
    model.load_state_dict(torch.load('diffusion_images/diffusion_model_final.pth', map_location=device))
    
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
    print("Diffusion process demonstration complete!")