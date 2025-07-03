import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt

# set divece
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def preprocess_image(image, target_size=512):
    """预处理图像：调整大小、转换为张量、归一化"""
    # 创建预处理转换管道
    preprocess = transforms.Compose([
        transforms.Resize(target_size),  # 调整图像大小
        transforms.CenterCrop(target_size),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量 [0, 1]
        transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
    ])
    # 添加批次维度并返回
    return preprocess(image).unsqueeze(0).to(device)

def postprocess_image(image_tensor):
    """将张量转换回PIL图像"""
    # 将张量从GPU移到CPU（如果需要）并转换为numpy数组
    image = image_tensor.detach().cpu()
    
    # 反归一化: [-1, 1] -> [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # 转换为numpy数组并调整维度顺序 (C, H, W) -> (H, W, C)
    image = image.permute(0, 2, 3, 1).numpy()
    
    # 转换为uint8并创建PIL图像
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])

def vae_reconstruct(vae, image_tensor):
    """
    使用VAE进行图像重建：
    1. 将图像编码为潜变量
    2. 将潜变量解码回图像
    """
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 编码图像为潜变量
        latent = vae.encode(image_tensor).latent_dist.sample()
        
        # 解码潜变量回图像
        reconstructed = vae.decode(latent).sample
    
    return reconstructed

def compare_reconstructions(original, reconstructions, titles):
    """可视化原始图像和不同VAE的重建结果"""
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, len(reconstructions) + 1, 1)
    plt.imshow(original)
    plt.title("raw image")
    plt.axis('off')
    
    # 显示每个重建结果
    for i, (img, title) in enumerate(zip(reconstructions, titles)):
        plt.subplot(1, len(reconstructions) + 1, i + 2)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("vae_comparison.png", bbox_inches='tight')
    plt.show()

def main():
    
    # PIL读取图像
    image_path = "/home/yanwei/code/Diffusion/images/cat.png"
    original_image = Image.open(image_path).convert("RGB")
    
    # 使用cv2读取图像
    # raw_image = cv2.imread(image_path)
    # original_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # 预处理图像
    image_tensor = preprocess_image(original_image).half()  # 半精.half()，全精.float()
    
    # 定义要测试的不同VAE模型
    vae_models = {
        "SD-v1.5": "/home/yanwei/models/stable-diffusion-v1-5",  # 原始Stable Diffusion V1.5的VAE
        # "SD-vae-ft-mse": "stabilityai/sd-vae-ft-mse",  # 微调的VAE，重建质量更好
        # "SD-vae-ft-ema": "stabilityai/sd-vae-ft-ema",  # 另一种微调版本
        # "Kandinsky-2.2": "kandinsky-community/kandinsky-2-2-decoder"  # 不同架构的VAE
    }
    
    reconstructed_images = []
    reconstruction_titles = []
    
    for name, model_id in vae_models.items():
        print(f"\n正在使用 {name} VAE ({model_id}) 进行重建...")
        
        try:
            # 加载VAE模型
            # 重要：设置torch_dtype=torch.float32以避免半精度错误
            vae = AutoencoderKL.from_pretrained(
                model_id, 
                subfolder="vae",
                torch_dtype=torch.float16  # 半精.float16，全精.float32
            ).to(device)  # 加载到GPU上，半精dtype=torch.float16，全精float32
            
            # 进行图像重建
            reconstructed_tensor = vae_reconstruct(vae, image_tensor)
            
            # 后处理重建图像
            reconstructed_image = postprocess_image(reconstructed_tensor)
            
            # 保存重建结果
            output_path = f"reconstructed_{name.replace('-', '_').lower()}.png"
            reconstructed_image.save(output_path)
            print(f"重建图像已保存至: {output_path}")
            
            # 收集结果用于比较
            reconstructed_images.append(reconstructed_image)
            reconstruction_titles.append(name)
            
            # 释放内存
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"使用 {name} VAE 时出错: {str(e)}")
            print(f"跳过 {name} VAE...")
    
    # 可视化比较结果
    print("\n生成比较图...")
    compare_reconstructions(original_image, reconstructed_images, reconstruction_titles)
    print("比较图已保存至: vae_comparison.png")

if __name__ == "__main__":
    main()