import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from PIL import Image
import numpy as np


def calculate_vitscore_for_gray_images(img_path1, img_path2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """计算两张灰度图像的ViTScore语义相似度"""
    # 加载预训练的Vision Transformer模型
    model = vit_b_16(pretrained=True)
    model.heads = nn.Identity()  # 只使用特征提取部分
    model = model.to(device)
    model.eval()

    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图像
    def load_and_preprocess(img_path):
        img = Image.open(img_path).convert('L')  # 转为灰度图
        img_array = np.array(img)

        # 扩展为三通道
        img_rgb = np.stack([img_array] * 3, axis=-1)

        # 转换为PIL图像并应用预处理
        img_pil = Image.fromarray(img_rgb)
        return transform(img_pil).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        features1 = model(load_and_preprocess(img_path1))
        features2 = model(load_and_preprocess(img_path2))

        # 计算余弦相似度
        similarity = nn.functional.cosine_similarity(features1, features2).item()

    return similarity


# 使用示例
img_path1 = r".\1.png"
img_path2 = r".\org_1.png"

similarity = calculate_vitscore_for_gray_images(img_path1, img_path2)
print(f"语义相似度: {similarity:.4f}")
