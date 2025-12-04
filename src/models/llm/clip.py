import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from curl_cffi import requests
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from transformers.utils import ModelOutput
from typing import Sequence, List, Tuple, Type
from pathlib import Path
from dataclasses import dataclass

from src.config import get_config_value
from src.constants import PRETRAINED_MODEL_DIR


@dataclass
class CLIPForwardOutput(ModelOutput):
    loss: float | torch.Tensor | None = None
    text_loss: float | torch.Tensor | None = None
    image_loss: float | torch.Tensor | None = None
    logits_per_text: torch.Tensor | None = None
    logits_per_image: torch.Tensor | None = None
    text_embeds: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None

class CLIP(nn.Module):
    def __init__(self, model_name: str="openai/clip-vit-base-patch32", cache_dir=PRETRAINED_MODEL_DIR, dtype=torch.float16):
        super().__init__()
        self.device = get_config_value("device", default="cuda")

        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=dtype)

        self.model.to(self.device)

    def encode(self, texts: Sequence[str]|None=None, images: Sequence[Image.Image|str|Path|np.ndarray]|None=None):
        if texts is None and images is None:
            raise ValueError("texts and images cannot be both None")
        
        if images is not None:
            images = [self.get_image(image) for image in images]

        model_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            text_feats = outputs.text_embeds
            image_feats = outputs.image_embeds
        
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

        return {
            "text": text_feats,
            "image": image_feats,
            "logits": (outputs.logits_per_image, outputs.logits_per_text),
            "rank": (outputs.logits_per_text.softmax(dim=-1), outputs.logits_per_image.softmax(dim=-1)),
        }

    def forward(self, texts: Sequence[str], images: Sequence[Image.Image|str|Path|np.ndarray], temperature=0.07):
        # 确保模型处于训练模式
        self.model.train()
        
        # 预处理图像
        images = [self.get_image(image) for image in images]
        
        # 处理输入数据
        model_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # 前向传播
        outputs = self.model(**model_inputs)
        
        # 获取文本和图像特征
        text_embeds = outputs.text_embeds  # [batch_size, embed_dim]
        image_embeds = outputs.image_embeds  # [batch_size, embed_dim]
        
        # 归一化特征向量
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵
        logits_per_text = text_embeds @ image_embeds.T / temperature
        logits_per_image = logits_per_text.T
        
        # 创建标签（对角线为正样本）
        batch_size = text_embeds.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算对比损失
        text_loss  = F.cross_entropy(logits_per_text, labels)
        image_loss = F.cross_entropy(logits_per_image, labels)
        total_loss = (text_loss + image_loss) / 2
        
        return CLIPForwardOutput(
            loss=total_loss.item(),
            text_loss=text_loss.item(),
            image_loss=image_loss.item(),
            logits_per_text=logits_per_text.detach(),
            logits_per_image=logits_per_image.detach(),
            text_embeds=text_embeds.detach(),
            image_embeds=image_embeds.detach(),
        )
    
    def set_train_mode(self, mode=True):
        """设置模型的训练/评估模式"""
        self.model.train(mode)
        return self
    
    def set_eval_mode(self):
        """设置模型为评估模式"""
        self.model.eval()
        return self
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return self.model.parameters()
    
    def save_model(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    def load_model(self, model_path: str):
        """加载模型"""
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        return self
    
    def train_epoch(self, dataloader, optimizer, temperature=0.07, log_interval=100):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器，应该返回(texts, images)元组
            optimizer: 优化器
            temperature: 对比学习温度参数
            log_interval: 日志打印间隔
            
        Returns:
            dict: 包含平均损失的字典
        """
        self.set_train_mode()
        total_loss = 0.0
        total_text_loss = 0.0
        total_image_loss = 0.0
        num_batches = 0
        
        for batch_idx, (texts, images) in enumerate(dataloader):
            # 训练单个批次
            results = self.train(texts, images, optimizer, temperature)
            
            total_loss += results['loss']
            total_text_loss += results['text_loss']
            total_image_loss += results['image_loss']
            num_batches += 1
            
            # 打印训练日志
            if batch_idx % log_interval == 0:
                print(f'Batch {batch_idx}: Loss={results["loss"]:.4f}, '
                      f'Text Loss={results["text_loss"]:.4f}, '
                      f'Image Loss={results["image_loss"]:.4f}')
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_text_loss = total_text_loss / num_batches
        avg_image_loss = total_image_loss / num_batches
        
        return {
            'avg_loss': avg_loss,
            'avg_text_loss': avg_text_loss,
            'avg_image_loss': avg_image_loss,
            'num_batches': num_batches
        }

    def get_image(self, image):
        if isinstance(image, str):
            if image.startswith("http"):
                image = requests.get(image).raw
            else:
                image = Image.open(image).convert("RGB")
        elif isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image
