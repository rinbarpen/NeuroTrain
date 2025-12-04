from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoImageProcessor,
        CLIPVisionModel,
        CLIPImageProcessor,
    )
except ImportError as exc:
    raise ImportError(
        "BertVLM requires transformers library. Please install: pip install transformers"
    ) from exc

from src.constants import PRETRAINED_MODEL_DIR


class BertVLM(nn.Module):
    """
    基于BERT的视觉-语言模型（Vision-Language Model）
    
    使用BERT作为文本编码器，ViT/CLIP视觉编码器作为图像编码器，
    通过多模态融合层实现文本和图像的联合表示学习。
    """

    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        *,
        embed_dim: int = 512,
        projection_dim: Optional[int] = None,
        freeze_text_encoder: bool = False,
        freeze_vision_encoder: bool = False,
        cache_dir: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        use_cross_attention: bool = False,
        cross_attention_layers: int = 2,
        temperature: float = 0.07,
    ) -> None:
        """
        初始化BertVLM模型
        
        Args:
            text_encoder_name: BERT模型名称或路径
            vision_encoder_name: 视觉编码器名称（CLIP ViT或ViT模型）
            embed_dim: 统一的嵌入维度
            projection_dim: 投影层维度（如果为None，则使用embed_dim）
            freeze_text_encoder: 是否冻结文本编码器
            freeze_vision_encoder: 是否冻结视觉编码器
            cache_dir: 模型缓存目录
            device: 设备
            dtype: 数据类型
            use_cross_attention: 是否使用交叉注意力进行多模态融合
            cross_attention_layers: 交叉注意力层数
            temperature: 对比学习温度参数
        """
        super().__init__()
        
        self.device = torch.device(device) if device is not None else self._default_device()
        self.dtype = dtype or torch.float32
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim or embed_dim
        self.temperature = temperature
        self.use_cross_attention = use_cross_attention
        
        cache_dir = cache_dir or PRETRAINED_MODEL_DIR
        
        # 初始化文本编码器（BERT）
        self.text_encoder = AutoModel.from_pretrained(
            text_encoder_name,
            cache_dir=cache_dir,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_name,
            cache_dir=cache_dir,
        )
        
        # 初始化视觉编码器
        if "clip" in vision_encoder_name.lower():
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vision_encoder_name,
                cache_dir=cache_dir,
            )
            self.image_processor = CLIPImageProcessor.from_pretrained(
                vision_encoder_name,
                cache_dir=cache_dir,
            )
        else:
            # 使用其他ViT模型
            self.vision_encoder = AutoModel.from_pretrained(
                vision_encoder_name,
                cache_dir=cache_dir,
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                vision_encoder_name,
                cache_dir=cache_dir,
            )
        
        # 移动到指定设备
        self.text_encoder = self.text_encoder.to(self.device)  # type: ignore
        self.vision_encoder = self.vision_encoder.to(self.device)  # type: ignore
        
        if self.dtype in (torch.float16, torch.bfloat16):
            self.text_encoder = self.text_encoder.to(dtype=self.dtype)  # type: ignore
            self.vision_encoder = self.vision_encoder.to(dtype=self.dtype)  # type: ignore
        
        # 获取编码器输出维度
        text_hidden_size = self.text_encoder.config.hidden_size
        if hasattr(self.vision_encoder.config, "hidden_size"):
            vision_hidden_size = self.vision_encoder.config.hidden_size
        elif hasattr(self.vision_encoder.config, "vision_config"):
            vision_hidden_size = self.vision_encoder.config.vision_config.hidden_size
        else:
            # 默认值
            vision_hidden_size = 768
        
        # 文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.embed_dim),
        )
        
        # 视觉投影层
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.embed_dim),
        )
        
        # 交叉注意力层（可选）
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=8,
                    batch_first=True,
                )
                for _ in range(cross_attention_layers)
            ])
            self.cross_attention_norm = nn.LayerNorm(self.embed_dim)
        
        # 冻结编码器参数
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # 初始化投影层
        self._init_projection_layers()
        
        # 移动到设备并设置数据类型
        self.to(self.device)
        
        # 确保投影层和交叉注意力层使用与编码器相同的数据类型
        if self.dtype in (torch.float16, torch.bfloat16):
            self.text_projection = self.text_projection.to(dtype=self.dtype)  # type: ignore
            self.vision_projection = self.vision_projection.to(dtype=self.dtype)  # type: ignore
            if use_cross_attention:
                self.cross_attention_layers = self.cross_attention_layers.to(dtype=self.dtype)  # type: ignore
                self.cross_attention_norm = self.cross_attention_norm.to(dtype=self.dtype)  # type: ignore
    
    def _default_device(self) -> torch.device:
        """获取默认设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _init_projection_layers(self):
        """初始化投影层"""
        for module in [self.text_projection, self.vision_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def encode_text(
        self,
        texts: Union[str, Sequence[str]],
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """
        编码文本
        
        Args:
            texts: 文本字符串或字符串列表
            return_pooled: 是否返回池化后的特征（[CLS] token）
        
        Returns:
            文本特征张量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded = self.text_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # 编码
        with torch.set_grad_enabled(self.training):
            outputs = self.text_encoder(**encoded, return_dict=True)
            
            if return_pooled:
                # 使用[CLS] token或pooler输出
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    text_features = outputs.pooler_output
                else:
                    text_features = outputs.last_hidden_state[:, 0]  # [CLS] token
            else:
                text_features = outputs.last_hidden_state
        
        # 投影到统一空间
        if return_pooled:
            text_features = self.text_projection(text_features)
        else:
            # 对序列进行投影
            batch_size, seq_len, hidden_dim = text_features.shape
            text_features = text_features.reshape(-1, hidden_dim)
            text_features = self.text_projection(text_features)
            text_features = text_features.reshape(batch_size, seq_len, self.embed_dim)
        
        return text_features
    
    def encode_image(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor, Sequence],
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像（PIL Image, numpy array, torch Tensor或列表）
            return_pooled: 是否返回池化后的特征
        
        Returns:
            图像特征张量
        """
        # 预处理图像
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            images = [images]
        
        # 转换为PIL Image
        processed_images = []
        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    try:
                        from curl_cffi import requests as curl_requests  # type: ignore
                        response = curl_requests.get(img)
                        img = Image.open(response.raw).convert("RGB")  # type: ignore
                    except ImportError:
                        import requests
                        response = requests.get(img, stream=True)
                        img = Image.open(response.raw).convert("RGB")  # type: ignore
                else:
                    img = Image.open(img).convert("RGB")
            elif isinstance(img, Path):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                # 假设已经是处理过的tensor
                processed_images.append(img)
                continue
            processed_images.append(img)
        
        # 使用processor处理
        if isinstance(processed_images[0], torch.Tensor):
            pixel_values = torch.stack(processed_images).to(self.device)
        else:
            pixel_values = self.image_processor(
                processed_images,
                return_tensors="pt",
            )["pixel_values"].to(self.device)
        
        # 编码
        with torch.set_grad_enabled(self.training):
            # 对于CLIPVisionModel，直接调用模型
            if isinstance(self.vision_encoder, CLIPVisionModel):
                # CLIP模型 - 直接调用，不使用return_dict参数
                outputs = self.vision_encoder(pixel_values=pixel_values)
                # CLIPVisionModel返回BaseModelOutputWithPooling
                if return_pooled:
                    # 使用pooler_output或[CLS] token
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    else:
                        # 使用[CLS] token (第一个token)
                        image_features = outputs.last_hidden_state[:, 0]
                else:
                    image_features = outputs.last_hidden_state
            else:
                # 其他ViT模型
                outputs = self.vision_encoder(
                    pixel_values=pixel_values,
                    return_dict=True,
                )
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None and return_pooled:
                    image_features = outputs.pooler_output
                elif return_pooled:
                    # 使用[CLS] token
                    image_features = outputs.last_hidden_state[:, 0]
                else:
                    image_features = outputs.last_hidden_state
        
        # 投影到统一空间
        if return_pooled:
            image_features = self.vision_projection(image_features)
        else:
            batch_size, seq_len, hidden_dim = image_features.shape
            image_features = image_features.reshape(-1, hidden_dim)
            image_features = self.vision_projection(image_features)
            image_features = image_features.reshape(batch_size, seq_len, self.embed_dim)
        
        return image_features
    
    def forward(
        self,
        texts: Optional[Union[str, Sequence[str]]] = None,
        images: Optional[Union[Image.Image, np.ndarray, torch.Tensor, Sequence]] = None,
        return_features: bool = False,
    ) -> dict:
        """
        前向传播
        
        Args:
            texts: 文本输入
            images: 图像输入
            return_features: 是否返回原始特征
        
        Returns:
            包含编码特征的字典
        """
        text_features = None
        image_features = None
        
        if texts is not None:
            text_features = self.encode_text(texts, return_pooled=True)
        
        if images is not None:
            image_features = self.encode_image(images, return_pooled=True)
        
        # 交叉注意力融合（如果启用）
        if self.use_cross_attention and text_features is not None and image_features is not None:
            # 将图像特征作为query，文本特征作为key和value
            image_features_expanded = image_features.unsqueeze(1)  # [B, 1, D]
            text_features_expanded = text_features.unsqueeze(1)  # [B, 1, D]
            
            fused_features = image_features_expanded
            for cross_attn in self.cross_attention_layers:
                fused_features, _ = cross_attn(
                    query=fused_features,
                    key=text_features_expanded,
                    value=text_features_expanded,
                )
                fused_features = self.cross_attention_norm(fused_features + image_features_expanded)
            
            fused_features = fused_features.squeeze(1)  # [B, D]
        else:
            fused_features = None
        
        result = {}
        if text_features is not None:
            result["text_features"] = text_features
        if image_features is not None:
            result["image_features"] = image_features
        if fused_features is not None:
            result["fused_features"] = fused_features
        
        return result
    
    def compute_similarity(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算文本和图像特征的相似度
        
        Args:
            text_features: 文本特征 [B, D]
            image_features: 图像特征 [B, D]
            normalize: 是否归一化特征
        
        Returns:
            相似度矩阵元组 (logits_per_text, logits_per_image)，每个都是 [B, B]
        """
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=-1)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits_per_text = torch.matmul(text_features, image_features.T) / self.temperature
        logits_per_image = logits_per_text.T
        
        return logits_per_text, logits_per_image
    
    def compute_contrastive_loss(
        self,
        texts: Union[str, Sequence[str]],
        images: Union[Image.Image, np.ndarray, torch.Tensor, Sequence],
    ) -> dict:
        """
        计算对比学习损失（类似CLIP）
        
        Args:
            texts: 文本输入
            images: 图像输入
        
        Returns:
            包含损失的字典
        """
        text_features = self.encode_text(texts, return_pooled=True)
        image_features = self.encode_image(images, return_pooled=True)
        
        # 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # 计算相似度
        logits_per_text, logits_per_image = self.compute_similarity(
            text_features, image_features, normalize=False
        )
        
        # 创建标签（对角线为正样本）
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算交叉熵损失
        text_loss = F.cross_entropy(logits_per_text, labels)
        image_loss = F.cross_entropy(logits_per_image, labels)
        total_loss = (text_loss + image_loss) / 2
        
        return {
            "loss": total_loss,
            "text_loss": text_loss,
            "image_loss": image_loss,
            "logits_per_text": logits_per_text,
            "logits_per_image": logits_per_image,
            "text_features": text_features,
            "image_features": image_features,
        }
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """保存模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        torch.save(self.state_dict(), save_path / "model.pt")
        
        # 保存文本编码器和tokenizer
        self.text_encoder.save_pretrained(save_path / "text_encoder")
        self.text_tokenizer.save_pretrained(save_path / "text_encoder")
        
        # 保存视觉编码器和processor
        self.vision_encoder.save_pretrained(save_path / "vision_encoder")
        self.image_processor.save_pretrained(save_path / "vision_encoder")
    
    def load_pretrained(self, load_path: Union[str, Path], strict: bool = True):
        """加载模型"""
        load_path = Path(load_path)
        
        # 加载模型状态
        if (load_path / "model.pt").exists():
            state_dict = torch.load(load_path / "model.pt", map_location=self.device)
            self.load_state_dict(state_dict, strict=strict)
        
        # 加载编码器（如果需要）
        if (load_path / "text_encoder").exists():
            self.text_encoder = AutoModel.from_pretrained(load_path / "text_encoder")
            self.text_tokenizer = AutoTokenizer.from_pretrained(load_path / "text_encoder")
        
        if (load_path / "vision_encoder").exists():
            if "clip" in str(load_path).lower():
                self.vision_encoder = CLIPVisionModel.from_pretrained(load_path / "vision_encoder")
                self.image_processor = CLIPImageProcessor.from_pretrained(load_path / "vision_encoder")
            else:
                self.vision_encoder = AutoModel.from_pretrained(load_path / "vision_encoder")
                self.image_processor = AutoImageProcessor.from_pretrained(load_path / "vision_encoder")
        
        self.to(self.device)
        return self


if __name__ == '__main__':
    """
    测试用例：验证BertVLM模型的基本功能
    """
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 80)
    print("BertVLM 模型测试")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 测试1: 模型初始化
        print("\n[测试1] 模型初始化...")
        model = BertVLM(
            text_encoder_name="bert-base-uncased",
            vision_encoder_name="openai/clip-vit-base-patch32",
            embed_dim=512,
            freeze_text_encoder=False,
            freeze_vision_encoder=False,
            device=device,
            dtype=torch.float32,
        )
        print("✓ 模型初始化成功")
        
        # 测试2: 文本编码
        print("\n[测试2] 文本编码...")
        texts = ["a cat sitting on a mat", "a dog playing in the park"]
        text_features = model.encode_text(texts, return_pooled=True)
        assert text_features.shape[0] == len(texts), "文本特征批次大小不匹配"
        assert text_features.shape[1] == model.embed_dim, "文本特征维度不匹配"
        print(f"✓ 文本编码成功: {text_features.shape}")
        
        # 测试3: 图像编码（使用随机生成的图像）
        print("\n[测试3] 图像编码...")
        # 创建随机RGB图像
        test_images = [
            Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
            for _ in range(2)
        ]
        image_features = model.encode_image(test_images, return_pooled=True)
        assert image_features.shape[0] == len(test_images), "图像特征批次大小不匹配"
        assert image_features.shape[1] == model.embed_dim, "图像特征维度不匹配"
        print(f"✓ 图像编码成功: {image_features.shape}")
        
        # 测试4: 前向传播（仅文本）
        print("\n[测试4] 前向传播（仅文本）...")
        result = model.forward(texts=texts)
        assert "text_features" in result, "前向传播未返回文本特征"
        assert result["text_features"].shape == text_features.shape, "前向传播文本特征形状不匹配"
        print("✓ 前向传播（仅文本）成功")
        
        # 测试5: 前向传播（仅图像）
        print("\n[测试5] 前向传播（仅图像）...")
        result = model.forward(images=test_images)
        assert "image_features" in result, "前向传播未返回图像特征"
        assert result["image_features"].shape == image_features.shape, "前向传播图像特征形状不匹配"
        print("✓ 前向传播（仅图像）成功")
        
        # 测试6: 前向传播（文本+图像）
        print("\n[测试6] 前向传播（文本+图像）...")
        result = model.forward(texts=texts, images=test_images)
        assert "text_features" in result, "前向传播未返回文本特征"
        assert "image_features" in result, "前向传播未返回图像特征"
        print("✓ 前向传播（文本+图像）成功")
        
        # 测试7: 相似度计算
        print("\n[测试7] 相似度计算...")
        logits_per_text, logits_per_image = model.compute_similarity(
            text_features, image_features, normalize=True
        )
        assert logits_per_text.shape == (len(texts), len(test_images)), "相似度矩阵形状不匹配"
        assert logits_per_image.shape == (len(test_images), len(texts)), "相似度矩阵形状不匹配"
        print(f"✓ 相似度计算成功: {logits_per_text.shape}")
        
        # 测试8: 对比学习损失
        print("\n[测试8] 对比学习损失计算...")
        loss_dict = model.compute_contrastive_loss(texts=texts, images=test_images)
        assert "loss" in loss_dict, "损失字典中缺少loss"
        assert "text_loss" in loss_dict, "损失字典中缺少text_loss"
        assert "image_loss" in loss_dict, "损失字典中缺少image_loss"
        assert loss_dict["loss"] > 0, "损失值应该大于0"
        print(f"✓ 对比学习损失计算成功: loss={loss_dict['loss']:.4f}")
        
        # 测试9: 单文本/单图像编码
        print("\n[测试9] 单文本/单图像编码...")
        single_text_feature = model.encode_text("a single text", return_pooled=True)
        assert single_text_feature.shape[0] == 1, "单文本特征批次大小应为1"
        single_image_feature = model.encode_image(test_images[0], return_pooled=True)
        assert single_image_feature.shape[0] == 1, "单图像特征批次大小应为1"
        print("✓ 单文本/单图像编码成功")
        
        # 测试10: 使用交叉注意力
        print("\n[测试10] 交叉注意力模型...")
        model_with_cross_attn = BertVLM(
            text_encoder_name="bert-base-uncased",
            vision_encoder_name="openai/clip-vit-base-patch32",
            embed_dim=512,
            use_cross_attention=True,
            cross_attention_layers=2,
            device=device,
            dtype=torch.float32,
        )
        result = model_with_cross_attn.forward(texts=texts, images=test_images)
        assert "fused_features" in result, "交叉注意力未返回融合特征"
        assert result["fused_features"].shape[0] == len(test_images), "融合特征批次大小不匹配"
        print("✓ 交叉注意力模型测试成功")
        
        # 测试11: 冻结编码器
        print("\n[测试11] 冻结编码器测试...")
        model_frozen = BertVLM(
            text_encoder_name="bert-base-uncased",
            vision_encoder_name="openai/clip-vit-base-patch32",
            embed_dim=512,
            freeze_text_encoder=True,
            freeze_vision_encoder=True,
            device=device,
            dtype=torch.float32,
        )
        # 检查参数是否被冻结
        text_params_frozen = all(not p.requires_grad for p in model_frozen.text_encoder.parameters())
        vision_params_frozen = all(not p.requires_grad for p in model_frozen.vision_encoder.parameters())
        assert text_params_frozen, "文本编码器参数未被冻结"
        assert vision_params_frozen, "视觉编码器参数未被冻结"
        # 投影层应该可训练
        proj_params_trainable = any(p.requires_grad for p in model_frozen.text_projection.parameters())
        assert proj_params_trainable, "投影层参数应该可训练"
        print("✓ 冻结编码器测试成功")
        
        # 测试12: 不同数据类型
        if torch.cuda.is_available():
            print("\n[测试12] 不同数据类型测试...")
            for dtype in [torch.float32, torch.float16]:
                try:
                    model_dtype = BertVLM(
                        text_encoder_name="bert-base-uncased",
                        vision_encoder_name="openai/clip-vit-base-patch32",
                        embed_dim=512,
                        device=device,
                        dtype=dtype,
                    )
                    text_feat = model_dtype.encode_text(["test"], return_pooled=True)
                    assert text_feat.dtype == dtype, f"数据类型不匹配: 期望 {dtype}, 得到 {text_feat.dtype}"
                    print(f"✓ {dtype} 数据类型测试成功")
                except Exception as e:
                    print(f"⚠ {dtype} 数据类型测试失败: {e}")
        
        print("\n" + "=" * 80)
        print("所有测试通过！✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

