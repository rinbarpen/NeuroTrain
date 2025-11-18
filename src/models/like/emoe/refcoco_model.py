"""
EMOE模型用于RefCOCO数据集的Region-Text对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Dict, Any

from .emoe import EViTMoE
from src.constants import PRETRAINED_MODEL_DIR
from src.utils.ndict import ModelOutput


class EMOE_RefCOCO(nn.Module):
    """
    EMOE模型用于RefCOCO数据集的Region-Text对齐
    
    模型结构：
    1. EMOE编码器：编码图像区域 (B, N_OBJ, C, H, W) -> (B, N_OBJ, D_region)
    2. 文本编码器：编码文本描述 -> (B, D_text)
    3. 对齐投影层：将region和text特征投影到对齐空间
    """
    
    def __init__(
        self,
        # EMOE配置
        backbone='vit_base_patch16_224',
        vit_hidden_dim=768,
        num_heads=8,
        num_experts=4,
        expert_hidden_dim=None,
        k=2,
        sparse=True,
        dropout=0.1,
        moe=True,
        attn=True,
        # 文本编码器配置
        text_encoder_name='openai/clip-vit-base-patch32',
        text_encoder_dim=512,
        # 对齐层配置
        alignment_dim=512,
        temperature=0.07,
        # 其他配置
        cache_dir=None,
    ):
        super(EMOE_RefCOCO, self).__init__()
        
        self.alignment_dim = alignment_dim
        self.temperature = temperature
        self.cache_dir = cache_dir or PRETRAINED_MODEL_DIR
        
        # EMOE区域编码器
        self.region_encoder = EViTMoE(
            backbone=backbone,
            vit_hidden_dim=vit_hidden_dim,
            num_heads=num_heads,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            sparse=sparse,
            dropout=dropout,
            moe=moe,
            attn=attn,
        )
        
        # 文本编码器（使用CLIP的文本编码器部分）
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_encoder_name,
            cache_dir=self.cache_dir
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            text_encoder_name,
            cache_dir=self.cache_dir
        )
        
        # 对齐投影层
        self.region_proj = nn.Linear(vit_hidden_dim, alignment_dim)
        self.text_proj = nn.Linear(text_encoder_dim, alignment_dim)
        
        # 初始化投影层
        nn.init.normal_(self.region_proj.weight, std=0.02)
        nn.init.normal_(self.text_proj.weight, std=0.02)
    
    def encode_regions(self, regions: torch.Tensor) -> torch.Tensor:
        """
        编码图像区域
        
        Args:
            regions: (B, N_OBJ, C, H, W) 图像区域tensor
        
        Returns:
            region_features: (B, N_OBJ, alignment_dim) 区域特征
        """
        # EMOE编码
        region_features = self.region_encoder(regions)  # (B, N_OBJ, vit_hidden_dim)
        
        # 投影到对齐空间
        region_features = self.region_proj(region_features)  # (B, N_OBJ, alignment_dim)
        
        return region_features
    
    def encode_text(self, texts: torch.Tensor, text_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码文本描述
        
        Args:
            texts: (B, max_regions, max_seq_len) 文本token ids
            text_attn_mask: (B, max_regions, max_seq_len) 文本attention mask
        
        Returns:
            text_features: (B, max_regions, alignment_dim) 文本特征
        """
        B, max_regions, max_seq_len = texts.shape
        
        # 将3D tensor reshape为2D以适配CLIP文本编码器
        # (B, max_regions, max_seq_len) -> (B * max_regions, max_seq_len)
        texts_2d = texts.reshape(B * max_regions, max_seq_len)
        if text_attn_mask is not None:
            text_attn_mask_2d = text_attn_mask.reshape(B * max_regions, max_seq_len)
        else:
            text_attn_mask_2d = None
        
        # 文本编码
        text_outputs = self.text_encoder(texts_2d, attention_mask=text_attn_mask_2d)
        # 使用pooler_output或取最后一个hidden state的平均值
        # text_outputs.last_hidden_state: (B * max_regions, max_seq_len, hidden_dim)
        text_features_2d = text_outputs.last_hidden_state.mean(dim=1)  # (B * max_regions, hidden_dim)
        
        # 将2D tensor reshape回3D
        # (B * max_regions, hidden_dim) -> (B, max_regions, hidden_dim)
        text_features = text_features_2d.reshape(B, max_regions, -1)
        
        # 投影到对齐空间
        text_features = self.text_proj(text_features)  # (B, max_regions, alignment_dim)
        
        return text_features
    
    def forward(
        self,
        regions: torch.Tensor,
        texts: torch.Tensor,
        text_attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            regions: (B, N_OBJ, C, H, W) 图像区域
            texts: (B, max_regions, max_seq_len) 文本token ids
            text_attn_mask: (B, max_regions, max_seq_len) 文本attention mask
            return_dict: 是否返回字典格式
        
        Returns:
            包含region_features和text_features的字典
        """
        # 编码区域
        region_features = self.encode_regions(regions)  # (B, N_OBJ, alignment_dim)
        
        # 编码文本
        text_features = self.encode_text(texts, text_attn_mask=text_attn_mask)  # (B, max_regions, alignment_dim)

        # L2归一化特征
        region_features = F.normalize(region_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度（用于对比学习）
        # text_features: (B, max_regions, alignment_dim)
        # region_features: (B, N_OBJ, alignment_dim)
        
        # remove background region
        global_region_features = region_features[:, 0, :]  # (B, alignment_dim)
        region_features = region_features[:, 1:, :]  # (B, N_OBJ-1, alignment_dim)
        
        # 确保text_features和region_features的第二个维度匹配
        # 如果text_features包含背景对应的文本，需要移除
        # 假设text_features的第二个维度应该是N_OBJ（不包括背景），需要匹配N_OBJ-1
        # 或者假设text_features的第二个维度已经是N_OBJ-1
        B, max_regions, alignment_dim = text_features.shape
        _, num_regions, _ = region_features.shape
        
        # 如果维度不匹配，取前num_regions个文本特征
        if max_regions > num_regions:
            text_features = text_features[:, :num_regions, :]  # (B, N_OBJ-1, alignment_dim)
        elif max_regions < num_regions:
            # 如果文本特征少于区域特征，需要padding（但这种情况不应该发生）
            padding = torch.zeros(B, num_regions - max_regions, alignment_dim, 
                                 device=text_features.device, dtype=text_features.dtype)
            text_features = torch.cat([text_features, padding], dim=1)  # (B, N_OBJ-1, alignment_dim)

        # 计算相似度：text_features @ region_features^T
        # text_features: (B, N_OBJ-1, alignment_dim)
        # region_features: (B, N_OBJ-1, alignment_dim)
        # 使用torch.bmm进行批次矩阵乘法
        logits_per_text = torch.bmm(
            text_features, 
            region_features.transpose(1, 2)
        ) / self.temperature  # (B, N_OBJ-1, N_OBJ-1)
        logits_per_region = logits_per_text.transpose(1, 2)  # (B, N_OBJ-1, N_OBJ-1)

        # 为对比学习准备标签：每个文本i应该匹配区域i
        num_regions = region_features.shape[1]
        labels = torch.arange(num_regions, device=region_features.device)  # (N_OBJ-1,)
        
        # Reshape logits for cross_entropy: (B, N_OBJ-1, N_OBJ-1) -> (B * N_OBJ-1, N_OBJ-1)
        B = logits_per_text.shape[0]
        logits_per_text_flat = logits_per_text.reshape(B * num_regions, num_regions)  # (B * N_OBJ-1, N_OBJ-1)
        logits_per_region_flat = logits_per_region.reshape(B * num_regions, num_regions)  # (B * N_OBJ-1, N_OBJ-1)
        
        # 扩展labels以匹配batch维度
        labels_expanded = labels.unsqueeze(0).expand(B, -1).reshape(-1)  # (B * N_OBJ-1,)
        
        loss_text = F.cross_entropy(logits_per_text_flat, labels_expanded)
        loss_region_raw = F.cross_entropy(logits_per_region_flat, labels_expanded)
        loss_region = (loss_text + loss_region_raw) / 2
        total_loss = loss_region

        # total_loss
        # region_features | global_region_features
        # text_features
        # logits_per_text | global_logits_per_text
        # logits_per_region | global_logits_per_region
        return {
            'region_features': region_features,
            'text_features': text_features,
            'global_region_features': global_region_features,
            'logits_per_text': logits_per_text,
            'logits_per_region': logits_per_region,
            'loss_text': loss_text,
            'loss_region': loss_region,
            'total_loss': total_loss,
        }
    
    def get_text_tokenizer(self):
        """获取文本tokenizer"""
        return self.tokenizer


class EMOERefCOCOModelWrapper(nn.Module):
    """
    EMOE模型包装器，用于适配训练器的输入格式
    
    训练器期望模型能够接受 (inputs, targets) 或 inputs，
    本包装器从batch数据中提取regions和texts，调用EMOE模型
    """
    def __init__(self, model: EMOE_RefCOCO):
        super(EMOERefCOCOModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, batch_data: Dict[str, Any]):
        """
        前向传播
        
        Args:
            batch_data: 包含'regions'和'texts'的字典
            return_dict: 是否返回字典格式
        
        Returns:
            如果return_dict=True，返回包含logits的字典，用于损失计算
            如果return_dict=False，返回 (outputs, loss)
        """
        regions = batch_data.get('regions')
        if regions is None:
            regions = batch_data.get('inputs')

        texts = batch_data.get('texts')
        if texts is None:
            texts = batch_data.get('text_ids')

        text_attn_mask = batch_data.get('text_attn_mask')
        batch_targets = batch_data.get('targets')
        if batch_targets is None:
            batch_targets = batch_data.get('category_ids')

        if regions is None or texts is None:
            raise ValueError("batch_data必须包含'regions'或'inputs'以及'texts'或'text_ids'")

        outputs = self.model(regions, texts, text_attn_mask=text_attn_mask)

        total_loss = outputs.get('total_loss')
        if total_loss is None:
            raise ValueError("EMOE model must provide 'total_loss' for training.")

        preds = outputs.get('logits_per_region')
        if preds is None:
            preds = outputs.get('logits_per_text')
        model_output = ModelOutput(preds=preds, targets=batch_targets, loss=total_loss)
        for key, value in outputs.items():
            if key == 'total_loss':
                continue
            model_output[key] = value

        return model_output, total_loss

