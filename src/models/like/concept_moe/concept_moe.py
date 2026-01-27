from typing import Optional, Sequence

from torch import nn
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPVisionModel, CLIPVisionConfig


from ...moe_layer import MoEDenseLayer, MoESparseLayer
from ...attention.MultiHeadAttention import MultiHeadCrossAttention

from src.constants import PRETRAINED_MODEL_DIR


class ObjectConceptMoELayer(nn.Module):
    def __init__(self, hidden_dim: int, individual_concepts: int = 32, shared_concepts: int = 4, expert_hidden_dim: Optional[int] = None, k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.individual_moe = MoESparseLayer(hidden_dim=hidden_dim, num_experts=individual_concepts, expert_hidden_dim=expert_hidden_dim, k=k, dropout=dropout)
        self.shared_moe = MoEDenseLayer(hidden_dim=hidden_dim, num_experts=shared_concepts, expert_hidden_dim=expert_hidden_dim, dropout=dropout)
        
    def forward(self, x, return_middle_outputs=False):
        # MoE层现在返回 (output, aux_loss) 或 (output, aux_loss, middle_outputs)
        individual_result = self.individual_moe(x, return_middle_outputs=return_middle_outputs)
        shared_result = self.shared_moe(x, return_middle_outputs=return_middle_outputs)
        
        if return_middle_outputs:
            individual_out, individual_aux_loss, individual_middle_outputs = individual_result
            shared_out, shared_aux_loss, shared_middle_outputs = shared_result
        else:
            individual_out, individual_aux_loss = individual_result
            shared_out, shared_aux_loss = shared_result
            individual_middle_outputs = None
            shared_middle_outputs = None
        
        x = individual_out + shared_out
        total_aux_loss = individual_aux_loss + shared_aux_loss
        
        if return_middle_outputs:
            return x, total_aux_loss, {
                'individual_out': individual_out,
                'shared_out': shared_out,
                'individual_aux_loss': individual_aux_loss,
                'shared_aux_loss': shared_aux_loss,
                'individual_middle_outputs': individual_middle_outputs,
                'shared_middle_outputs': shared_middle_outputs,
            }
        else:
            return x, total_aux_loss

class ObjectConceptMoE(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, individual_concepts: int = 32, shared_concepts: int = 4, expert_hidden_dim: Optional[int] = None, k: int = 16, dropout: float = 0.1, num_heads: int = 32, *, n_objs: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_objs = n_objs
        self.cross_attn = MultiHeadCrossAttention(embed_dim=hidden_dim, num_heads=num_heads, attn_dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.obj_concept_moe = ObjectConceptMoELayer(hidden_dim=hidden_dim, individual_concepts=individual_concepts, shared_concepts=shared_concepts, expert_hidden_dim=expert_hidden_dim, k=k, dropout=dropout)

    def forward(self, x, return_middle_outputs=False):
        B, N, D = x.shape
        B //= self.n_objs

        identity = x
        x = self.norm(x)
        x = self.cross_attn(x, x, x)
        x = x + identity

        identity = x
        x = self.norm(x)
        x = x.reshape(B, self.n_objs, N, D)
        result = self.obj_concept_moe(x, return_middle_outputs=return_middle_outputs)
        
        if return_middle_outputs:
            x, aux_loss, obj_concept_middle_outputs = result
        else:
            x, aux_loss = result
            obj_concept_middle_outputs = None
        
        x = x + identity
        if return_middle_outputs:
            return x, aux_loss, obj_concept_middle_outputs
        else:
            return x, aux_loss


class _ObjectConceptMoEWrapper(nn.Module):
    def __init__(self, oc_layer: ObjectConceptMoELayer, *, track_middle_outputs: bool = False):
        super().__init__()
        self.oc_layer = oc_layer
        self.track_middle_outputs = track_middle_outputs
        self.last_middle_outputs: Optional[dict] = None

    def forward(self, x):
        if self.track_middle_outputs:
            out, aux_loss, middle_outputs = self.oc_layer(x, return_middle_outputs=True)
            self.last_middle_outputs = middle_outputs
            return out
        out, aux_loss = self.oc_layer(x)
        return out


def _get_transformer_blocks(vit_model: nn.Module):
    blocks = getattr(vit_model, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks
    encoder = getattr(vit_model, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError("无法在给定模型中找到可替换的Transformer blocks。")


def _infer_embed_dim(vit_model: nn.Module, target_block: nn.Module):
    embed_dim = getattr(vit_model, "embed_dim", None)
    if embed_dim is None:
        config = getattr(vit_model, "config", None)
        embed_dim = getattr(config, "hidden_size", None)
    if embed_dim is None:
        mlp = getattr(target_block, "mlp", None)
        fc2 = getattr(mlp, "fc2", None)
        if isinstance(fc2, nn.Linear):
            embed_dim = fc2.out_features
    if not isinstance(embed_dim, int):
        raise ValueError("无法推断ViT的嵌入维度，请显式传入。")
    if embed_dim is None:
        raise ValueError("无法推断ViT的嵌入维度，请显式传入。")
    return embed_dim


def replace_vit_ffn_with_object_concept_moe(
    vit_model: nn.Module,
    *,
    block_index: int = -2,
    individual_concepts: int = 32,
    shared_concepts: int = 4,
    expert_hidden_dim: Optional[int] = None,
    k: int = 16,
    dropout: float = 0.1,
    track_middle_outputs: bool = False,
) -> int:
    """
    将给定ViT骨干的某个Block中的FFN（mlp）替换为ObjectConceptMoELayer。
    """
    blocks = _get_transformer_blocks(vit_model)

    if block_index < 0:
        block_index = len(blocks) + block_index
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"block_index={block_index} 超出范围，当前共有 {len(blocks)} 个blocks。")

    target_block = blocks[block_index]
    embed_dim = _infer_embed_dim(vit_model, target_block)

    inferred_hidden = None
    mlp_module = getattr(target_block, "mlp", None)
    hidden_features = getattr(mlp_module, "hidden_features", None)
    if isinstance(hidden_features, int):
        inferred_hidden = hidden_features
    else:
        fc1 = getattr(mlp_module, "fc1", None)
        if isinstance(fc1, nn.Linear):
            inferred_hidden = fc1.out_features
    if expert_hidden_dim is None:
        expert_hidden_dim = inferred_hidden or embed_dim

    oc_layer = ObjectConceptMoELayer(
        hidden_dim=embed_dim,
        individual_concepts=individual_concepts,
        shared_concepts=shared_concepts,
        expert_hidden_dim=expert_hidden_dim,
        k=k,
        dropout=dropout,
    )
    if track_middle_outputs:
        target_block.mlp = _ObjectConceptMoEWrapper(oc_layer, track_middle_outputs=True)
    else:
        target_block.mlp = oc_layer
    return block_index


class ViTWithObjectConceptMoE(nn.Module):
    """
    基于CLIP视觉骨干，将倒数第二层（默认）FFN替换成ObjectConceptMoELayer，并加载预训练权重。
    """

    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch16",
        pretrained: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        block_index: int = -2,
        cache_dir: str = PRETRAINED_MODEL_DIR,
        individual_concepts: int = 32,
        shared_concepts: int = 4,
        expert_hidden_dim: Optional[int] = None,
        k: int = 16,
        dropout: float = 0.1,
        track_middle_outputs: bool = False,
    ):
        super().__init__()
        self._track_middle_outputs = track_middle_outputs
        if pretrained:
            vision_model = CLIPVisionModel.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            config = CLIPVisionConfig.from_pretrained(model_name, cache_dir=cache_dir)
            vision_model = CLIPVisionModel(config)

        if pretrained_checkpoint is not None:
            state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
            missing = vision_model.load_state_dict(state_dict, strict=False)
            if missing.unexpected_keys:
                raise RuntimeError(f"加载预训练权重时出现未匹配key: {missing.unexpected_keys}")

        self.vit = vision_model
        self._backbone = self.vit.vision_model
        self._blocks = _get_transformer_blocks(self._backbone)
        self._ocm_block_index = replace_vit_ffn_with_object_concept_moe(
            self._backbone,
            block_index=block_index,
            individual_concepts=individual_concepts,
            shared_concepts=shared_concepts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            track_middle_outputs=track_middle_outputs,
        )

        for _, param in self.vit.named_parameters():
            param.requires_grad = False

    def forward(self, x, return_middle_outputs=False):
        # x: (B, N_OBJ, C, H, W)
        B, N_OBJ, C, H, W = x.shape
        x = x.reshape(B * N_OBJ, C, H, W)
        outputs = self.vit(pixel_values=x, return_dict=True)
        tokens = outputs.last_hidden_state  # (B * N_OBJ, num_tokens, embed_dim)
        if tokens.dim() != 3:
            raise RuntimeError(f"CLIP视觉模型应返回(B*num_objs, num_tokens, embed_dim)，当前形状: {tokens.shape}")
        _, N, D = tokens.shape
        tokens = tokens.reshape(B, N_OBJ, N, D)
        if return_middle_outputs:
            middle_outputs = self.get_last_middle_outputs()
            return tokens, middle_outputs
        return tokens

    def get_last_middle_outputs(self):
        if not self._track_middle_outputs:
            raise RuntimeError("track_middle_outputs=False，无法获取中间输出")
        blocks = self._blocks
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("当前ViT模型不包含blocks。")
        block = blocks[self._ocm_block_index]
        mlp = getattr(block, "mlp", None)
        if mlp is None or not isinstance(mlp, _ObjectConceptMoEWrapper):
            raise RuntimeError("目标block的mlp不是ObjectConceptMoEWrapper，无法读取中间输出。")
        return mlp.last_middle_outputs

class ViTOCM_CLIP_Alignment(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", pretrained: bool = True, pretrained_checkpoint: Optional[str] = None, block_index: int = -2, cache_dir: str = PRETRAINED_MODEL_DIR, individual_concepts: int = 32, shared_concepts: int = 4, expert_hidden_dim: Optional[int] = None, k: int = 16, dropout: float = 0.1, *, n_objs: int = 1):
        super().__init__()
        self.n_objs = n_objs
        self.vit_ocm = ViTWithObjectConceptMoE(
            model_name=model_name,
            pretrained=pretrained,
            pretrained_checkpoint=pretrained_checkpoint,
            block_index=block_index,
            cache_dir=cache_dir,
            individual_concepts=individual_concepts,
            shared_concepts=shared_concepts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            track_middle_outputs=True,
        )

        self.text_encoder = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.text_tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    def _flatten_texts(self, texts, expected_count: int):
        if isinstance(texts, (list, tuple)):
            if len(texts) == expected_count and isinstance(texts[0], str):
                return list(texts)
            if len(texts) != expected_count and all(isinstance(t, (list, tuple)) for t in texts):
                flat = [item for seq in texts for item in seq]
            else:
                flat = list(texts)
            if len(flat) != expected_count:
                raise ValueError(f"文本数量({len(flat)})与对象数量({expected_count})不匹配。")
            return flat
        raise TypeError("texts 需为字符串序列或其嵌套序列，或已tokenize的字典。")

    def forward(self, x, texts: Sequence[Sequence[str]] | Sequence[str] | dict, return_middle_outputs=False):
        B, N_OBJ, C, H, W = x.shape
        if N_OBJ != self.n_objs:
            raise ValueError(f"Expect n_objs={self.n_objs}, but got {N_OBJ}")

        vision_tokens, concept_middle_outputs = self.vit_ocm(x, return_middle_outputs=True)
        concept_tensor = concept_middle_outputs['individual_out']  # (B, N_OBJ, N, D)
        shape_tensor = concept_middle_outputs['shared_out']        # (B, N_OBJ, N, D)
        concept_feats = concept_tensor.reshape(B, N_OBJ, -1)
        shape_feats = shape_tensor.reshape(B, N_OBJ, -1)

        device = x.device
        expected_texts = B * N_OBJ
        if isinstance(texts, dict):
            text_inputs = {k: v.to(device) for k, v in texts.items()}
            text_batch = next(iter(text_inputs.values())).shape[0]
            if text_batch != expected_texts:
                raise ValueError(f"已tokenize文本batch={text_batch}，需与 {expected_texts} 一致。")
        else:
            flat_texts = self._flatten_texts(texts, expected_texts)
            text_inputs = self.text_tokenizer(flat_texts, padding=True, truncation=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        self.text_encoder = self.text_encoder.to(device)
        text_feats = self.text_encoder.get_text_features(**text_inputs)  # (B*N_OBJ, D_text)
        text_feats = text_feats.view(B, N_OBJ, -1)

        outputs = {
            "vision_tokens": vision_tokens,
            "concept_feats": concept_feats,
            "shape_feats": shape_feats,
            "text_feats": text_feats,
        }
        if return_middle_outputs:
            outputs["moe_middle_outputs"] = concept_middle_outputs
        return outputs


if __name__ == "__main__":
    x = torch.randn(2, 12, 3, 224, 224)
    sample_texts = [["object" for _ in range(12)] for _ in range(2)]
    model = ViTOCM_CLIP_Alignment(n_objs=12)
    outputs = model(x, sample_texts)
    print(outputs["vision_tokens"].shape)