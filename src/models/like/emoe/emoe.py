import torch
from torch import nn
import torch.nn.functional as F
import timm

from src.constants import PRETRAINED_MODEL_DIR

class MoELayer(nn.Module):
    """
    Mixture-of-Experts Layer for object-level routing.
    输入: x shape (B, N_OBJ, D)
    对 N_OBJ 维度做路由（每个OBJ分配专家），每个对象分配top-k个专家。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts=4,
        expert_hidden_dim=None,
        k=1,
        sparse=True,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.sparse = sparse
        self.dropout = dropout

        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim

        # Experts: list of MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, output_dim),
                nn.Dropout(dropout),
            ) for _ in range(num_experts)
        ])

        # Routing: 用于对每个OBJ进行专家选择
        # 使用简单线性层将OBJ特征转为专家分数
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x: (B, N_OBJ, D)
        路由在N_OBJ维度，对每个OBJ选择专家。
        返回 shape: (B, N_OBJ, output_dim)
        """
        B, N_OBJ, D = x.shape

        routing_logits = self.router(x)  # (B, N_OBJ, num_experts)
        if self.sparse:
            # 只选k个专家，top-k mask
            topk_vals, topk_idx = routing_logits.topk(self.k, dim=-1)  # (B, N_OBJ, k)
            mask = torch.zeros_like(routing_logits)  # (B, N_OBJ, num_experts)
            mask.scatter_(-1, topk_idx, 1.0)
            coef = F.softmax(topk_vals, dim=-1)  # (B, N_OBJ, k)
        else:
            # dense: 所有专家参与
            coef = F.softmax(routing_logits, dim=-1)  # (B, N_OBJ, num_experts)
            mask = torch.ones_like(coef)

        # 对于每个专家，将输入中的OBJ送入相应专家，每个OBJ输出组合
        out = 0
        for eid, expert in enumerate(self.experts):
            expert_out = expert(x)  # (B, N_OBJ, output_dim)

            if self.sparse:
                # 对于每个对象，检查是否选择了这个专家，如果选择了，找到它在topk中的位置
                eid_coef = torch.zeros(B, N_OBJ, 1, device=x.device, dtype=x.dtype)
                # topk_idx: (B, N_OBJ, k), coef: (B, N_OBJ, k)
                # 找到每个对象中eid在topk中的位置
                for b_idx in range(B):
                    for obj_idx in range(N_OBJ):
                        obj_topk = topk_idx[b_idx, obj_idx]  # (k,)
                        if (obj_topk == eid).any():
                            # 找到eid在topk中的位置
                            k_pos = (obj_topk == eid).nonzero(as_tuple=True)[0][0]
                            eid_coef[b_idx, obj_idx, 0] = coef[b_idx, obj_idx, k_pos]
                out = out + expert_out * eid_coef
            else:
                eid_coef = coef[..., eid].unsqueeze(-1)  # (B, N_OBJ, 1)
                out = out + expert_out * eid_coef

        return out

class SA(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_dropout=0.0, qkv_bias=False):
        super(SA, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim需能被num_heads整除"
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        # q, k, v shape: (B, N_OBJ, D) or (B, N_OBJ, N, D)
        if len(q.shape) == 3:
            # 如果是3维输入，添加N维度 (N=1)
            q = q.unsqueeze(2)  # (B, N_OBJ, 1, D)
            k = k.unsqueeze(2)  # (B, N_OBJ, 1, D)
            v = v.unsqueeze(2)  # (B, N_OBJ, 1, D)
        
        B, N_OBJ, N, D = q.shape
        # For object-level attention, reshape to treat objects as sequence
        # we want to attend over N_OBJ dimension, i.e. objects
        # Merge batch and N (intra-obj tokens) dims: new_q = (B*N, N_OBJ, D)
        q = q.transpose(1, 2).reshape(B*N, N_OBJ, D)  # (B*N, N_OBJ, D)
        k = k.transpose(1, 2).reshape(B*N, N_OBJ, D)
        v = v.transpose(1, 2).reshape(B*N, N_OBJ, D)

        # Compute qkv
        qkv = self.qkv(q)  # (B*N, N_OBJ, 3*D)
        qkv = qkv.reshape(B*N, N_OBJ, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B*N, num_heads, N_OBJ, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B*N, num_heads, N_OBJ, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B*N, num_heads, N_OBJ, N_OBJ)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)  # (B*N, num_heads, N_OBJ, head_dim)
        out = out.transpose(1, 2).reshape(B*N, N_OBJ, D)  # (B*N, N_OBJ, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # reshape back to (B, N_OBJ, N, D)
        out = out.reshape(B, N, N_OBJ, D).transpose(1, 2)  # (B, N_OBJ, N, D)
        
        # 如果原始输入是3维的，去掉N维度
        if N == 1:
            out = out.squeeze(2)  # (B, N_OBJ, D)
        
        return out

class EViTMoE(nn.Module):
    def __init__(self, backbone='vit_base_patch16_224', vit_hidden_dim=768, num_heads=8, num_experts=4, expert_hidden_dim=None, k=2, sparse=True, dropout=0.1, *, moe=True, attn=True):
        super(EViTMoE, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, cache_dir=PRETRAINED_MODEL_DIR, num_classes=0)
        self.norm1 = nn.LayerNorm(vit_hidden_dim)
        self.norm2 = nn.LayerNorm(vit_hidden_dim)
        if attn:
            self.attn = SA(embed_dim=vit_hidden_dim, num_heads=num_heads, attn_dropout=dropout)
        else:
            self.attn = None
        if moe:
            self.moe = MoELayer(input_dim=vit_hidden_dim, output_dim=vit_hidden_dim, num_experts=num_experts, expert_hidden_dim=expert_hidden_dim, k=k, sparse=sparse, dropout=dropout)
        else:
            self.moe = None

    def forward(self, x):
        b, N_OBJ, c, h, w = x.shape
        
        x_objs = [self.backbone(x_obj) for x_obj in x.unbind(dim=1)]
        x = torch.stack(x_objs, dim=1) # (B, N_OBJ, D)
        
        if self.attn is not None:
            identity = x
            x = self.norm1(x)
            x = self.attn(x, x, x)
            x = x + identity # (B, N_OBJ, D)
        
        if self.moe is not None:
            identity = x
            x = self.norm2(x)
            x = self.moe(x)
            x = x + identity

        return x

if __name__ == '__main__':
    model = EViTMoE(backbone='vit_base_patch16_224', vit_hidden_dim=768, num_heads=8, num_experts=4, expert_hidden_dim=None, k=2, sparse=True, dropout=0.1)
    model = model.cuda()
    x = torch.randn(2, 32, 3, 224, 224)
    x = x.cuda()
    print(model(x).shape)
