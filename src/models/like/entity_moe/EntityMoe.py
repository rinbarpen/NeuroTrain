import torch
from torch import nn
from ...moe_layer import MoELayer
from ...attention.MultiHeadAttention import MultiHeadCrossAttention


class ObjectMoELayer(nn.Module):
    """
    基于MoELayer实现的 object-level MoE 层，对每个 object 先用 MoE 表示，再通过 cross-attention 关联各 object，最后加FFN（含dropout）。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts=4,
        num_experts_shared=2,
        expert_hidden_dim=None,
        k=1,
        sparse=True,
        dropout=0.1,
        ffn_hidden_dim=None,
        cross_attn_heads=1,
        cross_attn_dropout=0.0,
    ):
        super().__init__()
        # 基于MoELayer实现
        self.moe = MoELayer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            sparse=sparse,
            dropout=dropout,
        )
        # 正确引用MultiHeadCrossAttention(项目已有)
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=output_dim,
            num_heads=cross_attn_heads,
            attn_dropout=cross_attn_dropout,
        )
        # FFN部分
        ffn_hidden_dim = ffn_hidden_dim or (output_dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        """
        x: shape = (B, N_OBJ, ..., D)，即除N_OBJ和D外可以有任意数量的中间维度
        context: (可选) (B, M, output_dim)，用于作为cross-attention的key/value，若为None则degrade为self-attention。
        输出: shape = (B, N_OBJ, ..., output_dim)
        """
        if x.dim() < 3:
            raise ValueError(f"x must be at least 3D (B, N_OBJ, D), got {x.shape}")

        B, N_OBJ = x.shape[:2]
        other_shape = x.shape[2:-1]  # ... 部分
        D = x.shape[-1]

        num_items = 1
        for s in other_shape:
            num_items *= s
        # 将所有object维度flatten，方便批量处理
        x_flat = x.view(B * N_OBJ * num_items, D)
        out_flat = self.moe(x_flat)  # (B*N_OBJ*num_items, output_dim)
        # 恢复到 (B, N_OBJ, ..., output_dim)
        out_shape = (B, N_OBJ, *other_shape, -1)
        obj_feat = out_flat.view(
            B, N_OBJ * num_items, -1
        )  # (B, N_OBJ*num_items, output_dim)

        if context is not None:
            if context.shape[0] != B or context.shape[2] != obj_feat.shape[2]:
                raise ValueError("context shape must be (B, M, output_dim)")
            query = obj_feat
            key = context
            value = context
        else:
            query = obj_feat
            key = obj_feat
            value = obj_feat

        rel_out, _ = self.cross_attn(
            query, key, value
        )  # (B, N_OBJ*num_items, output_dim)
        rel_out = rel_out.view(*out_shape)  # (B, N_OBJ, ..., output_dim)
        ffn_out = self.ffn(rel_out)
        return ffn_out


class EntityMoELayer(nn.Module):
    """
    EntityMoELayer
    该层以entity作为最小单元，每个entity内部包含多个object。输入形状应为 (B, N_ENTITY, N_OBJ_PER_ENTITY, D)
    用法:
      - entity粒度的Gating，entity内部多个object整体送入MoE并聚合表示
      - 可结合MoELayer
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_experts=4,
        num_experts_shared=0,
        expert_hidden_dim=None,
        k=1,
        sparse=True,
        dropout=0.1,
        ffn_hidden_dim=None,
        cross_attn_heads=1,
        cross_attn_dropout=0.0,
    ):
        super().__init__()
        # 基于MoELayer，每个entity整体送入MoE
        self.moe = MoELayer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            num_experts_shared=num_experts_shared,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            sparse=sparse,
            dropout=dropout,
        )
        # 正确引用MultiHeadCrossAttention
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=output_dim,
            num_heads=cross_attn_heads,
            attn_dropout=cross_attn_dropout,
        )
        ffn_hidden_dim = ffn_hidden_dim or (output_dim * 4)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None):
        """
        x: shape = (B, N_ENTITY, N_OBJ_PER_ENTITY, D)
        context: (可选) (B, M, output_dim)，用于作为cross-attention的key/value，若为None则degrade为self-attention。
        输出: shape = (B, N_ENTITY, output_dim)
        加权聚合entity内部objects, 对EntityFeats进行加权
        """
        if x.dim() != 4:
            raise ValueError(
                f"x should be 4D (B, N_ENTITY, N_OBJ_PER_ENTITY, D), got {x.shape}"
            )

        B, N_ENTITY, N_OBJ_PER_ENTITY, D = x.shape

        # ---- Step 1: 对entity内部objects做加权聚合 ----
        attn_linear = getattr(self, "attn_linear", None)
        if attn_linear is None or attn_linear.in_features != D:
            # run-time实例化防止N_OBJ_PER_ENTITY未定，新加属性
            self.attn_linear = nn.Linear(D, 1, bias=False).to(x.device)
        logits = self.attn_linear(x)  # (B, N_ENTITY, N_OBJ_PER_ENTITY, 1)
        weights = torch.softmax(logits, dim=2)  # (B, N_ENTITY, N_OBJ_PER_ENTITY, 1)
        x_agg = (x * weights).sum(dim=2)  # (B, N_ENTITY, D)
        # ------------------------------------------------

        x_flat = x_agg.view(B * N_ENTITY, D)
        moe_out = self.moe(x_flat)  # (B*N_ENTITY, output_dim)
        entity_feat = moe_out.view(B, N_ENTITY, -1)  # (B, N_ENTITY, output_dim)

        # entity间做cross-attention建模
        if context is not None:
            if context.shape[0] != B or context.shape[2] != entity_feat.shape[2]:
                raise ValueError("context shape must be (B, M, output_dim)")
            query = entity_feat
            key = context
            value = context
        else:
            query = entity_feat
            key = entity_feat
            value = entity_feat

        rel_out, _ = self.cross_attn(query, key, value)  # (B, N_ENTITY, output_dim)
        ffn_out = self.ffn(rel_out)  # (B, N_ENTITY, output_dim)
        return ffn_out
