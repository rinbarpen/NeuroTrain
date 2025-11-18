import torch
from torch import nn

class MoEDenseLayer(nn.Module):
    """
    Mixture-of-Experts (MoE) module for object-level representations (Dense version).
    只支持致密(dense)路由。
    """
    def __init__(
        self,
        hidden_dim,
        num_experts=4,
        expert_hidden_dim=None,
        dropout=0.1,
    ):
        """
        hidden_dim: 输入向量维数
        num_experts: 专家数量
        expert_hidden_dim: 专家mlp隐藏层, 若None则同input_dim
        dropout: Dropout概率，应用于每个专家内部
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim * 4

        # 定义所有专家的mlp，每个专家包含Dropout
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_hidden_dim),
                    nn.GELU(),
                    nn.Linear(expert_hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        # 门控网络，全部专家都参与门控
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x, return_middle_outputs=False):
        """
        x: shape (batch_size, ..., input_dim)
        """
        B = x.shape[0]
        D = x.shape[-1]
        orig_shape = x.shape
        x_flat = x.view(B, -1, D)  # (B, seq_len, D)
        gating_logits = self.gate(x_flat)  # (B, seq_len, num_experts)
        gate_scores = nn.functional.softmax(gating_logits, dim=-1)  # (B, seq_len, num_experts)

        # 专家输出
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=1
        )  # (B, num_experts, seq_len, output_dim)

        # 转置expert_outputs以匹配gate_scores的形状
        expert_outputs = expert_outputs.transpose(1, 2)  # (B, seq_len, num_experts, output_dim)
        gate_scores_unsq = gate_scores.unsqueeze(-1)  # (B, seq_len, num_experts, 1)
        output = (expert_outputs * gate_scores_unsq).sum(dim=2)  # (B, seq_len, output_dim)
        output = output.view(*orig_shape)
        if return_middle_outputs:
            return output, {
                'expert_outputs': expert_outputs,
                'gate_scores': gate_scores,
            }
        else:
            return output


class MoESparseLayer(nn.Module):
    """
    Mixture-of-Experts (MoE) module for object-level representations (Sparse version).
    只支持稀疏(sparse)路由。
    """
    def __init__(
        self,
        hidden_dim,
        num_experts=4,
        expert_hidden_dim=None,
        k=1,
        dropout=0.1,
    ):
        """
        hidden_dim: 输入向量维数
        num_experts: 专家数量
        expert_hidden_dim: 专家mlp隐藏层, 若None则同input_dim
        k: 每个输入选择的专家个数
        dropout: Dropout概率，应用于每个专家内部
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        if expert_hidden_dim is None:
            expert_hidden_dim = hidden_dim * 4

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, expert_hidden_dim),
                    nn.GELU(),
                    nn.Linear(expert_hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(hidden_dim, num_experts)

    def forward(self, x, return_middle_outputs=False):
        """
        x: shape (batch_size, ..., input_dim)
        更高效稀疏路由实现
        """
        B = x.shape[0]
        D = x.shape[-1]
        orig_shape = x.shape
        x_flat = x.view(B, -1, D)  # (B, seq_len, D)
        x_2d = x_flat.reshape(-1, D)  # (B * seq_len, D)

        gating_logits = self.gate(x_2d)  # (B * seq_len, num_experts)
        gate_scores = nn.functional.softmax(gating_logits, dim=-1)  # (B * seq_len, num_experts)

        topk_scores, topk_indices = gate_scores.topk(self.k, dim=-1)  # (B * seq_len, k)

        expert_outputs = torch.stack(
            [expert(x_2d) for expert in self.experts], dim=1
        )  # (B * seq_len, num_experts, output_dim)

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        topk_expert_outputs = torch.gather(expert_outputs, 1, gather_index)  # (B * seq_len, k, output_dim)

        out = (topk_expert_outputs * topk_scores.unsqueeze(-1)).sum(dim=1)  # (B * seq_len, output_dim)
        out = out.view(*orig_shape)
        if return_middle_outputs:
            return out, {
                'expert_outputs': expert_outputs,
                'topk_scores': topk_scores,
                'topk_indices': topk_indices,
                'gate_scores': gate_scores,
            }
        else:
            return out


class VisionMoELayer(MoESparseLayer):
    """
    VisionMoELayer 即 MoESparseLayer，专为视觉任务配置。
    """
    def __init__(self, hidden_dim, num_experts=32, expert_hidden_dim=None, k=16, dropout=0.1):
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout
        )

if __name__ == "__main__":
    x = torch.randn(2, 197, 768)
    model = VisionMoELayer(hidden_dim=768, num_experts=32, expert_hidden_dim=None, k=16, dropout=0.1)
    y = model(x)
    print(y.shape)