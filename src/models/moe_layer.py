import torch
from torch import nn

class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) module for object-level representations.
    支持稀疏(sparse)与致密(dense)两种路由方式，并结合共享专家（shared expert）。
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
    ):
        """
        input_dim: 输入向量维数
        output_dim: 输出向量维数
        num_experts: 专家数量
        num_experts_shared: 共享专家数量
        expert_hidden_dim: 专家mlp隐藏层, 若None则同input_dim
        k: sparse时每个输入选择的专家个数，dense时无效
        sparse: 是否采用稀疏路由（否则为dense全专家）
        dropout: Dropout概率，应用于每个专家内部
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_experts_shared = num_experts_shared
        self.k = k
        self.sparse = sparse
        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim

        # 定义所有专家的mlp，每个专家包含Dropout
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, output_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )
        # 共享专家
        self.experts_shared = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, output_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts_shared)
            ]
        )
        # 门控网络，全部专家和共享专家都参与门控
        self.gate = nn.Linear(input_dim, num_experts + num_experts_shared)

    def forward(self, x):
        """
        x: shape (batch_size, input_dim)
        """
        gating_logits = self.gate(x)  # (batch, num_experts_shared + num_experts)
        gate_scores = nn.functional.softmax(gating_logits, dim=-1)  # 概率

        # 分离共享专家和非共享专家的对应门控权重
        shared_gate_scores = gate_scores[
            :, : self.num_experts_shared
        ]  # (batch, num_experts_shared)
        expert_gate_scores = gate_scores[
            :, self.num_experts_shared :
        ]  # (batch, num_experts)

        # 专家输出
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (batch, num_experts, output_dim)

        # 共享专家输出
        shared_outputs = None
        if self.num_experts_shared > 0:
            shared_outputs = torch.stack(
                [expert(x) for expert in self.experts_shared], dim=1
            )  # (batch, num_experts_shared, output_dim)
            # 门控加权共享专家输出
            shared_output = (shared_gate_scores.unsqueeze(-1) * shared_outputs).sum(
                dim=1
            )  # (batch, output_dim)
        else:
            shared_output = torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=expert_outputs.dtype
            )  # (batch, output_dim)

        if self.sparse:
            # 稀疏路由：每个样本最多选择k个普通专家（不包括共享专家），再加上共享专家加权输出
            topk_score, topk_idx = expert_gate_scores.topk(self.k, dim=-1)  # (batch, k)
            output = torch.zeros(
                x.shape[0], self.output_dim, device=x.device, dtype=expert_outputs.dtype
            )
            for b in range(x.shape[0]):
                for i, expert_id in enumerate(topk_idx[b]):
                    output[b] += topk_score[b, i] * expert_outputs[b, expert_id]
            output = output + shared_output  # 加上共享专家部分
            return output
        else:
            # 致密路由：全部专家+共享专家加权求和
            if self.num_experts_shared > 0 and shared_outputs is not None:
                all_outputs = torch.cat(
                    [shared_outputs, expert_outputs], dim=1
                )  # (batch, num_experts_shared + num_experts, output_dim)
            else:
                all_outputs = expert_outputs  # (batch, num_experts, output_dim)
            gate_scores_unsq = gate_scores.unsqueeze(
                -1
            )  # (batch, num_experts_shared + num_experts, 1)
            output = (all_outputs * gate_scores_unsq).sum(dim=1)
            return output
