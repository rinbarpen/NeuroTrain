import torch
from torch import nn

def compute_load_balance_loss(gate_scores, num_experts):
    """
    计算负载均衡损失（Load Balancing Loss）
    
    使用变异系数的平方（CV²）来衡量负载分布的均匀性。
    值越小表示负载分布越均匀。
    
    Args:
        gate_scores: (..., num_experts) 门控分数，最后一维是专家维度
        num_experts: 专家数量
    
    Returns:
        load_balance_loss: 标量损失值
    """
    # 计算每个专家接收的总分数（作为负载的代理）
    # gate_scores shape: (..., num_experts)
    # 对所有非专家维度求和
    if gate_scores.dim() > 1:
        expert_load = gate_scores.sum(dim=tuple(range(gate_scores.dim() - 1)))  # (num_experts,)
    else:
        expert_load = gate_scores  # 如果已经是1维，直接使用
    
    # 计算负载的变异系数（coefficient of variation）
    # CV = std / mean, CV² 作为损失
    load_mean = expert_load.mean()
    
    if load_mean > 1e-8:  # 避免除零
        load_std = expert_load.std()
        cv_squared = (load_std / load_mean) ** 2
    else:
        cv_squared = torch.tensor(0.0, device=gate_scores.device, dtype=gate_scores.dtype)
    
    return cv_squared


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
        load_balance=False,
        load_balance_weight=0.01,
    ):
        """
        hidden_dim: 输入向量维数
        num_experts: 专家数量
        expert_hidden_dim: 专家mlp隐藏层, 若None则同input_dim
        dropout: Dropout概率，应用于每个专家内部
        load_balance: 是否启用负载均衡损失
        load_balance_weight: 负载均衡损失的权重
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance = load_balance
        self.load_balance_weight = load_balance_weight
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
        
        Returns:
            output: 输出张量
            aux_loss: 辅助损失（如果启用负载均衡），否则为0.0
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
        
        # 计算负载均衡损失
        aux_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        if self.load_balance:
            aux_loss = compute_load_balance_loss(gate_scores, self.num_experts) * self.load_balance_weight
        
        if return_middle_outputs:
            return output, aux_loss, {
                'expert_outputs': expert_outputs,
                'gate_scores': gate_scores,
            }
        else:
            return output, aux_loss


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
        load_balance=False,
        load_balance_weight=0.01,
    ):
        """
        hidden_dim: 输入向量维数
        num_experts: 专家数量
        expert_hidden_dim: 专家mlp隐藏层, 若None则同input_dim
        k: 每个输入选择的专家个数
        dropout: Dropout概率，应用于每个专家内部
        load_balance: 是否启用负载均衡损失
        load_balance_weight: 负载均衡损失的权重
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        self.load_balance = load_balance
        self.load_balance_weight = load_balance_weight
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
        
        Returns:
            output: 输出张量
            aux_loss: 辅助损失（如果启用负载均衡），否则为0.0
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
        
        # 计算负载均衡损失
        # 对于稀疏路由，我们使用topk_scores来计算负载
        aux_loss = torch.tensor(0.0, device=out.device, dtype=out.dtype)
        if self.load_balance:
            # 创建一个mask，标记哪些专家被选中
            expert_mask = torch.zeros_like(gate_scores)  # (B * seq_len, num_experts)
            expert_mask.scatter_(1, topk_indices, topk_scores)  # 使用topk_scores作为权重
            aux_loss = compute_load_balance_loss(expert_mask, self.num_experts) * self.load_balance_weight
        
        if return_middle_outputs:
            return out, aux_loss, {
                'expert_outputs': expert_outputs,
                'topk_scores': topk_scores,
                'topk_indices': topk_indices,
                'gate_scores': gate_scores,
            }
        else:
            return out, aux_loss


class VisionMoELayer(MoESparseLayer):
    """
    VisionMoELayer 即 MoESparseLayer，专为视觉任务配置。
    """
    def __init__(self, hidden_dim, num_experts=32, expert_hidden_dim=None, k=16, dropout=0.1,
                 load_balance=False, load_balance_weight=0.01):
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            load_balance=load_balance,
            load_balance_weight=load_balance_weight
        )

if __name__ == "__main__":
    x = torch.randn(2, 197, 768)
    
    # 测试不带负载均衡
    model = VisionMoELayer(hidden_dim=768, num_experts=32, expert_hidden_dim=None, k=16, dropout=0.1)
    y, aux_loss = model(x)
    print(f"Output shape: {y.shape}, Aux loss: {aux_loss}")
    
    # 测试带负载均衡
    model_lb = VisionMoELayer(hidden_dim=768, num_experts=32, expert_hidden_dim=None, k=16, 
                              dropout=0.1, load_balance=True, load_balance_weight=0.01)
    y_lb, aux_loss_lb = model_lb(x)
    print(f"Output shape (with load balance): {y_lb.shape}, Aux loss: {aux_loss_lb}")