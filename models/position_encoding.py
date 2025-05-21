import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor):
        # x.shape = (N, B, D)
        self.pe = self.pe.to(x.device)
        return x + self.pe[:x.size(0), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()

        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.positional_encoding.weight, -0.1, 0.1)

    def forward(self, x):
        # x.shape = (N, B, D)
        N, _, _ = x.size()
        positions = torch.arange(N, device=x.device).unsqueeze(1)
        pos_encoding = self.positional_encoding(positions).transpose(0, 1)
        return x + pos_encoding

class Rotator:
    """根据hidden_dim，和position_ids 生成对应的旋转位置编码, 和论文中定义略有不同，一个个二维的子空间被
    分割到了前后两部分，分别进行旋转，然后拼接起来
    """
    def __init__(self, D: int, N: int):
        """ position_ids: [seq_len], D 和单个头的hidden_dim对应 """
        base = 10000
        d = D / 2
        B = base ** (1/d)
        theta_base = 1.0 / (B ** (torch.arange(0, d)))    # 等比数列， $\Theta$
        position_ids = torch.arange(0, N)
        thetas = position_ids.outer(theta_base)  # [seq_len, D/2]
        full_thetas = torch.cat((thetas, thetas), dim=-1)  # [seq_len, D]
        self.cos = full_thetas.cos()
        self.sin = full_thetas.sin()

    def rotate(self, x: torch.Tensor):
        """ trick1
        x: [bs, num_attention_heads, seq_len, D]
        q: [bs, num_attention_heads, seq_len, D]
        cos: [seq_len, D]
        [x,y] @ [[cos, sin], [-sin, cos]] = [x*cos-y*sin, ycos+x*sin] =[x,y]*cos+[-y, x]*sin
        """
        return x * self.cos + Rotator.reverse_half(x) * self.sin

    @staticmethod
    def reverse_half(q: torch.Tensor):
        """ q: [bs, num_attention_heads, seq_len, D] trick2 """
        u = q[..., : q.shape[-1] // 2]
        v = q[..., q.shape[-1] // 2:]
        return torch.cat((-v, u), dim=-1)
