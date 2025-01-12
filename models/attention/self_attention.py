import torch
from torch import nn


def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == "all":  # 全局注意力，只关注下三角（只能看到之前的token）
        b = torch.tril(torch.ones([n, n]), diagonal=-1)
    elif attn_mode == "local":  # 局部注意力，只关注下三角（只能看到之前的token）
        bandwidth = local_attn_ctx
        ctx = n - 1 if n < bandwidth else bandwidth - 1
        b = torch.tril(torch.ones([n, n]), diagonal=ctx)
    elif attn_mode == "strided":  # 稀疏注意力，只关注下三角（只能看到之前的token）
        stride = local_attn_ctx
        x = torch.reshape(
            torch.range(0, n - 1, dtype=torch.int32), [n, 1]
        )
        y = x.t()
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod((q - k), stride).int(), torch.zeros([n, n]))
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError("Not yet implemented!")
    b = torch.reshape(b, [1, 1, n, n])
    return b
