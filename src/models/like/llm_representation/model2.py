import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class EarlyQueryFusion(nn.Module):
    def __init__(self, base_model, early_layer=4, target_layer=20, hidden_size=4096, gate_init=0.1):
        super().__init__()
        self.base_model = base_model
        self.early_layer = early_layer
        self.target_layer = target_layer
        
        # 投影，把早层 hidden states 映射到语义空间
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # gate 系数，可学习
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # 保存 early states
        self._early_states = None

        # 注册 forward hook 来拿 early layer hidden states
        def save_early_states(module, input, output):
            self._early_states = output[0].detach()  # (batch, seq, hidden)

        self.base_model.model.layers[early_layer].register_forward_hook(save_early_states)

        # 替换 target_layer 的 self_attn 前向函数
        orig_attn_forward = self.base_model.model.layers[target_layer].self_attn.forward

        def fused_attn_forward(self_attn, hidden_states, *args, **kwargs):
            # 调用原始 attention forward，兼容不同返回长度
            attn_result = orig_attn_forward(hidden_states, *args, **kwargs)
            attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result

            if self._early_states is not None:
                B, L, H = hidden_states.size()

                # early states → query
                q_early = self.proj(self._early_states)  # (B, L, H)

                # 从模型配置读取 head 数，避免依赖具体实现细节
                cfg = getattr(self.base_model, 'config', None)
                if cfg is None and hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'config'):
                    cfg = self.base_model.model.config
                num_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'num_heads', 32))
                num_key_value_heads = getattr(cfg, 'num_key_value_heads', num_heads)
                head_dim = H // num_heads

                # 使用当前层的 k_proj 和 v_proj 获取 key 和 value
                k = self_attn.k_proj(hidden_states).view(B, L, num_key_value_heads, head_dim).transpose(1, 2)  # (B, Kv, L, D)
                v = self_attn.v_proj(hidden_states).view(B, L, num_key_value_heads, head_dim).transpose(1, 2)  # (B, Kv, L, D)

                # 调整 q_early 的维度以匹配 k 和 v（注意：Q 有 Nh 个头，而 K/V 只有 Kv 个头，GQA 需要重复）
                q_early = q_early.view(B, L, num_heads, head_dim).transpose(1, 2)  # (B, Nh, L, D)
                # 对齐精度到K/V的dtype，避免混合精度下溢出
                q_early = q_early.to(dtype=k.dtype)

                # 如果需要，将KV头重复以匹配Q头（GQA: Nh = Kv * g）
                if k.size(1) != q_early.size(1):
                    if q_early.size(1) % k.size(1) == 0:
                        repeat_factor = q_early.size(1) // k.size(1)
                        k = k.repeat_interleave(repeat_factor, dim=1)
                        v = v.repeat_interleave(repeat_factor, dim=1)
                    else:
                        # 兜底：近似重复并截断到目标头数（非典型配置）
                        repeat_factor = (q_early.size(1) + k.size(1) - 1) // k.size(1)
                        k = k.repeat_interleave(repeat_factor, dim=1)[:, :q_early.size(1), ...]
                        v = v.repeat_interleave(repeat_factor, dim=1)[:, :q_early.size(1), ...]

                # dot product attention（early→late）
                attn_scores = torch.matmul(q_early, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, Nh, L, L)
                attn_probs = attn_scores.softmax(dim=-1)
                early_output = torch.matmul(attn_probs, v)  # (B, Nh, L, D)

                # 调整 early_output 的维度以匹配 attn_output
                early_output = early_output.transpose(1, 2).contiguous().view(B, L, H)  # (B, L, H)

                # gated residual 融合（保持与原注意力输出dtype一致）
                attn_output = attn_output + self.gate.to(attn_output.dtype) * early_output

            # 按原始返回格式返回
            if isinstance(attn_result, tuple):
                return (attn_output,) + attn_result[1:]
            else:
                return attn_output

        # 使用 partial 绑定 self_attn
        import functools
        self.base_model.model.layers[target_layer].self_attn.forward = functools.partial(
            fused_attn_forward, self.base_model.model.layers[target_layer].self_attn
        )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

from src.config import PRETRAINED_MODEL_DIR

# ===== 使用例子 =====
# model_name = "meta-llama/Llama-2-7b-hf"  # 你可以换成别的
model_name = "mistralai/Mistral-7B-v0.1"  # 你可以换成别的
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=PRETRAINED_MODEL_DIR, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=PRETRAINED_MODEL_DIR, local_files_only=True)

# 包装
model = EarlyQueryFusion(base_model, early_layer=4, target_layer=20, hidden_size=4096)
# model = base_model

# inputs = tokenizer("What is the capital of France? The answer is", return_tensors="pt")
inputs = tokenizer("请详细描述一下巴黎圣母院的建筑特点，包括它的哥特式建筑风格、尖塔、飞扶壁、玫瑰窗等标志性元素。同时也请介绍它在历史上的重要地位和文化意义。", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
