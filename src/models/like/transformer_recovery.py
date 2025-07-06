from torch import nn
import torch
import torch.nn.functional as F
from typing import Literal, Optional
from einops import einops, einsum, rearrange
from ..attention.attention_mask import get_attn_mask
from ..embedding import PatchEmbeddingWithPE, EmbeddingWithPE
from ..norm.RMSNorm import RMSNorm

def pair(x: int|tuple[int, int]):
    if isinstance(x, int):
        x = (x, x)
    return x

class _MLP(nn.Module):
    def __init__(self, embed_dim: int, r: int, mlp_dropout: float=0.2, mlp_act=nn.LeakyReLU):
        super(_MLP, self).__init__()

        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * r),
            nn.Dropout(mlp_dropout),
            mlp_act(inplace=True),
            nn.Linear(embed_dim * r, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


AttentionType = Literal['mha', 'gqa', 'mqa']

class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False, alpha_v: Optional[float]=None, use_linear=False):
        super(_MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        if use_linear and attn_type != 'mha':
            attn_type = 'mha'
            print(f'switch attn_type to mha(origin is {attn_type})')
            if share_kv:
                share_kv = False
                print('turn off share_kv')

        self.use_linear = use_linear

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if attn_type == 'mha':
            self.num_groups = num_heads
        elif attn_type == 'mqa':
            self.num_groups = 1
        elif attn_type == 'gqa':
            self.num_groups = num_groups
            assert self.num_groups is not None, "num_groups must be set and greater than 0 with QGA"

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=qkv_bias)
        if share_kv:
            self.proj_k = nn.Linear(embed_dim, self.num_groups * self.head_dim, bias=qkv_bias)
        else:
            self.proj_k = self.proj_v

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.alpha = nn.Parameter(torch.randn(1)) if not alpha_v else alpha_v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        raw_v = v
        if self.use_linear:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
            
            attn_scores = einsum(k, v, 'b h n d, b h N d -> b h n N') * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            o = einsum(q, attn_weights, 'b h n d, b h n n -> b h n d')
            o = rearrange(o, 'b h n d -> b n (h d)')

            o = self.alpha * raw_v + o
            o = self.out_proj(o)
            return o, attn_weights.sum(dim=(1, 2))
        else:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
            k = rearrange(k, 'b n (g d) -> b g n d', g=self.num_groups)
            v = rearrange(v, 'b n (g d) -> b g n d', g=self.num_groups)
            
            q = rearrange(q, 'b (g h_per_g) n d -> b g h_per_g n d', g=self.num_groups)

            attn_scores = einsum(q, k, 'b g h q d, b g n d -> b g h q n') * self.scale
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            o = einsum(attn_weights, v, 'b g h q n, b g n d -> b g h q d')
            o = rearrange(o, 'b g h n d -> b n (g h d)')

            o = self.alpha * raw_v + o

            o = self.out_proj(o)
            return o, attn_weights.sum(dim=(1, 2))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, attn_dropout=0.2, *, share_kv=False, alpha_v=None, use_linear=False):
        super(MultiHeadAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=share_kv, alpha_v=alpha_v, use_linear=use_linear)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(x, x, x, mask=mask)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int|None=None, attn_type: AttentionType='mha',
     qkv_bias=True, attn_dropout=0.2, alpha_v=None):
        super(MultiHeadCrossAttention, self).__init__()

        self.attn = _MultiHeadAttention(embed_dim, num_heads, num_groups, attn_type=attn_type, qkv_bias=qkv_bias, attn_dropout=attn_dropout, share_kv=False, alpha_v=alpha_v, use_linear=False)
    
    def forward(self, qk: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None=None):
        return self.attn(qk, qk, v, mask=mask)

class ImageReconstruction(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int, patch_size: int|tuple[int, int]):
        super(ImageReconstruction, self).__init__()

        patch_size = pair(patch_size)

        self.patch_h, self.patch_w = patch_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(embed_dim, n_channels * self.patch_h * self.patch_w)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]):
        B, N, D = x.shape
        H_out, W_out = output_size

        x = self.proj(x)

        x = x.view(B, N, self.n_channels, self.patch_h, self.patch_w)
        N_h, N_w = H_out // self.patch_h, W_out // self.patch_w

        if N != N_h * N_w:
            raise ValueError(f"{N} != {N_h} * {N_w}")

        x = rearrange(x, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=N_h, nw=N_w)

        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_channels: int, embed_dim: int, num_heads: int, patch_size: int|tuple[int, int], max_num_patch: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, *, share_kv=False, attn_alpha_v=None, attn_use_linear=True):
        super(VisionTransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.pe = PatchEmbeddingWithPE(n_channels, embed_dim, patch_size=patch_size, max_num_patch=max_num_patch)
        self.attn = MultiHeadAttention(embed_dim, num_heads, num_groups=num_groups, attn_type=attn_type, attn_dropout=attn_dropout, qkv_bias=qkv_bias, share_kv=share_kv, alpha_v=attn_alpha_v, use_linear=attn_use_linear)
        self.attn_dropout = nn.Dropout(attn_out_dropout)
        self.mlp = _MLP(embed_dim=embed_dim, r=r, mlp_dropout=mlp_dropout, mlp_act=mlp_act)
        self.mlp_dropout = nn.Dropout(mlp_out_dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)

        self.unpack = ImageReconstruction(embed_dim, n_channels, patch_size)

    def forward(self, x: torch.Tensor):
        x = self.pe(x)
        attn_weights = []
        for _ in range(self.n_layers):
            x = self.norm1(x)
            x, attn_weight = self.attn(x)
            x = self.attn_dropout(x) + x
            x = self.norm2(x)
            x = self.mlp(x)
            x = self.mlp_dropout(x) + x

            attn_weights.append(attn_weight)

        x = self.unpack(x)
        return x, attn_weights

class CrossVisionTransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_channels: int, embed_dim: int, num_heads: int, patch_size: int|tuple[int, int], max_num_patch: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, attn_alpha_v=None):
        super(CrossVisionTransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.pe = PatchEmbeddingWithPE(n_channels, embed_dim, patch_size=patch_size, max_num_patch=max_num_patch)
        self.attn = MultiHeadCrossAttention(embed_dim, num_heads, num_groups=num_groups, attn_type=attn_type, attn_dropout=attn_dropout, qkv_bias=qkv_bias, alpha_v=attn_alpha_v)
        self.attn_dropout = nn.Dropout(attn_out_dropout)
        self.mlp = _MLP(embed_dim=embed_dim, r=r, mlp_dropout=mlp_dropout, mlp_act=mlp_act)
        self.mlp_dropout = nn.Dropout(mlp_out_dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)

        self.unpack = ImageReconstruction(embed_dim, n_channels, patch_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        x, y, z = self.pe(x), self.pe(y), self.pe(z)
        attn_weights = []
        for _ in range(self.n_layers):
            x = self.norm1(x)
            x, attn_weight = self.attn(x, y, z)
            x = self.attn_dropout(x) + x
            x = self.norm2(x)
            x = self.mlp(x)
            x = self.mlp_dropout(x) + x

            attn_weights.append(attn_weight)

        x = self.unpack(x)
        return x, attn_weights

# alpha: 
#   float: 1.0 to residual, 0.0 to disable
#          None -> nn.Parameter(torch.randn(1))
#   
class RecoveryNet(nn.Module):
    def __init__(self, mode: str=Literal['small', 'normal', 'large', 's', 'n', 'l'], image_size: int|tuple[int, int]=224, patch_size: int|tuple[int, int]=14, *, gate_or_linear='gate'):
        super(RecoveryNet, self).__init__()

        image_size = pair(image_size)
        patch_size = pair(patch_size)

        max_num_patch = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        encoder_config = {
            'n_layers': 4,
            'n_channels': 3,
            'embed_dim': 568,
            'num_heads': 8,
            'patch_size': patch_size,
            'max_num_patch': max_num_patch,
            'r': 4,
        }
        if mode.startswith('s'):
            self.encoder1 = VisionTransformerEncoder(**encoder_config)
            self.up1 = nn.Upsample(2)
            encoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.encoder2 = VisionTransformerEncoder(**encoder_config)
            self.up2 = nn.Upsample(2)
            decoder_config = encoder_config
            decoder_config['n_layers'] = encoder_config['n_layers'] * 2
            decoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.decoder = CrossVisionTransformerEncoder(**decoder_config)
        elif mode.startswith('n'):
            encoder_config['n_layers'] = 6
            encoder_config['num_heads'] = 12
            self.encoder1 = VisionTransformerEncoder(**encoder_config)
            self.up1 = nn.Upsample(2)
            encoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.encoder2 = VisionTransformerEncoder(**encoder_config)
            self.up2 = nn.Upsample(2)
            decoder_config = encoder_config
            decoder_config['n_layers'] = encoder_config['n_layers'] * 2
            decoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.decoder = CrossVisionTransformerEncoder(**decoder_config)
        elif mode.startswith('l'):
            encoder_config['n_layers'] = 8
            encoder_config['num_heads'] = 16
            self.encoder1 = VisionTransformerEncoder(**encoder_config)
            self.up1 = nn.Upsample(2)
            encoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.encoder2 = VisionTransformerEncoder(**encoder_config)
            self.up2 = nn.Upsample(2)
            decoder_config = encoder_config
            decoder_config['n_layers'] = encoder_config['n_layers'] * 2
            decoder_config['max_num_patch'] = encoder_config['max_num_patch'] * 4
            self.decoder = CrossVisionTransformerEncoder(**decoder_config)
        
        if gate_or_linear == 'gate':
            self.gate = nn.Parameter(torch.randn(1))
        elif gate_or_linear == 'linear':
            self.gate = nn.Linear(max_num_patch * 5, max_num_patch * 4)
        
        self.config = {}
        self.config['gate_or_linear'] = gate_or_linear

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x1, attn1 = self.encoder1(x)
        o1 = self.up1(x1)
        x2, attn2 = self.encoder2(o1)
        o2 = self.up2(x2)

        x3 = torch.concat([x1, x2], dim=1)
        x3 = self._gate(x3)
        o, o_attn = self.decoder(x3, o2)
        return o, (attn1, attn2, o_attn)

    def _gate(self, x):
        if self.config['gate_or_linear'] == 'gate':
            return self.gate * x
        elif self.config['gate_or_linear'] == 'linear':
            return self.gate(x)
