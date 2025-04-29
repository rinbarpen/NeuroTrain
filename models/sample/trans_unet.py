import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x):
        x = self.proj(x)
        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])  # Self-Attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scaled Dot-Product Attention

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)  # Query
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)  # Key
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)  # Value

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y): # x: decoder (query), y: encoder (key, value)
        B, N, C = x.shape  # B: batch_size, N: number of patches, C: embedding dimension
        M = y.shape[1]  # M: number of patches in encoder feature map (y)
        H = self.num_heads

        q = self.to_q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3) # (B, H, N, C//H)
        k = self.to_k(y).reshape(B, M, H, C // H).permute(0, 2, 1, 3) # (B, H, M, C//H)
        v = self.to_v(y).reshape(B, M, H, C // H).permute(0, 2, 1, 3) # (B, H, M, C//H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_cross_attn=False, cross_attn_dim=None, num_heads=8):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + cross_attn_dim, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + cross_attn_dim, out_channels)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossAttention(dim=out_channels, num_heads=num_heads)  # output size must match up conv input size

    def forward(self, x1, x2, transformer_output=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if self.use_cross_attn:
            B, C, H, W = x.shape
            # Reshape x for cross-attention
            x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # [B, N, C]
            transformer_output_used = transformer_output # we assume transformer output is already in [B, N, C]
            x_attended = self.cross_attn(x_reshaped, transformer_output_used)  # Apply cross-attention
            x_attended = x_attended.reshape(B, H, W, C).permute(0, 3, 1, 2)
            x = x_attended  # Replace the initial CNN feature map with the attended one


        return self.conv(x) # apply convolution at the end


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size=256, patch_size=16, emb_size=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 bilinear=True, use_cross_attn=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.patch_embedding = PatchEmbedding(n_channels, patch_size, emb_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderLayer(dim=emb_size, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(emb_size)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear ,use_cross_attn, cross_attn_dim = emb_size,num_heads=num_heads)
        self.up2 = Up(512 // factor, 256 // factor, bilinear,use_cross_attn, cross_attn_dim = emb_size,num_heads=num_heads)
        self.up3 = Up(256 // factor, 128 // factor, bilinear,use_cross_attn, cross_attn_dim = emb_size,num_heads=num_heads)
        self.up4 = Up(128, 64, bilinear) # no cross attention in the last layer

        self.outc = OutConv(64, n_classes)
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size


    def forward(self, x):
        # CNN Encoder
        x1 = self.inc(x) #x1: [1, 64, 256, 256]
        x2 = self.down1(x1) #x2: [1, 128, 128, 128]
        x3 = self.down2(x2) #x3: [1, 256, 64, 64]
        x4 = self.down3(x3) #x4: [1, 512, 32, 32]
        x5 = self.down4(x4) #x5: [1, 512, 16, 16]

        # Transformer Encoder
        B, C, H, W = x.shape
        x_patches = self.patch_embedding(x) # [B, num_patches, emb_size]  -- emb_size is the hidden dimension of the transformer -- C
        x_patches = x_patches + self.pos_embed
        x_patches = self.pos_drop(x_patches)
        transformer_output = self.transformer_encoder(x_patches) # Apply transformer layers -- [B, num_patches, emb_size]
        transformer_output = self.norm(transformer_output)


        # CNN Decoder with Cross-Attention:  Upsample, concatenate encoder features, apply cross-attention, and apply convolution.
        x = self.up1(x5, x4, transformer_output)  # Apply cross attention using the transformer output
        x = self.up2(x, x3, transformer_output) # Apply cross attention using the transformer output
        x = self.up3(x, x2, transformer_output)   # Apply cross attention using the transformer output
        x = self.up4(x, x1)   # Apply upsampling and concatenation without cross-attention

        logits = self.outc(x)  # Final convolution to get the segmentation output.
        return logits