import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, cast
from einops import rearrange
import math
import numpy as np


def to_tuple(x: Union[int, Tuple[int, ...]], ndim: int = 2) -> Tuple[int, ...]:
    """Convert int or tuple to tuple with specified dimensions."""
    if isinstance(x, int):
        return tuple([x] * ndim)
    elif isinstance(x, tuple):
        if len(x) == ndim:
            return x
        elif len(x) == 1:
            return tuple([x[0]] * ndim)
        else:
            raise ValueError(f"Cannot convert {x} to {ndim}D tuple")
    else:
        raise TypeError(f"Expected int or tuple, got {type(x)}")


class PatchEmbedding2D(nn.Module):
    """2D Patch Embedding."""
    def __init__(self, n_channels: int, embed_dim: int, patch_size: Union[int, Tuple[int, int]]):
        super().__init__()
        patch_size_tuple = to_tuple(patch_size, 2)
        self.patch_size = cast(Tuple[int, int], patch_size_tuple)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            n_channels, embed_dim,
            kernel_size=cast(Tuple[int, int], self.patch_size),
            stride=cast(Tuple[int, int], self.patch_size),
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, N, D) where N = (H//P) * (W//P)
        """
        x = self.proj(x)  # (B, D, H//P, W//P)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


class PatchEmbedding3D(nn.Module):
    """3D Patch Embedding."""
    def __init__(self, n_channels: int, embed_dim: int, patch_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        patch_size_tuple = to_tuple(patch_size, 3)
        self.patch_size = cast(Tuple[int, int, int], patch_size_tuple)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(
            n_channels, embed_dim,
            kernel_size=cast(Tuple[int, int, int], self.patch_size),
            stride=cast(Tuple[int, int, int], self.patch_size),
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            patches: (B, N, D) where N = (D//P) * (H//P) * (W//P)
        """
        x = self.proj(x)  # (B, D, D//P, H//P, W//P)
        B, D, D_p, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


class PatchEmbedding(nn.Module):
    """Unified Patch Embedding supporting both 2D and 3D."""
    def __init__(self, n_channels: int, embed_dim: int, patch_size: Union[int, Tuple[int, ...]], ndim: int = 2):
        super().__init__()
        self.ndim = ndim
        if ndim == 2:
            patch_size_2d = patch_size if isinstance(patch_size, (int, tuple)) and (isinstance(patch_size, int) or len(patch_size) <= 2) else to_tuple(patch_size, 2)
            self.embedding = PatchEmbedding2D(n_channels, embed_dim, cast(Union[int, Tuple[int, int]], patch_size_2d))
        elif ndim == 3:
            patch_size_3d = patch_size if isinstance(patch_size, (int, tuple)) and (isinstance(patch_size, int) or len(patch_size) <= 3) else to_tuple(patch_size, 3)
            self.embedding = PatchEmbedding3D(n_channels, embed_dim, cast(Union[int, Tuple[int, int, int]], patch_size_3d))
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, activation=nn.GELU):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = activation()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with self-attention."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_dropout,
            bias=qkv_bias,
            batch_first=True
        )
        self.dropout = nn.Dropout(proj_dropout)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, mlp_dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """MAE Encoder: processes only visible patches."""
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_learnable_pos: bool = False,
        max_num_patches: int = 10000
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional encoding
        if use_learnable_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, embed_dim))
        else:
            self.register_buffer('pos_embed', torch.zeros(1, max_num_patches, embed_dim))
            self._init_pos_embed()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias, attn_dropout, proj_dropout, mlp_dropout, norm_layer
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.use_learnable_pos = use_learnable_pos
    
    def _init_pos_embed(self):
        """Initialize positional embedding with sin/cos encoding."""
        num_patches = self.pos_embed.shape[1]
        grid_size = int(math.sqrt(num_patches))
        
        # Check if it's a perfect square (for 2D) or use 1D encoding
        if grid_size * grid_size == num_patches:
            # 2D positional encoding
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
        else:
            # 1D positional encoding for non-square cases (e.g., 3D)
            pos_embed = get_1d_sincos_pos_embed(self.embed_dim, num_patches)
        
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N_visible, D) - visible patches only
            mask: (B, N_visible) - optional attention mask
        Returns:
            encoded: (B, N_visible, D)
        """
        B, N, D = x.shape
        
        # Add positional embedding
        if self.use_learnable_pos:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            pos_embed = self.pos_embed[:, :N, :]
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        
        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    """MAE Decoder: reconstructs all patches from encoded visible patches and mask tokens."""
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_learnable_pos: bool = False,
        max_num_patches: int = 10000
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional encoding for all patches
        if use_learnable_pos:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, decoder_embed_dim))
        else:
            self.register_buffer('decoder_pos_embed', torch.zeros(1, max_num_patches, decoder_embed_dim))
            self._init_decoder_pos_embed()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, num_heads, mlp_ratio,
                qkv_bias, attn_dropout, proj_dropout, mlp_dropout, norm_layer
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(decoder_embed_dim)
        self.use_learnable_pos = use_learnable_pos
    
    def _init_decoder_pos_embed(self):
        """Initialize decoder positional embedding."""
        num_patches = self.decoder_pos_embed.shape[1]
        grid_size = int(math.sqrt(num_patches))
        
        # Check if it's a perfect square (for 2D) or use 1D encoding
        if grid_size * grid_size == num_patches:
            # 2D positional encoding
            pos_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim, grid_size)
        else:
            # 1D positional encoding for non-square cases (e.g., 3D)
            pos_embed = get_1d_sincos_pos_embed(self.decoder_embed_dim, num_patches)
        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        ids_keep: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N_visible, embed_dim) - encoded visible patches
            ids_restore: (B, N) - indices to restore full sequence
            ids_keep: (B, N_visible) - indices of kept patches
        Returns:
            decoded: (B, N, decoder_embed_dim)
        """
        B, N_visible, _ = x.shape
        N = ids_restore.shape[1]
        
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible, decoder_embed_dim)
        
        # Expand mask tokens
        mask_tokens = self.mask_token.repeat(B, N - N_visible, 1)  # (B, N_masked, decoder_embed_dim)
        
        # Concatenate visible patches and mask tokens
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_embed_dim)
        
        # Unshuffle: restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim))
        
        # Add positional embedding
        if self.use_learnable_pos:
            pos_embed = self.decoder_pos_embed[:, :N, :]
        else:
            pos_embed = self.decoder_pos_embed[:, :N, :]
        x_full = x_full + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x_full = block(x_full)
        
        x_full = self.norm(x_full)
        return x_full


class PatchReconstruction2D(nn.Module):
    """Reconstruct 2D image from patches."""
    def __init__(self, embed_dim: int, n_channels: int, patch_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.patch_size = to_tuple(patch_size, 2)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Linear(embed_dim, n_channels * self.patch_size[0] * self.patch_size[1])
    
    def forward(self, x: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - patch embeddings
            img_size: (H, W) - target image size
        Returns:
            img: (B, C, H, W)
        """
        B, N, D = x.shape
        H, W = img_size
        
        # Project to patch pixels
        x = self.proj(x)  # (B, N, C*P*P)
        x = x.view(B, N, self.n_channels, self.patch_size[0], self.patch_size[1])
        
        # Reshape to image
        N_h = H // self.patch_size[0]
        N_w = W // self.patch_size[1]
        
        if N != N_h * N_w:
            raise ValueError(f"Number of patches {N} != {N_h} * {N_w}")
        
        x = rearrange(x, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=N_h, nw=N_w)
        return x


class PatchReconstruction3D(nn.Module):
    """Reconstruct 3D image from patches."""
    def __init__(self, embed_dim: int, n_channels: int, patch_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.patch_size = to_tuple(patch_size, 3)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Linear(embed_dim, n_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
    
    def forward(self, x: torch.Tensor, img_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - patch embeddings
            img_size: (D, H, W) - target image size
        Returns:
            img: (B, C, D, H, W)
        """
        B, N, D = x.shape
        D_img, H, W = img_size
        
        # Project to patch pixels
        x = self.proj(x)  # (B, N, C*P*P*P)
        x = x.view(B, N, self.n_channels, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        
        # Reshape to image
        N_d = D_img // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]
        
        if N != N_d * N_h * N_w:
            raise ValueError(f"Number of patches {N} != {N_d} * {N_h} * {N_w}")
        
        x = rearrange(x, 'b (nd nh nw) c pd ph pw -> b c (nd pd) (nh ph) (nw pw)', nd=N_d, nh=N_h, nw=N_w)
        return x


class PatchReconstruction(nn.Module):
    """Unified Patch Reconstruction supporting both 2D and 3D."""
    def __init__(self, embed_dim: int, n_channels: int, patch_size: Union[int, Tuple[int, ...]], ndim: int = 2):
        super().__init__()
        self.ndim = ndim
        if ndim == 2:
            patch_size_2d = patch_size if isinstance(patch_size, (int, tuple)) and (isinstance(patch_size, int) or len(patch_size) <= 2) else to_tuple(patch_size, 2)
            self.reconstruction = PatchReconstruction2D(embed_dim, n_channels, cast(Union[int, Tuple[int, int]], patch_size_2d))
        elif ndim == 3:
            patch_size_3d = patch_size if isinstance(patch_size, (int, tuple)) and (isinstance(patch_size, int) or len(patch_size) <= 3) else to_tuple(patch_size, 3)
            self.reconstruction = PatchReconstruction3D(embed_dim, n_channels, cast(Union[int, Tuple[int, int, int]], patch_size_3d))
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    
    def forward(self, x: torch.Tensor, img_size: Tuple[int, ...]) -> torch.Tensor:
        return self.reconstruction(x, img_size)


def get_1d_sincos_pos_embed(embed_dim: int, num_positions: int) -> np.ndarray:
    """
    Create 1D sin/cos positional embeddings.
    Args:
        embed_dim: embedding dimension (must be divisible by 2)
        num_positions: number of positions
    Returns:
        pos_embed: (num_positions, embed_dim)
    """
    assert embed_dim % 2 == 0
    
    # Create position indices
    position = np.arange(num_positions, dtype=np.float32)[:, None]  # (num_positions, 1)
    
    # Create embedding
    pos_embed = np.zeros([num_positions, embed_dim], dtype=np.float32)
    
    # Generate position encodings
    dim_t = np.arange(embed_dim // 2, dtype=np.float32)
    dim_t = 10000 ** (2 * dim_t / embed_dim)
    
    pos = position / dim_t  # (num_positions, embed_dim//2)
    pos_embed[:, 0::2] = np.sin(pos)
    pos_embed[:, 1::2] = np.cos(pos)
    
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Create 2D sin/cos positional embeddings.
    Args:
        embed_dim: embedding dimension (must be divisible by 4)
        grid_size: grid size (assumed square)
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0
    
    # Create grid coordinates
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (H, W) and (H, W)
    
    # Flatten grids
    grid_w_flat = grid[0].flatten()  # (H*W,)
    grid_h_flat = grid[1].flatten()  # (H*W,)
    
    # Create embedding
    pos_embed = np.zeros([grid_size * grid_size, embed_dim], dtype=np.float32)
    
    # Generate position encodings for each dimension
    dim_t = np.arange(embed_dim // 4, dtype=np.float32)
    dim_t = 10000 ** (2 * dim_t / (embed_dim // 2))
    
    # X position encoding
    pos_x = grid_w_flat[:, None] / dim_t  # (H*W, embed_dim//4)
    pos_embed[:, 0::4] = np.sin(pos_x)
    pos_embed[:, 1::4] = np.cos(pos_x)
    
    # Y position encoding
    pos_y = grid_h_flat[:, None] / dim_t  # (H*W, embed_dim//4)
    pos_embed[:, 2::4] = np.sin(pos_y)
    pos_embed[:, 3::4] = np.cos(pos_y)
    
    return pos_embed


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random masking for MAE.
    Args:
        x: (B, N, D) - input patches
        mask_ratio: ratio of patches to mask
    Returns:
        x_masked: (B, N_visible, D) - visible patches
        ids_restore: (B, N) - indices to restore full sequence
        ids_keep: (B, N_visible) - indices of kept patches
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    
    # Random shuffle
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)  # (B, N)
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # (B, N)
    
    # Keep first len_keep patches
    ids_keep = ids_shuffle[:, :len_keep]  # (B, len_keep)
    
    # Extract visible patches
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    return x_masked, ids_restore, ids_keep


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for image reconstruction.
    Supports both 2D and 3D images.
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, ...]],
        patch_size: Union[int, Tuple[int, ...]],
        n_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_learnable_pos: bool = False,
        ndim: int = 2
    ):
        super().__init__()
        self.ndim = ndim
        self.n_channels = n_channels
        self.mask_ratio = mask_ratio
        
        # Determine image and patch sizes
        if ndim == 2:
            self.img_size = to_tuple(img_size, 2)
            self.patch_size = to_tuple(patch_size, 2)
            self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        elif ndim == 3:
            self.img_size = to_tuple(img_size, 3)
            self.patch_size = to_tuple(patch_size, 3)
            self.num_patches = (
                (self.img_size[0] // self.patch_size[0]) *
                (self.img_size[1] // self.patch_size[1]) *
                (self.img_size[2] // self.patch_size[2])
            )
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(n_channels, embed_dim, patch_size, ndim)
        
        # Encoder
        self.encoder = MAEEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            mlp_dropout=mlp_dropout,
            norm_layer=norm_layer,
            use_learnable_pos=use_learnable_pos,
            max_num_patches=self.num_patches
        )
        
        # Decoder
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            mlp_dropout=mlp_dropout,
            norm_layer=norm_layer,
            use_learnable_pos=use_learnable_pos,
            max_num_patches=self.num_patches
        )
        
        # Reconstruction head
        self.reconstruction = PatchReconstruction(decoder_embed_dim, n_channels, patch_size, ndim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input image
                - 2D: (B, C, H, W)
                - 3D: (B, C, D, H, W)
            mask_ratio: Masking ratio (overrides self.mask_ratio if provided)
            return_mask: Whether to return the mask
        Returns:
            reconstructed: Reconstructed image (same shape as input)
            mask: (optional) Mask tensor (B, N) where 1 indicates visible patches
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Patch embedding
        patches = self.patch_embed(x)  # (B, N, D)
        
        # Random masking
        patches_visible, ids_restore, ids_keep = random_masking(patches, mask_ratio)
        
        # Encode visible patches
        encoded = self.encoder(patches_visible)  # (B, N_visible, embed_dim)
        
        # Decode all patches
        decoded = self.decoder(encoded, ids_restore, ids_keep)  # (B, N, decoder_embed_dim)
        
        # Reconstruct image
        reconstructed = self.reconstruction(decoded, self.img_size)
        
        if return_mask:
            # Create mask: 1 for visible, 0 for masked
            mask = torch.zeros(patches.shape[0], patches.shape[1], device=patches.device)
            mask.scatter_(1, ids_keep, 1)
            return reconstructed, mask
        
        return reconstructed
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass through encoder only (for feature extraction).
        Args:
            x: Input image
            mask_ratio: Masking ratio
        Returns:
            encoded: Encoded visible patches (B, N_visible, embed_dim)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        patches = self.patch_embed(x)
        patches_visible, _, _ = random_masking(patches, mask_ratio)
        encoded = self.encoder(patches_visible)
        return encoded
    
    def forward_decoder(self, encoded: torch.Tensor, ids_restore: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder only.
        Args:
            encoded: Encoded visible patches (B, N_visible, embed_dim)
            ids_restore: Indices to restore full sequence (B, N)
            ids_keep: Indices of kept patches (B, N_visible)
        Returns:
            reconstructed: Reconstructed image
        """
        decoded = self.decoder(encoded, ids_restore, ids_keep)
        reconstructed = self.reconstruction(decoded, self.img_size)
        return reconstructed

if __name__ == "__main__":
    import sys
    
    print("Testing MAE Model...")
    print("=" * 50)
    
    # Test 2D MAE
    print("\n[Test 1] Testing 2D MAE Model")
    print("-" * 50)
    try:
        model_2d = MAE(
            img_size=(224, 224),
            patch_size=16,
            n_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mask_ratio=0.75,
            ndim=2
        )
        
        # Create dummy 2D image
        batch_size = 2
        x_2d = torch.randn(batch_size, 3, 224, 224)
        print(f"Input shape: {x_2d.shape}")
        
        # Forward pass
        reconstructed_2d, mask_2d = model_2d(x_2d, return_mask=True)
        print(f"Reconstructed shape: {reconstructed_2d.shape}")
        print(f"Mask shape: {mask_2d.shape}")
        print(f"Mask ratio: {1 - mask_2d.sum() / mask_2d.numel():.3f}")
        
        # Test encoder only
        encoded_2d = model_2d.forward_encoder(x_2d)
        print(f"Encoded shape: {encoded_2d.shape}")
        
        assert reconstructed_2d.shape == x_2d.shape, "2D reconstruction shape mismatch"
        print("✓ 2D MAE test passed!")
        
    except Exception as e:
        print(f"✗ 2D MAE test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3D MAE
    print("\n[Test 2] Testing 3D MAE Model")
    print("-" * 50)
    try:
        model_3d = MAE(
            img_size=(32, 64, 64),
            patch_size=(4, 8, 8),
            n_channels=1,
            embed_dim=512,
            depth=8,
            num_heads=8,
            decoder_embed_dim=256,
            decoder_depth=4,
            decoder_num_heads=8,
            mask_ratio=0.75,
            ndim=3
        )
        
        # Create dummy 3D image
        batch_size = 2
        x_3d = torch.randn(batch_size, 1, 32, 64, 64)
        print(f"Input shape: {x_3d.shape}")
        
        # Forward pass
        reconstructed_3d, mask_3d = model_3d(x_3d, return_mask=True)
        print(f"Reconstructed shape: {reconstructed_3d.shape}")
        print(f"Mask shape: {mask_3d.shape}")
        print(f"Mask ratio: {1 - mask_3d.sum() / mask_3d.numel():.3f}")
        
        # Test encoder only
        encoded_3d = model_3d.forward_encoder(x_3d)
        print(f"Encoded shape: {encoded_3d.shape}")
        
        assert reconstructed_3d.shape == x_3d.shape, "3D reconstruction shape mismatch"
        print("✓ 3D MAE test passed!")
        
    except Exception as e:
        print(f"✗ 3D MAE test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test different mask ratios
    print("\n[Test 3] Testing Different Mask Ratios")
    print("-" * 50)
    try:
        for mask_ratio in [0.5, 0.75, 0.9]:
            reconstructed = model_2d(x_2d, mask_ratio=mask_ratio)
            assert reconstructed.shape == x_2d.shape, f"Shape mismatch with mask_ratio={mask_ratio}"
            print(f"✓ Mask ratio {mask_ratio} test passed!")
    except Exception as e:
        print(f"✗ Mask ratio test failed: {e}")
        sys.exit(1)
    
    # Test patch size variations
    print("\n[Test 4] Testing Different Patch Sizes")
    print("-" * 50)
    try:
        # Test 2D with different patch sizes
        for patch_size in [8, 16, (14, 14)]:
            model = MAE(
                img_size=(224, 224),
                patch_size=patch_size,
                n_channels=3,
                embed_dim=384,
                depth=6,
                num_heads=6,
                ndim=2
            )
            x = torch.randn(1, 3, 224, 224)
            reconstructed = model(x)
            assert reconstructed.shape == x.shape, f"Shape mismatch with patch_size={patch_size}"
            print(f"✓ Patch size {patch_size} test passed!")
        
        # Test 3D with different patch sizes
        for patch_size in [4, (4, 8, 8), (8, 16, 16)]:
            model = MAE(
                img_size=(32, 64, 64),
                patch_size=patch_size,
                n_channels=1,
                embed_dim=256,
                depth=4,
                num_heads=4,
                ndim=3
            )
            x = torch.randn(1, 1, 32, 64, 64)
            reconstructed = model(x)
            assert reconstructed.shape == x.shape, f"Shape mismatch with patch_size={patch_size}"
            print(f"✓ 3D patch size {patch_size} test passed!")
            
    except Exception as e:
        print(f"✗ Patch size test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test gradient flow
    print("\n[Test 5] Testing Gradient Flow")
    print("-" * 50)
    try:
        model_2d.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        reconstructed = model_2d(x)
        loss = F.mse_loss(reconstructed, x)
        loss.backward()
        
        # Check if gradients exist
        has_grad = False
        for param in model_2d.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"
        print(f"Loss: {loss.item():.4f}")
        print("✓ Gradient flow test passed!")
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    
    # Usage examples
    print("\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    print("1. Direct run: python src/models/like/recon/mae.py")
    print("2. Module run: python -m src.models.like.recon.mae")
    print("3. Import and use:")
    print("   from src.models.like.recon.mae import MAE")
    print("   model = MAE(img_size=(224, 224), patch_size=16, ndim=2)")