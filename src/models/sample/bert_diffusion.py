from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.constants import PRETRAINED_MODEL_DIR


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000.0) / (half_dim - 1))
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MAEDiffusion(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_patches = math.ceil(latent_dim / patch_size)
        self.total_dim = self.num_patches * patch_size

        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.cond_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size),
        )
        self.output_proj = nn.Linear(self.total_dim, latent_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent_tokens = self._to_tokens(latent)
        cond_tokens = self._to_tokens(condition)

        tokens = self.patch_embed(latent_tokens) + self.cond_embed(cond_tokens)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        encoded = self.encoder(tokens)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape(latent.size(0), -1)
        decoded = self.output_proj(decoded)
        return decoded

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.total_dim:
            pad = self.total_dim - x.size(-1)
            if pad > 0:
                x = F.pad(x, (0, pad))
            else:
                x = x[..., : self.total_dim]
        return x.view(x.size(0), self.num_patches, self.patch_size)


@dataclass
class DiffusionOutput:
    loss: torch.Tensor
    preds: torch.Tensor
    targets: torch.Tensor
    noisy_latents: torch.Tensor
    timesteps: torch.Tensor


class BertDiffusionModel(nn.Module):
    """
    基于 CoBERT 文本编码器 + MAE Diffusion 的示例模型。

    使用 CoBERT 生成文本条件，利用 Masked Auto-Encoder 风格网络预测噪声。
    """

    def __init__(
        self,
        text_encoder_name: str = "BAAI/CoBERT-base",
        *,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cache_dir: str | None = None,
        device: torch.device | str | None = None,
        freeze_text_encoder: bool = False,
        mae_patch_size: int = 16,
        mae_embed_dim: int | None = None,
        mae_depth: int = 6,
        mae_num_heads: int = 8,
        mae_mlp_ratio: float = 4.0,
        mae_dropout: float = 0.0,
        load_weights: bool = True,
        weights_path: str | Path | None = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BertDiffusionModel 需要 transformers 库，请先安装：pip install transformers"
            ) from exc

        self.device = torch.device(device) if device is not None else _default_device()
        self.text_encoder_name = text_encoder_name
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir=cache_dir)
        nn.Module.to(self.text_encoder, self.device)

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps

        condition_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(condition_dim, latent_dim)
        self.time_embed = SinusoidalTimeEmbedding(latent_dim)
        self.time_proj = nn.Linear(latent_dim, latent_dim)
        self.noise_predictor = MAEDiffusion(
            latent_dim,
            patch_size=mae_patch_size,
            embed_dim=mae_embed_dim or hidden_dim,
            depth=mae_depth,
            num_heads=mae_num_heads,
            mlp_ratio=mae_mlp_ratio,
            dropout=mae_dropout,
        )
        self.mse = nn.MSELoss()

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]))
        self.betas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.alphas_cumprod_prev: torch.Tensor

        self.logger = logging.getLogger(self.__class__.__name__)

        default_weights_path = Path(PRETRAINED_MODEL_DIR) / "bert_diffusion_mae.pt"
        self.weights_path = Path(weights_path) if weights_path is not None else default_weights_path

        self.to(self.device)

        if load_weights:
            self._load_weights(self.weights_path, strict=strict_load)

    def forward(
        self,
        texts: Sequence[str],
        *,
        timesteps: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        batch_size = len(texts)
        device = self.device

        if latents is None:
            latents = torch.randn(batch_size, self.latent_dim, device=device)
        if timesteps is None:
            timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(latents)

        alpha_t = self.alphas_cumprod[timesteps].unsqueeze(-1)
        sigma_t = torch.sqrt(1 - alpha_t)
        noisy_latents = torch.sqrt(alpha_t) * latents + sigma_t * noise

        text_embeds = self._encode_text(texts)
        time_embeds = self.time_proj(self.time_embed(timesteps))
        condition = self.text_proj(text_embeds) + time_embeds

        pred_noise = self.noise_predictor(noisy_latents, condition)
        loss = self.mse(pred_noise, noise)

        if not return_dict:
            return loss

        return DiffusionOutput(
            loss=loss,
            preds=pred_noise,
            targets=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def sample(self, texts: Sequence[str], steps: int | None = None) -> torch.Tensor:
        steps = steps or self.diffusion_steps
        device = self.device
        batch_size = len(texts)

        latents = torch.randn(batch_size, self.latent_dim, device=device)
        text_embeds = self._encode_text(texts)
        condition = self.text_proj(text_embeds)

        for step in reversed(range(steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            time_embeds = self.time_proj(self.time_embed(t))
            cond = condition + time_embeds
            pred_noise = self.noise_predictor(latents, cond)

            beta_t = self.betas[t].unsqueeze(-1)
            alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
            alpha_prev = self.alphas_cumprod_prev[t].unsqueeze(-1)
            mean = (1 / torch.sqrt(alpha_t)) * (latents - beta_t / torch.sqrt(1 - alpha_t) * pred_noise)

            if step > 0:
                noise = torch.randn_like(latents)
                variance = torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * beta_t)
                latents = mean + variance * noise
            else:
                latents = mean

        return latents

    def _encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.text_encoder(**encoded, return_dict=True)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        return pooled

    def _load_weights(self, path: Path, *, strict: bool) -> None:
        if not path.exists():
            self.logger.info(f"No weights found at {path}, skipping load.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict", "module"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            self.logger.warning(f"Unsupported checkpoint format from {path}, expect dict.")
            return

        checkpoint = {
            k.replace("module.", "", 1): v for k, v in checkpoint.items()
        }

        missing, unexpected = self.load_state_dict(checkpoint, strict=strict)
        if missing:
            self.logger.warning(f"Missing keys when loading {path}: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys when loading {path}: {unexpected}")
        if not missing and not unexpected:
            self.logger.info(f"Successfully loaded weights from {path}.")

