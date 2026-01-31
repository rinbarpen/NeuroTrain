"""
Pretrained model Manager: list known models, resolve key to local path, download if missing.
Uses get_pretrained_dir() and download_and_save() for cache and download.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.paths import get_pretrained_dir

logger = logging.getLogger(__name__)

_PROVIDERS = ("torchvision", "timm", "huggingface")


def _cache_subpath(key: str) -> Path:
    """Key to cache subpath: HF org/repo -> dir with fullwidth slash to avoid path sep."""
    if "/" in key and "\\" not in key:
        return get_pretrained_dir() / key.replace("/", "\uFF0F")  # fullwidth solidus
    return get_pretrained_dir() / key


def get_pretrained_path(key: str, provider: Optional[str] = None) -> Path:
    """
    Resolve pretrained model key to local path (directory or file). Download if missing.

    Args:
        key: Model identifier (e.g. 'resnet50', 'vit_b_16', 'org/repo' for HF).
        provider: Optional 'torchvision' | 'timm' | 'huggingface'; if None, try all.

    Returns:
        Path to cache directory (or existing file) for that key.
    """
    cache_root = get_pretrained_dir()
    sub = _cache_subpath(key)
    if sub.exists():
        return sub

    from src.download_pretrained import download_and_save

    save_dir = download_and_save(key, desired_provider=provider)
    return Path(save_dir)


def list_known_models(provider: Optional[str] = None) -> list[str]:
    """
    List known model keys for a provider (or all supported for torchvision).

    Args:
        provider: 'torchvision' returns TORCHVISION_MODEL_MAP keys;
                  'timm' / 'huggingface' return [] for now.

    Returns:
        List of model key strings.
    """
    if provider is None or provider == "torchvision":
        from src.download_pretrained import TORCHVISION_MODEL_MAP
        return sorted(TORCHVISION_MODEL_MAP.keys())
    if provider in ("timm", "huggingface"):
        return []
    raise ValueError(f"Unknown provider: {provider}. Use one of {_PROVIDERS}")


def list_local() -> list[str]:
    """
    List keys that exist under get_pretrained_dir() (direct children: dirs or files).

    Returns:
        List of names (relative to cache root).
    """
    root = get_pretrained_dir()
    if not root.exists():
        return []
    out = []
    for p in root.iterdir():
        if p.name.startswith("."):
            continue
        out.append(p.name.replace("\uFF0F", "/"))  # fullwidth back to slash
    return sorted(out)


def resolve_pretrained_to_file(path: Path) -> Optional[Path]:
    """
    If path is a directory, find first weight file (*.pt, *.pth, pytorch_model.bin).
    If path is a file, return as-is. Otherwise return None.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    for name in ("pytorch_model.bin", "model.safetensors"):
        f = path / name
        if f.exists():
            return f
    for ext in ("*.pt", "*.pth"):
        for f in path.glob(ext):
            return f
    return None
