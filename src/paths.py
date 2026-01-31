"""
Unified path resolution for any module.
Resolves project root, output dir, pretrained dir, data root, run dir;
prefers config when loaded, otherwise falls back to constants.
"""
from pathlib import Path

from src.constants import (
    DATASET_ROOT_DIR,
    OUTPUT_DIR,
    PRETRAINED_MODEL_DIR,
    RUN_DIR,
)


def _safe_get_config():
    """Return config dict if loaded; empty dict otherwise."""
    try:
        from src.config import get_config
        c = get_config()
        return c if c is not None else {}
    except Exception:
        return {}


def get_project_root() -> Path:
    """Project root: NEUROTRAIN_ROOT env, or parent of src, or cwd."""
    import os
    root = os.environ.get("NEUROTRAIN_ROOT")
    if root:
        return Path(root).resolve()
    try:
        # Path of this file: .../src/paths.py -> parent.parent is project root
        this_file = Path(__file__).resolve()
        candidate = this_file.parent.parent
        if (candidate / "pyproject.toml").exists() or (candidate / "src").is_dir():
            return candidate
    except Exception:
        pass
    return Path.cwd()


def get_output_dir() -> Path:
    """Output directory; from config if set, else constants."""
    c = _safe_get_config()
    out = c.get("output_dir")
    if out:
        return Path(out).resolve()
    return Path(OUTPUT_DIR).resolve()


def get_pretrained_dir() -> Path:
    """Pretrained model cache; from config.model.cache_dir if set, else constants."""
    c = _safe_get_config()
    cache = c.get("model", {}).get("cache_dir")
    if cache:
        return Path(cache).resolve()
    return Path(PRETRAINED_MODEL_DIR).resolve()


def get_data_root() -> Path:
    """Dataset root; from config.dataset.root_dir if set, else constants."""
    c = _safe_get_config()
    root = c.get("dataset", {})
    if isinstance(root, dict):
        root = root.get("root_dir")
    elif isinstance(root, str):
        pass
    else:
        root = None
    if root:
        return Path(root).resolve()
    return Path(DATASET_ROOT_DIR).resolve()


def get_run_dir() -> Path:
    """Run directory; from config if set, else constants."""
    c = _safe_get_config()
    run = c.get("run_dir") or c.get("output_dir")
    if run:
        return Path(run).resolve()
    return Path(RUN_DIR).resolve()
