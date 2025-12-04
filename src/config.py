# runs/{task_name}/{timestamp}/{train|test|predict}/{classification|detection|segmentation|weights|..}
import json
import math
import yaml
import toml
from pathlib import Path
import logging
from collections.abc import Mapping
from typing import Any
from src.constants import SINGLE_CONFIG_DIR

# GLOBAL CONSTANTS
TRAIN_MODE = 1
TEST_MODE  = 2
PREDICT_MODE = 4
ALL_MODE = TRAIN_MODE | TEST_MODE | PREDICT_MODE

ALL_METRIC_LABELS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']
ALL_STYLES = ['cyan', 'magenta', 'green', 'yellow', 'blue', 'red']
DEFAULT_STYLE_CONFIG = {
    'default': {
        'metric_table': tuple(ALL_STYLES),
        'summary_table': tuple(ALL_STYLES),
    }
}

CONFIG: dict = {}

def is_verbose():
    return CONFIG['private']['verbose']
def is_train():
    return CONFIG['private']['mode'] & TRAIN_MODE == TRAIN_MODE 
def is_test():
    return CONFIG['private']['mode'] & TEST_MODE == TEST_MODE
def is_predict():
    return CONFIG['private']['mode'] & PREDICT_MODE == PREDICT_MODE

def is_test_after_training():
    return is_train() and is_test()

def is_predict_after_training():
    return is_train() and is_predict()

def wandb_is_available():
    return CONFIG['private']['wandb']

def set_config(config: dict):
    global CONFIG

    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        else:  # numbers or strings, for exapmle.
            return obj  # No conversion needed

    CONFIG = convert_to_native_types(config)

def get_config() -> dict:
    return CONFIG

def get_config_value(key_field: str, split: str='.', default=None):
    keys = key_field.split(split)
    value = CONFIG
    try:
        for key in keys:
            value = value[key]
    except KeyError:
        print(f'{key_field} is not in config. Returning default value: {default}')
        return default
    return value

def load_config(filename: Path) -> dict:
    match filename.suffix:
        case '.json':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = json.load(f)
        case '.yaml'|'.yml':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        case '.toml':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = toml.load(f)
        case _:
            raise ValueError(f'Unsupported config file format: {filename.suffix}')

    logging.info(f'Loading config from {filename}')
    return config

def save_config(filename: Path, config: dict):
    match filename.suffix:
        case '.json':
            with filename.open(mode='w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, sort_keys=False)
        case '.yaml'|'.yml':
            with filename.open(mode='w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False)
        case '.toml':
            with filename.open(mode='w', encoding='utf-8') as f:
                toml.dump(config, f)
        case _:
            raise ValueError(f'Unsupported config file format: {filename.suffix}')
    logging.info(f'Saving config to {filename}')

def dump_config(filename: Path):
    save_config(filename, CONFIG)


def _get_nested_value(source: Mapping[str, Any] | None, path: str | None):
    if not source or not path:
        return None

    current: Any = source
    for key in path.split('.'):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def get_styles(path: str | None = None, fallback: str | None = 'default.metric_table') -> list[str]:
    """
    获取指定路径下的样式列表，支持在配置文件中使用层级结构（例如：styles.train.metric_table）。
    若未配置则返回默认样式或回退路径对应的样式。
    """
    if not path:
        path = 'default.metric_table'

    styles_cfg = CONFIG.get('styles')
    palette = _get_nested_value(styles_cfg if isinstance(styles_cfg, Mapping) else None, path)

    if palette is None and fallback:
        palette = _get_nested_value(styles_cfg if isinstance(styles_cfg, Mapping) else None, fallback)

    if palette is None:
        palette = _get_nested_value(DEFAULT_STYLE_CONFIG, path)

    if palette is None and fallback:
        palette = _get_nested_value(DEFAULT_STYLE_CONFIG, fallback)

    if palette is None:
        palette = ALL_STYLES

    if not isinstance(palette, (list, tuple)):
        raise ValueError(f"Styles for '{path}' must be a list or tuple, got {type(palette)}")
    return list(palette)


def get_style_sequence(path: str, count: int, fallback: str | None = 'default.metric_table') -> list[str]:
    """
    获取指定数量的样式列表。如果配置的样式数量不足，将按顺序循环填充。
    """
    if count <= 0:
        return []

    base_styles = get_styles(path, fallback)
    if not base_styles:
        base_styles = list(ALL_STYLES) or ['white']

    if len(base_styles) >= count:
        return base_styles[:count]

    repeats = math.ceil(count / len(base_styles))
    expanded = (base_styles * repeats)[:count]
    return expanded
