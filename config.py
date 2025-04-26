# output/{task_name}/{timestamp}/{train|valid|test|predict}/{classification|detection|segmentation|weights|..}
import json
import yaml
import toml
from pathlib import Path
import logging

TRAIN_MODE = 1
TEST_MODE  = 2
PREDICT_MODE = 4
TRAIN_TEST_MODE = TRAIN_MODE | TEST_MODE
TEST_PREIDCT_MODE = TEST_MODE | PREDICT_MODE
ALL_MODE = TRAIN_MODE | TEST_MODE | PREDICT_MODE

ALL_METRIC_LABELS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']

ALL_STYLES = ['cyan', 'magenta', 'green', 'yellow', 'blue', 'red']

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

def wandb_is_available():
    return CONFIG['private']['wandb']

def set_config(config: dict):
    global CONFIG
    CONFIG = config

def get_config() -> dict:
    return CONFIG

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

    logging.info(f'Loading config from {filename}')
    return config

def dump_config(filename: Path):
    c = CONFIG
    match filename.suffix:
        case '.json':
            with filename.open(mode='w', encoding='utf-8') as f:
                json.dump(c, f)
        case '.yaml'|'.yml':
            with filename.open(mode='w', encoding='utf-8') as f:
                yaml.safe_dump(c, f, sort_keys=False)
        case '.toml':
            with filename.open(mode='w', encoding='utf-8') as f:
                toml.dump(c, f)

# GLOBAL CONSTANTS
