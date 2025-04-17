# output/{task_name}/{timestamp}/{train|valid|test|predict}/{classification|detection|segmentation|weights|..}
import json
import yaml
import toml
from pathlib import Path
import logging

TRAIN_MODE = 1
TEST_MODE  = 2
PREDICT_MODE = 4

ALL_METRIC_LABELS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']

CONFIG = {}

# CONFIG = {
#     "output_dir": "./output",
#     "task": "TestModel", # project
#     "entity": "Lab",
#     "run_id": "0",
#     "device": "cuda",
#     "seed": 42,
#     "classes": ["label0", "label1"],
#     "model": {
#         "name": "unet",
#         "continue_checkpoint": "",
#         "config": {
#             "n_channels": 1,
#             "n_classes": 1
#         }
#     },
#     "train": {
#         "dataset": {
#             "name": "DRIVE",
#             "path": "./data/DRIVE",
#             "num_workers": 0,
#         },
#         "batch_size": 1,
#         "epoch": 40,
#         "augment_boost": {
#             "enabled": False,
#             "config": {}
#         },
#         "save_every_n_epoch": 0, # <= 0 is unavailable, > 0 is available
#         "optimizer": {
#             "learning_rate": 3e-6,
#             "weight_decay": 1e-8,
#             "eps": 1e-8
#         },
#         "lr_scheduler": {
#             "enabled": False,
#             "warmup": 50,
#             "warmup_lr": 0.03
#         },
#         "scaler": {
#             "enabled": False,
#             "compute_type": "bfloat16",
#         },
#         "early_stopping": {
#             "enabled": False,
#             "patience": 3,
#         },
#     },
#     "test": {
#         "dataset": {
#             "name": "DRIVE",
#             "path": "./data/DRIVE",
#             "num_workers": 0
#         },
#         "batch_size": 1
#     },
#     "predict": {
#         "input": "",
#         "config": {
#             "show_cam": False,
#         }
#     },

#     "private": {
#         "wandb": False,
#         "log": True,
#         "verbose": False,
#         "mode": 0, # 1 for train, 2 for test, 4 for predict
#     }
# }


# from dotenv import load_dotenv
# load_dotenv()

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

def set_config(config):
    global CONFIG
    CONFIG = config

def get_config():
    return CONFIG

def load_config(filename: Path):
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
    CONFIG = get_config()
    match filename.suffix:
        case '.json':
            with filename.open(mode='w', encoding='utf-8') as f:
                json.dump(CONFIG, f)
        case '.toml':
            with filename.open(mode='w', encoding='utf-8') as f:
                toml.dump(CONFIG, f)
        case '.yaml'|'.yml':
            with filename.open(mode='w', encoding='utf-8') as f:
                yaml.safe_dump_all(CONFIG, f, allow_unicode=True, encoding='utf-8')

# GLOBAL CONSTANTS
