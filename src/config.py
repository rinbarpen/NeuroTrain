# output/{task_name}/{timestamp}/{train|test|predict}/{classification|detection|segmentation|weights|..}
import json
import yaml
import toml
from pathlib import Path
import logging

# GLOBAL CONSTANTS
TRAIN_MODE = 1
TEST_MODE  = 2
PREDICT_MODE = 4
ALL_MODE = TRAIN_MODE | TEST_MODE | PREDICT_MODE

ALL_METRIC_LABELS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice']
ALL_STYLES = ['cyan', 'magenta', 'green', 'yellow', 'blue', 'red']

DATASET_ROOT_DIR = 'data'
CACHE_DIR = 'data/cache' # for numpy version's dataset
MODEL_DIR = 'models'
PRETRAINED_MODEL_DIR = 'models/pretrained'
CONFIG_DIR = 'configs'
CAHCER_CONFIG_DIR = 'configs/cacher'
DATASET_CONFIG_DIR = 'configs/dataset'
PIPELINE_CONFIG_DIR = 'configs/pipeline'
SINGLE_CONFIG_DIR = 'configs/single'

INPUT_SHAPE = (1, 512, 512)  # Default input shape for models

class TrainOutputFilenameEnv:
    OUTPUT_TRAIN_LOSS_FILENAME = '{train_dir}/train_epoch_loss.png'
    OUTPUT_VALID_LOSS_FILENAME = '{train_dir}/valid_epoch_loss.png'
    OUTPUT_LAST_MODEL_FILENAME = '{train_dir}/weights/last.pt'
    OUTPUT_BEST_MODEL_FILENAME = '{train_dir}/weights/best.pt'
    OUTPUT_LAST_EXT_MODEL_FILENAME = '{train_dir}/weights/last.ext.pt'
    OUTPUT_BEST_EXT_MODEL_FILENAME = '{train_dir}/weights/best.ext.pt'
    OUTPUT_TRAIN_LOSS_DETAILS_FILENAME = '{train_dir}/train_epoch_loss.csv'
    OUTPUT_VALID_LOSS_DETAILS_FILENAME = '{train_dir}/valid_epoch_loss.csv'

class InferenceOutputFilenameEnv:
    OUTPUT_RESULT_FILENAME = '{infer_dir}/{input_filename}.{input_file_ext}'


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
