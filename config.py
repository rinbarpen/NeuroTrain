# output/{timestamp}/{task_name}/{train|validate|test}/{classification,detection,segmentation|weights|..}

CONFIG = {
    "device": "cuda",
    "seed": 42,
    "model": {
        "name": "test",
        "save": "",
        "load": "",
        "config": {},
    },
    "train": {
        "dataset": {
            "name": "",
            "path": "",
            "num_workers": 0,
        },
        "batch_size": 1,
        "epoch": 0,
        "augment_boost": {
            "on": False,
            "config": {},
        },
        "save_every_n_epoch": 0, # <= 0 is unavailable, > 0 is available
        "optimizer": {
            "learning_rate": 3e-6,
            "weight_decay": 1e-8,
            "eps": 1e-8,
        },
        "lr_scheduler": {
            "warmup": 50, # epoch
            "warmup_ratio": 0.03, # learning_rate 
        },
        "amp": False, # bfloat16
        "early_stopping": False,
        "patience": 3,
    },
    "test": {
        "dataset": {
            "name": "",
            "path": "",
            "num_workers": 0,
        },
        "batch_size": 1,
    },
    "predict": {
        "input": None,
        "config": {},
    },
    "classes": ["Background", "Retina"],
    # output
    "output_dir": "./output",
    "task": "",
    "run_id": "",

    "private": {
        "wandb": False,
        "log": True,
        "verbose": False,
        "mode": 0,
    }
}

from dotenv import load_dotenv
load_dotenv()

def is_verbose():
    return CONFIG['private']['verbose']
def is_train():
    return CONFIG['private']['mode'] == 0
def is_test():
    return CONFIG['private']['mode'] == 1
def is_predict():
    return CONFIG['private']['mode'] == 2

def set_config(config):
    global CONFIG
    CONFIG = config

def get_config():
    return CONFIG

# GLOBAL CONSTANTS
