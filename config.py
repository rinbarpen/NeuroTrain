# output/{task_name}/{timestamp}/{train|valid|test|predict}/{classification|detection|segmentation|weights|..}

CONFIG = {
    "output_dir": "./output",
    "task": "TestModel",
    "run_id": "0",
    "device": "cuda",
    "seed": 42,
    "classes": ["Background", "Retina"],
    "model": {
        "name": "unet",
        "continue_checkpoint": "",
        "config": {
            "n_channels": 1,
            "n_classes": 1
        }
    },
    "train": {
        "dataset": {
            "name": "DRIVE",
            "path": "./data/DRIVE",
            "num_workers": 0
        },
        "batch_size": 1,
        "epoch": 40,
        "augment_boost": {
            "on": True,
            "config": {}
        },
        "save_every_n_epoch": 0, # <= 0 is unavailable, > 0 is available
        "optimizer": {
            "learning_rate": 3e-6,
            "weight_decay": 1e-8,
            "eps": 1e-8
        },
        "lr_scheduler": {
            "on": False,
            "warmup": 50,
            "warmup_lr": 0.03
        },
        "amp": False, # default is bfloat16
        "early_stopping": False,
        "patience": 3
    },
    "test": {
        "dataset": {
            "name": "DRIVE",
            "path": "./data/DRIVE",
            "num_workers": 0
        },
        "batch_size": 1
    },
    "predict": {
        "input": "",
        "config": {}
    },

    "private": {
        "wandb": False,
        "log": True,
        "verbose": False,
        "mode": 0, # 0 for train, 1 for test, 2 for predict
    }
}


from dotenv import load_dotenv
load_dotenv()

def is_verbose():
    return CONFIG['private']['verbose']
def is_train():
    return CONFIG['private']['mode'] == 0 or CONFIG['private']['mode'] == 3 
def is_test():
    return CONFIG['private']['mode'] == 1 or CONFIG['private']['mode'] == 3
def is_predict():
    return CONFIG['private']['mode'] == 2

def wandb_is_available():
    return CONFIG['private']['wandb']

def set_config(config):
    global CONFIG
    CONFIG = config

def get_config():
    return CONFIG

# GLOBAL CONSTANTS
