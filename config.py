# output/{timestamp}/{task_name}/{classification,detect,segment,inference}/{train|validate|test}/{weights|..}

CONFIG = {
    "device": "cuda",
    "seed": 42,
    "model": {
        "save_path": "",
        "load_path": "",
        "config": {},
    },
    "train": {
        "dataset": {
            "name": "",
            "path": "",
        },
        "batch_size": 1,
        "epoch": 0,
        "learning_rate": 3e-6,
        "augment_boost": {
            "on": False,
            "config": {},
        },
        "save_every_n_epoch": 0, # < 0 for unavailable, > 0 for available
        "optimizer": {
            "weight_decay": 1e-8,
        },
        "criterion": [],
        "memory": {
            "amp": False,
            "mode": "", # TODO: Unimplement!
        },
        "early_stopping": False,
        "patience": 3,
    },
    "validate": {
        "dataset": {
            "name": "",
            "path": "",
        },
        "batch_size": 1,
    },
    "test": {
        "dataset": {
            "name": "",
            "path": "",
        },
        "batch_size": 1,
    },
    "inference": {
        "input": None,
        "config": {},
    },
    # output
    "output_dir": "./output",
    "task": "", # dataset
    "task_class": "",

    "private": {
        "wandb": False,
        "log": True,
        "progress": True,
        "verbose": False,
    }
}

IS_TRAIN = False
IS_TEST = False
IS_INFERENCE = False


from dotenv import load_dotenv
load_dotenv()

# GLOBAL CONSTANTS
