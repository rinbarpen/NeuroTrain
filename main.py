import os
import time
from pathlib import Path

import colorama
import logging
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

colorama.init()
from config import get_config, is_predict, is_train, is_test
from options import dump_config, parse_args
from utils.util import load_model, prepare_logger, save_model
from models.models import get_model
from utils.dataset.dataset import get_test_dataset, get_train_valid_dataset
from model_operation import Trainer, Tester, Predictor

# from torchvision.models import resnet50


def end_task():
    # CONFIG = get_config()
    # if is_train():
    #     # draw epoch-loss curve
    #     # metrics = scores(targets, outputs, labels)
    #     # scores_map = {k: v for k, v in metrics.items() if not any(keyword == k for keyword in ['argmax', 'argmin', 'mean'])}
    #     # Plot(2, 3).metrics(scores_map).save(self.output_dir / "metrics.png")
    #     pass
    # if is_test():
    #     # draw metrics curve
    #     metrics_file = (
    #         Path(CONFIG["output_dir"])
    #         / CONFIG["run_id"]
    #         / CONFIG["task"]
    #         / "test"
    #         / "metrics.parquet"
    #     )
    #     df = pd.read_parquet(metrics_file)
    #     losses = df["loss"]
    #     # Plot(1, 1).metrics().complete().save()
    # if is_predict():
    #     # paint the predicted data
    #     pass
    pass


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    prepare_logger()
    parse_args()

    # register models
    CONFIG = get_config()
    if CONFIG["seed"] >= 0:
        set_seed(CONFIG["seed"])

    output_dir = Path(CONFIG["output_dir"]) / CONFIG["task"] / CONFIG["run_id"]
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    predict_dir = output_dir / "predict"

    if CONFIG["private"]["wandb"]:
        import wandb
        wandb.init(entity="lpoutyoumu", project=CONFIG["task"], id=CONFIG["run_id"])

    model = get_model(CONFIG["model"]["name"], CONFIG["model"]["config"])
    if CONFIG['model']['continue_checkpoint'] != "":
        model_path = Path(CONFIG['model']['continue_checkpoint'])
        model_params = load_model(model_path, 'cuda')
        logging.info(f'Load model: {model_path}')
        model.load_state_dict(model_params)

    if is_train():
        train_dataset, valid_dataset = get_train_valid_dataset(
            dataset_name=CONFIG["train"]["dataset"]["name"],
            base_dir=Path(CONFIG["train"]["dataset"]["path"]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["train"]["batch_size"],
            pin_memory=True,
            num_workers=CONFIG["train"]["dataset"]["num_workers"],
            shuffle=True,
        )
        valid_loader = (
            DataLoader(
                valid_dataset,
                batch_size=CONFIG["train"]["batch_size"],
                pin_memory=True,
                num_workers=CONFIG["train"]["dataset"]["num_workers"],
                shuffle=True,
            )
            if valid_dataset
            else None
        )
        optimizer = AdamW(
            model.parameters(),
            lr=CONFIG["train"]["optimizer"]["learning_rate"],
            weight_decay=CONFIG["train"]["optimizer"]["weight_decay"],
            eps=CONFIG["train"]["optimizer"]["eps"],
        )
        lr_scheduler = LRScheduler(optimizer) if CONFIG['train']['lr_scheduler']['on'] else None
        # lr_scheduler = None if CONFIG['train']['lr_scheduler']['on'] else None
        criterion = nn.BCEWithLogitsLoss()
        handle = Trainer(train_dir, model)
        handle.train(
            num_epochs=CONFIG["train"]["epoch"],
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            lr_scheduler=lr_scheduler,
            early_stop=CONFIG["train"]["early_stopping"],
        )
        dump_config(train_dir / "config.json")
        dump_config(train_dir / "config.toml")
        dump_config(train_dir / "config.yaml")
        
        last_model_filepath = Path(output_dir / "last.pt")
        best_model_filepath = Path(output_dir / "best.pt")
        last_model_filepath.symlink_to(handle.last_model_file_path)
        best_model_filepath.symlink_to(handle.best_model_file_path)

        logging.info(f'Link(soft) last.pt and best.pt to {output_dir}')

    if is_test():
        if is_train():
            model_path = train_dir / "weights" / "best.pt"
            # model_path = output_dir / "best.pt"
            model_params = load_model(model_path, 'cuda')
            logging.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        test_dataset = get_test_dataset(
            dataset_name=CONFIG["test"]["dataset"]["name"],
            base_dir=Path(CONFIG["test"]["dataset"]["path"]),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG["test"]["batch_size"],
            pin_memory=True,
            num_workers=CONFIG["test"]["dataset"]["num_workers"],
        )
        # callback = lambda outputs: return outputs
        handle = Tester(test_dir, model)
        handle.test(test_dataloader=test_loader)
        dump_config(test_dir / "config.json")
        dump_config(test_dir / "config.toml")
        dump_config(test_dir / "config.yaml")

    if is_predict():
        if is_train():
            model_path = train_dir / "weights" / "best.pt"
            # model_path = output_dir / "best.pt"
            model_params = load_model(model_path, 'cuda')
            logging.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        handle = Predictor(predict_dir, model)
        input_path = Path(CONFIG["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handle.predict(inputs, **CONFIG["predict"]["config"])
        dump_config(predict_dir / "config.json")
        dump_config(predict_dir / "config.toml")
        dump_config(predict_dir / "config.yaml")

    end_task()
