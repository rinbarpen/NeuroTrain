from pathlib import Path

import logging
from torch import nn
import warnings
import torch.distributed as dist

warnings.filterwarnings("ignore")

from src.config import get_config, dump_config, is_predict, is_train, is_test
from src.options import parse_args
from src.engine import Trainer, Tester, Predictor
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader
from src.utils import (
    get_train_tools,
    get_train_criterion,
    load_model,
    load_model_ext,
    prepare_logger,
    set_seed,
    model_info,
    model_flops,
    str2dtype,
)

if __name__ == "__main__":
    parse_args()
    c = get_config()
    output_dir = Path(c["output_dir"]) / c["task"] / c["run_id"]
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    predict_dir = output_dir / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_logger(output_dir, ('train', 'test', 'predict'))

    if c["seed"] >= 0:
        set_seed(c["seed"])
    device = c["device"]
    # device = ','.join(device)
    is_multigpu = "," in device
    if is_multigpu:
        dist.barrier()

    is_continue_mode = False  # use this mode if encountering crash while training

    model = get_model(c["model"]["name"], c["model"]["config"])
    model = model.to(device)
    pretrained_model = c["model"].get("pretrained")
    continue_checkpoint = c["model"].get("continue_checkpoint")
    if pretrained_model and pretrained_model != "":
        model_path = Path(pretrained_model)
        model_params = load_model(model_path, device)
        logging.info(f"Load model: {model_path}")
        model.load_state_dict(model_params)

    if continue_checkpoint and continue_checkpoint != "":
        model_path = Path(continue_checkpoint)
        model_params = load_model(model_path, device)
        logging.info(f"Load model: {model_path}, Now is in continue mode")
        model.load_state_dict(model_params)
        is_continue_mode = True

    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader()
    if is_train():
        train_dir.mkdir(exist_ok=True)
        dump_config(train_dir / "config.json")
        dump_config(train_dir / "config.toml")
        dump_config(train_dir / "config.yaml")
        logger = logging.getLogger("train")
        logger.info(
            f"Dumping config[.json|.yaml|.toml] to the {train_dir}/config[.json|.yaml|.toml]"
        )

        finished_epoch = 0
        tools = get_train_tools(model)
        optimizer = tools["optimizer"]
        lr_scheduler = tools["lr_scheduler"]
        scaler = tools["scaler"]
        criterion = get_train_criterion()

        if is_continue_mode and c["model"]["continue_ext_checkpoint"] != "":
            model_ext_path = Path(c["model"]["continue_ext_checkpoint"])
            model_ext_params = load_model_ext(model_ext_path, device)
            finished_epoch = model_ext_params["epoch"]
            logger.info(f"Continue Train from {finished_epoch}")

            try:
                optimizer.load_state_dict(model_ext_params["optimizer"])
                if lr_scheduler and model_ext_params["lr_scheduler"]:
                    lr_scheduler.load_state_dict(model_ext_params["lr_scheduler"])
                if scaler and model_ext_params["scaler"]:
                    scaler.load_state_dict(model_ext_params["scaler"])
            except Exception as e:
                logger.error(f"{e} WHILE LOADING EXT CHECKPOINT")

        handler = Trainer(train_dir, model, is_continue_mode=is_continue_mode)
        handler.train(
            num_epochs=c["train"]["epoch"],
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            lr_scheduler=lr_scheduler,
            early_stop="early_stopping" in c["train"],
            last_epoch=finished_epoch,
        )

    if is_test():
        test_dir.mkdir(exist_ok=True)
        dump_config(test_dir / "config.json")
        dump_config(test_dir / "config.yaml")
        dump_config(test_dir / "config.toml")
        logger = logging.getLogger("test")
        logger.info(
            f"Dumping config[.json|.yaml|.toml] to the {test_dir}/config[.json|.yaml|.toml]"
        )

        if is_train() and not is_continue_mode:
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f"Load model: {model_path}")
            model.load_state_dict(model_params)

        handler = Tester(test_dir, model)
        handler.test(test_dataloader=test_loader)

    if is_predict():
        predict_dir.mkdir(exist_ok=True)
        dump_config(predict_dir / "config.json")
        dump_config(predict_dir / "config.toml")
        dump_config(predict_dir / "config.yaml")
        logger = logging.getLogger("predict")
        logger.info(
            f"Dumping config[.json|.yaml|.toml] to the {predict_dir}/config[.json|.yaml|.toml]"
        )

        if is_train() and not is_continue_mode:
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f"Load model: {model_path}")
            model.load_state_dict(model_params)

        handler = Predictor(predict_dir, model)
        input_path = Path(c["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handler.predict(inputs, **c["predict"]["config"])
        else:
            inputs = [input_path]
            handler.predict(inputs, **c["predict"]["config"])

    # get_input_size
    input_sizes = c["model"]["config"]["input_sizes"] # get a list
    dtypes = c["model"]["config"].get("dtypes")
    if dtypes is not None and len(dtypes) > 0:
        dtypes = [str2dtype(dtype) for dtype in dtypes]
    else:
        dtype = None

    model_info(output_dir, model, input_sizes, dtypes=dtypes)
    model_flops(output_dir, model, input_sizes)
