from pathlib import Path

import logging
from torch import nn
import warnings

warnings.filterwarnings("ignore")

from config import get_config, dump_config, is_predict, is_train, is_test
from options import parse_args
from model_operation import Trainer, Tester, Predictor
from models.models import get_model
from utils.util import (get_train_tools, get_train_valid_test_dataloader, load_model, load_model_ext, prepare_logger, set_seed)
from utils.criterion import CombineCriterion

if __name__ == "__main__":
    parse_args()
    c = get_config()
    prepare_logger()

    if c["seed"] >= 0:
        set_seed(c["seed"])
    device = c["device"]

    output_dir = Path(c["output_dir"]) / c["task"] / c["run_id"]
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    predict_dir = output_dir / "predict"

    is_continue_mode = False

    model = get_model(c["model"]["name"], c["model"]["config"])
    if c['model']['continue_checkpoint'] != "":
        model_path = Path(c['model']['continue_checkpoint'])
        model_params = load_model(model_path, device)
        logging.info(f'Load model: {model_path}')
        model.load_state_dict(model_params)
        is_continue_mode = True

    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader()
    if is_train():
        dump_config(train_dir / "config.json")
        dump_config(train_dir / "config.toml")
        dump_config(train_dir / "config.yaml")
        logger = logging.getLogger('train')
        logger.info(f'Dumping config[.json|.yaml|.toml] to the {train_dir}/config[.json|.yaml|.toml]')

        finished_epoch = 0
        tools = get_train_tools(model)
        optimizer = tools['optimizer']
        lr_scheduler = tools['lr_scheduler']
        scaler = tools['scaler']
        criterion = CombineCriterion(nn.BCEWithLogitsLoss()) # Loss

        if is_continue_mode and c['model']['continue_ext_checkpoint'] != "":
            model_ext_path = Path(c['model']['continue_ext_checkpoint'])
            model_ext_params = load_model_ext(model_ext_path, device)
            finished_epoch = model_ext_params['epoch']
            logger.info(f'Continue Train from {finished_epoch}')

            try:
                optimizer.load_state_dict(model_ext_params['optimizer'])
                if lr_scheduler and model_ext_params['lr_scheduler']:
                    lr_scheduler.load_state_dict(model_ext_params['lr_scheduler'])
                if scaler and model_ext_params['scaler']:
                    scaler.load_state_dict(model_ext_params['scaler'])
            except Exception as e:
                logger.error(f'{e} WHILE LOADING EXT CHECKPOINT')

        handle = Trainer(train_dir, model)
        handle.train(
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
        dump_config(test_dir / "config.json")
        dump_config(test_dir / "config.yaml")
        dump_config(test_dir / "config.toml")
        logger = logging.getLogger('test')
        logger.info(f'Dumping config[.json|.yaml|.toml] to the {test_dir}/config[.json|.yaml|.toml]')

        if is_train() and not is_continue_mode:
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        handle = Tester(test_dir, model)
        handle.test(test_dataloader=test_loader)


    if is_predict():
        dump_config(predict_dir / "config.json")
        dump_config(predict_dir / "config.toml")
        dump_config(predict_dir / "config.yaml")
        logger = logging.getLogger('predict')
        logger.info(f'Dumping config[.json|.yaml|.toml] to the {predict_dir}/config[.json|.yaml|.toml]')

        if is_train() and not is_continue_mode:
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        handle = Predictor(predict_dir, model)
        input_path = Path(c["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handle.predict(inputs, **c["predict"]["config"])
        else:
            inputs = [input_path]
            handle.predict(inputs, **c["predict"]["config"])


