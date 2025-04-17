import platform
from pathlib import Path

import colorama
import logging
import torch
from torch import nn
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

colorama.init()
from config import get_config, dump_config, is_predict, is_train, is_test
from options import parse_args
from models.models import get_model
from utils.transform import get_transforms
from model_operation import Trainer, Tester, Predictor
from utils.util import (get_logger, get_train_tools, get_train_valid_test_dataloader, load_model, prepare_logger, set_seed)
from utils.criterion import CombineCriterion
from utils.see_cam import ImageHeatMapGenerator, ClassifierOutputTarget, SemanticSegmentationTarget

class UNetPredictor(Predictor):
    def __init__(self, output_dir: Path, model: nn.Module):
        super(UNetPredictor, self).__init__(output_dir, model)

    def record(self, input: np.ndarray, pred: np.ndarray, **kwargs):
        output_filename = self.output_dir / (kwargs['filename'] + '.png')
        heatmap_filename = self.output_dir / (kwargs['filename'] + '_heat.png')
        original_size = kwargs['original_size'] # (W, H)
        c = get_config()
        device = c['device']

        input_tensor = torch.from_numpy(input)
        input_tensor = input_tensor.to(device)
        self.model = self.model.to(device)
        self.model.eval()

        pred = self.model(input_tensor)
        pred = pred.squeeze(0).squeeze(0)
        pred = Image.fromarray(pred.detach().cpu().numpy(), mode='L')
        pred = pred.resize(original_size)
        pred.save(output_filename)

        generator = ImageHeatMapGenerator(self.model, target_layers=[self.model[-1]], targets=[ClassifierOutputTarget(0)], model_params_file=c['model']['continue_checkpoint'])

        def transform_fn(x):
            transforms = get_transforms()
            y = transforms(x).unsqueeze(0)
            return y

        heatmap = generator.check(Image.fromarray(input, mode='L'), transform_fn=transform_fn)
        heatmap.save(heatmap_filename)




    @classmethod
    def preprocess(self, input: Path):
        input = Image.open(input).convert('L')
        size = input.size # (H, W)
        transforms = get_transforms()
        input = transforms(input).unsqueeze(0)
        return input, size

    @classmethod
    def postprocess(self, pred: torch.Tensor):
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.astype(np.uint8)
        image = Image.fromarray(pred, mode='L')
        return image


if __name__ == "__main__":
    parse_args()
    prepare_logger()

    CONFIG = get_config()
    if CONFIG["seed"] >= 0:
        set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    output_dir = Path(CONFIG["output_dir"]) / CONFIG["task"] / CONFIG["run_id"]
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    predict_dir = output_dir / "predict"

    model = get_model(CONFIG["model"]["name"], CONFIG["model"]["config"])
    if CONFIG['model']['continue_checkpoint'] != "":
        model_path = Path(CONFIG['model']['continue_checkpoint'])
        model_params = load_model(model_path, device)
        logging.info(f'Load model: {model_path}')
        model.load_state_dict(model_params)

    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader()
    if is_train():
        finished_epoch = 0

        tools = get_train_tools(model)
        optimizer = tools['optimizer']
        lr_scheduler = tools['lr_scheduler']
        scaler = tools['scaler']
        criterion = CombineCriterion(nn.BCEWithLogitsLoss())

        if CONFIG['model']['continue_ext_checkpoint'] != "":
            model_ext_path = Path(CONFIG['model']['continue_ext_checkpoint'])
            model_ext_params = load_model(model_path, device)
            finished_epoch = model_ext_params['epoch']
            logging.info(f'Continue Train from {finished_epoch}')

            optimizer.load_state_dict(model_ext_params['optimizer'])
            if lr_scheduler and model_ext_params['lr_scheduler']:
                lr_scheduler.load_state_dict(model_ext_params['lr_scheduler'])
            if scaler and model_ext_params['scaler']:
                scaler.load_state_dict(model_ext_params['scaler'])

        handle = Trainer(train_dir, model)
        handle.train(
            num_epochs=CONFIG["train"]["epoch"],
            criterion=criterion,
            optimizer=optimizer,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            lr_scheduler=lr_scheduler,
            early_stop=CONFIG["train"]["early_stopping"]["enabled"],
            last_epoch=finished_epoch,
        )
        dump_config(train_dir / "config.json")
        dump_config(train_dir / "config.toml")
        dump_config(train_dir / "config.yaml")
        
        logger = get_logger('train')
        logger.info(f'Dumping config under the {train_dir}')
        
        if platform.system() == 'Windows':
            last_model_filepath = Path(output_dir / "last.pt")
            best_model_filepath = Path(output_dir / "best.pt")
            last_model_filepath.symlink_to(handle.last_model_file_path)
            best_model_filepath.symlink_to(handle.best_model_file_path)
            logging.info(f'Link(soft) last.pt and best.pt under the {output_dir}')

    if is_test():
        logger = get_logger('test')

        if is_train():
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        # callback = lambda outputs: return outputs
        handle = Tester(test_dir, model)
        handle.test(test_dataloader=test_loader)
        dump_config(test_dir / "config.json")
        dump_config(test_dir / "config.toml")
        dump_config(test_dir / "config.yaml")

        logger.info(f'Dumping config under the {test_dir}')

    if is_predict():
        logger = get_logger('predict')
        if is_train():
            model_path = train_dir / "weights" / "best.pt"
            model_params = load_model(model_path, device)
            logger.info(f'Load model: {model_path}')
            model.load_state_dict(model_params)

        handle = Predictor(predict_dir, model)
        input_path = Path(CONFIG["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handle.predict(inputs, **CONFIG["predict"]["config"])
        dump_config(predict_dir / "config.json")
        dump_config(predict_dir / "config.toml")
        dump_config(predict_dir / "config.yaml")

        logger.info(f'Dumping config under the {predict_dir}')

