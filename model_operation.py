import os
import os.path
from pathlib import Path
from pprint import pp
import numpy as np
import logging
import colorlog
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader

from config import get_config
from utils.early_stopping import EarlyStopping
from utils.recorder import Recorder
from utils.util import ScoreCalculator, save_model
from utils.painter import Plot
from utils.transform import image_transform, image_transforms, VisionTransformersBuilder

class Trainer:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.train_loss_image_path = output_dir / "train-epoch-loss.png"
        self.valid_loss_image_path = output_dir / "valid-epoch-loss.png"
        self.save_model_dir = output_dir / "weights"
        self.last_model_file_path = self.save_model_dir / "last.pt"
        self.best_model_file_path = self.save_model_dir / "best.pt"
        self.last_model_ext_file_path = self.save_model_dir / "last-ext.pt"
        self.best_model_ext_file_path = self.save_model_dir / "best-ext.pt"

        self.model = model

        self.save_model_dir.mkdir(exist_ok=True, parents=True)

    def train(self, num_epochs: int, 
              criterion: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader | None, 
              lr_scheduler: LRScheduler | None = None,
              *, early_stop: bool = False):
        CONFIG = get_config()
        device = torch.device(CONFIG['device'])
        save_every_n_epoch = CONFIG["train"]["save_every_n_epoch"]
        enable_valid_when_training = valid_dataloader is not None

        self.model = self.model.to(device)

        scaler = GradScaler(enabled=CONFIG['train']['scaler']['enabled'])
        if early_stop and valid_dataloader is None:
            colorlog.warning("Validate isn't launched, early_stop will be cancelled")
            early_stopper = None
        elif early_stop and valid_dataloader:
            early_stopper = EarlyStopping(CONFIG['train']['patience'])
        else:
            early_stopper = None

        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        class_labels = CONFIG['classes']
        metric_labels = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice'] # TODO: load from CONFIG
        train_calculator = ScoreCalculator(class_labels, metric_labels)
        valid_calculator = ScoreCalculator(class_labels, metric_labels) if enable_valid_when_training else None
        for epoch in range(1, num_epochs+1):
            # train
            self.model.train()
            train_loss = 0.0
            # last_train_loss = float('inf')
            for inputs, targets in tqdm(train_dataloader, desc=f'{epoch}/{num_epochs}, Training...'):
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)

                device_type = 'cuda' if 'cuda' in CONFIG['device'] else 'cpu'
                compute_type = torch.float16 if CONFIG['scaler']['compute_type'] != 'bfloat16' else torch.bfloat16
                with autocast(device_type, dtype=compute_type, enabled=CONFIG['scaler']['enabled']):
                    outputs = self.model(inputs)

                    loss = criterion(targets, outputs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                # logging.info(f'Epoch-Loss Variant: {loss.item() - last_train_loss}')
                # last_train_loss = loss.item()

                targets[targets >= 0.5] = 1
                targets[targets < 0.5] = 0
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                train_calculator.add_one_batch(
                    targets.detach().cpu().float().numpy(), 
                    outputs.detach().cpu().float().numpy())

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            colorlog.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')

            train_calculator.finish_one_epoch()

            # validate
            if enable_valid_when_training:
                valid_loss = 0.0
                # last_valid_loss = float('inf')
                self.model.eval()

                with torch.no_grad():
                    for inputs, targets in tqdm(valid_dataloader, desc=f'{epoch}/{num_epochs}, Validating...'):
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = self.model(inputs)
                        loss = criterion(targets, outputs)

                        valid_loss += loss.item()
                        # logging.info(f'Epoch-Loss Variant: {loss.item() - last_valid_loss}')
                        # last_valid_loss = loss.item()

                        targets[targets >= 0.5] = 1
                        targets[targets < 0.5] = 0
                        outputs[outputs >= 0.5] = 1
                        outputs[outputs < 0.5] = 0
                        valid_calculator.add_one_batch(
                            targets.detach().cpu().float().numpy(), 
                            outputs.detach().cpu().float().numpy())

                valid_loss /= len(valid_dataloader)
                valid_losses.append(valid_loss)

                colorlog.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')

                valid_calculator.finish_one_epoch()

                if early_stop:
                    early_stopper(valid_loss)
                    if early_stopper.is_stopped():
                        colorlog.info("Early stopping")
                        break

            # save
            if save_every_n_epoch > 0 and epoch % save_every_n_epoch == 0:
                save_model_filename = self.save_model_dir / f'{CONFIG["model"]["name"]}-{epoch}of{num_epochs}.pt'
                save_model_ext_filename = self.save_model_dir / f'{CONFIG["model"]["name"]}-{epoch}of{num_epochs}-ext.pt'
                save_model(save_model_filename, self.model,             
                           ext_path=save_model_ext_filename, optimizer=optimizer,        scaler=scaler, lr_scheduler=lr_scheduler,
                           epoch=epoch, version=CONFIG['run_id'])
                colorlog.info(f'save model to {save_model_filename} when {epoch=}, {train_loss=}')

            target_loss = valid_loss if valid_dataloader else train_loss
            if target_loss < best_loss:
                best_loss = target_loss
                save_model(self.best_model_file_path, self.model, 
                            ext_path=self.best_model_ext_file_path,
                            optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler,
                            epoch=epoch, version=CONFIG['run_id'])
                colorlog.info(f'save model params to {self.best_model_file_path} when {epoch=}, {best_loss=}')
                colorlog.info(f'save model ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')
            
            save_model(self.last_model_file_path, self.model,
                       ext_path=self.last_model_ext_file_path,
                       optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, 
                       epoch=num_epochs, version=CONFIG['run_id'])

        train_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)
        if enable_valid_when_training:
            valid_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)

        colorlog.info(f'save model to {self.last_model_file_path} when meeting to the last epoch')

        train_losses = np.array(train_losses, dtype=np.float64)
        valid_losses = np.array(valid_losses, dtype=np.float64) if enable_valid_when_training else None
        self._save_after_train(num_epochs, train_losses, valid_losses, optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler)

    def _save_after_train(self, num_epochs: int, train_losses: np.ndarray, valid_losses: np.ndarray|None, optimizer=None, scaler=None, lr_scheduler=None):
        CONFIG = get_config()
        labels = CONFIG['classes'][1:]
        Recorder.record_loss(train_losses, self.output_dir)

        plot = Plot(1, 1)
        plot.subplot().epoch_loss(num_epochs, train_losses, labels, title='Train-Epoch-Loss').complete()
        plot.save(self.train_loss_image_path)

        logging.info(f'Save train-epoch-loss graph to {self.train_loss_image_path}')
        if valid_losses:
            Recorder.record_loss(valid_losses, self.output_dir)
            
            plot = Plot(1, 1)
            plot.subplot().epoch_loss(num_epochs, valid_losses, labels, title='Valid-Epoch-Loss').complete()
            plot.save(self.valid_loss_image_path)
            logging.info(f'Save valid-epoch-loss graph to {self.valid_loss_image_path}')


        if CONFIG['private']['wandb']:
            import wandb
            wandb.log({
                'train': {
                    'losses': train_losses,
                    'loss_image': self.train_loss_image_path
                },
                'valid': {
                    'losses': valid_losses,
                    'loss_image': self.valid_loss_image_path
                },
                "config": {
                    "batch_size": CONFIG['train']['batch_size'],
                    "epoch": CONFIG['train']['epoch'],
                    "dataset": {
                        "name": CONFIG['name'],
                        "num_workers": CONFIG['num_workers'],
                    },
                    "save_every_n_epoch": CONFIG['train']['save_every_n_epoch'],
                    "early_stopping": CONFIG['train']['early_stopping'],
                    "patience": CONFIG['train']['patience'],
                },
                "model_data": {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "lr_scheduler": {
                        'weight': lr_scheduler.state_dict(),
                        "warmup": CONFIG['train']['lr_scheduler']['warmup'],
                    } if lr_scheduler else None,
                },
            })

class Tester:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        CONFIG = get_config()
        device = torch.device(CONFIG['device'])
        class_labels = CONFIG['classes']

        self.model = self.model.to(device)
        metric_labels = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice'] # TODO: load from CONFIG

        calculator = ScoreCalculator(class_labels, metric_labels)

        for inputs, targets in tqdm(test_dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)

            targets[targets >= 0.5] = 1
            targets[targets < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

            # (B, N, H, W) => N is n_classes
            calculator.add_one_batch(
                targets.detach().cpu().numpy(), 
                outputs.detach().cpu().numpy())

        calculator.record_batches(self.output_dir)

class Vaildator:
    def __init__(self, output_dir: Path, model: nn.Module):
        super().__init__()
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def valid(self, valid_dataloader: DataLoader):
        CONFIG = get_config()
        class_labels = CONFIG['classes']
        metric_labels = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice'] # TODO: load from CONFIG

        calculator = ScoreCalculator(class_labels, metric_labels)

        for inputs, targets in valid_dataloader:
            outputs = self.model(inputs)

            targets[targets >= 0.5] = 1
            targets[targets < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

            calculator.add_one_batch(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        calculator.record_batches(self.output_dir)

class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        from PIL import Image
        CONFIG = get_config()
        device = torch.device(CONFIG['device'])

        self.model = self.model.to(device)
        self.model.eval()
        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name
            input = Image.open(input).convert('L')
            input = image_transform(input, size=(512, 512)).unsqueeze(0)
            input = input.to(device)

            pred = self.model(input)

            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            pred = pred.detach().cpu().numpy()
            pred = pred.squeeze(0).squeeze(0)
            pred = pred.astype(np.uint8)

            image = Image.fromarray(pred, mode='L')
            image.save(self.output_dir / input_filename)
