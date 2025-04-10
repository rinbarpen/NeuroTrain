import os
import os.path
from pathlib import Path
from pprint import pp
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import wandb
import torch
import logging
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader

from config import get_config, ALL_METRIC_LABELS
from utils.early_stopping import EarlyStopping
from utils.recorder import Recorder
from utils.util import get_logger, save_model
from utils.scores import ScoreCalculator
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
        self.logger = get_logger('train')

    def train(self, num_epochs: int, 
              criterion: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader | None, 
              lr_scheduler: LRScheduler | None = None,
              *, early_stop: bool = False):
        c = get_config()
        device = torch.device(c['device'])
        save_every_n_epoch = c["train"]["save_every_n_epoch"]
        enable_valid_when_training = valid_dataloader is not None

        self.model = self.model.to(device)

        scaler = GradScaler(enabled=c['train']['scaler']['enabled'])
        if early_stop and valid_dataloader is None:
            self.logger.warning("Validate isn't launched, early_stop will be cancelled")
            early_stopper = None
        elif early_stop and valid_dataloader:
            early_stopper = EarlyStopping(c['train']['early_stopping']['patience'])
        else:
            early_stopper = None

        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        class_labels = c['classes']
        metric_labels = ALL_METRIC_LABELS # TODO: load from c
        train_calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger)
        valid_calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger) if enable_valid_when_training else None
        
        pbar_total = tqdm(total=num_epochs)
        for epoch in range(1, num_epochs+1):
            pbar_total.set_description(f'{epoch}/{num_epochs}, Training...')
            # train
            self.model.train()
            train_loss = 0.0
            # last_train_loss = float('inf')
            
            pbar = tqdm(total=len(train_dataloader))
            for i, (inputs, targets) in enumerate(train_dataloader):
                pbar.set_description(f'{i}/{len(train_dataloader)}, Training...')

                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)

                device_type = 'cuda' if 'cuda' in c['device'] else 'cpu'
                compute_type = torch.float16 if c['train']['scaler']['compute_type'] != 'bfloat16' else torch.bfloat16
                with autocast(device_type, dtype=compute_type, enabled=c['train']['scaler']['enabled']):
                    outputs = self.model(inputs)

                    loss = criterion(targets, outputs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss = loss.item()
                train_loss += batch_loss
                pbar.set_postfix({'batch_loss': batch_loss})

                targets, outputs = self.postprocess(targets, outputs)
                train_calculator.add_one_batch(
                    targets.detach().cpu().float().numpy(), 
                    outputs.detach().cpu().float().numpy())

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            pbar_total.set_postfix({'epoch_loss': train_loss})

            self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')

            train_calculator.finish_one_epoch()

            # validate
            if enable_valid_when_training:
                valid_loss = 0.0
                # last_valid_loss = float('inf')
                self.model.eval()

                with torch.no_grad():
                    pbar = tqdm(total=len(valid_dataloader))
                    for i, (inputs, targets) in enumerate(valid_dataloader):
                        pbar.set_description(f'{i}/{len(valid_dataloader)}, Validating...')

                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = self.model(inputs)
                        loss = criterion(targets, outputs)

                        batch_loss = loss.item()
                        valid_loss += batch_loss
                        pbar.set_postfix({'batch_loss': batch_loss})
                        # logging.info(f'Epoch-Loss Variant: {loss.item() - last_valid_loss}')
                        # last_valid_loss = loss.item()

                        targets, outputs = self.postprocess(targets, outputs)
                        valid_calculator.add_one_batch(
                            targets.detach().cpu().float().numpy(), 
                            outputs.detach().cpu().float().numpy())

                valid_loss /= len(valid_dataloader)
                valid_losses.append(valid_loss)

                pbar_total.set_postfix({'valid_epoch_loss': valid_loss})

                self.logger.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')

                valid_calculator.finish_one_epoch()

                if early_stop:
                    early_stopper(valid_loss)
                    if early_stopper.is_stopped():
                        self.logger.info("Early stopping")
                        break

            # save
            if save_every_n_epoch > 0 and epoch % save_every_n_epoch == 0:
                save_model_filename = self.save_model_dir / f'{c["model"]["name"]}-{epoch}of{num_epochs}.pt'
                save_model_ext_filename = self.save_model_dir / f'{c["model"]["name"]}-{epoch}of{num_epochs}-ext.pt'
                save_model(save_model_filename, self.model,             
                           ext_path=save_model_ext_filename, optimizer=optimizer,        scaler=scaler, lr_scheduler=lr_scheduler,
                           epoch=epoch, version=c['run_id'])
                self.logger.info(f'save model to {save_model_filename} when {epoch=}, {train_loss=}')

            target_loss = valid_loss if enable_valid_when_training else train_loss
            if target_loss < best_loss:
                best_loss = target_loss
                save_model(self.best_model_file_path, self.model, 
                            ext_path=self.best_model_ext_file_path,
                            optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler,
                            epoch=epoch, version=c['run_id'])
                self.logger.info(f'save model params to {self.best_model_file_path} when {epoch=}, {best_loss=}')
                self.logger.info(f'save model ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')
            
            save_model(self.last_model_file_path, self.model,
                       ext_path=self.last_model_ext_file_path,
                       optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, 
                       epoch=num_epochs, version=c['run_id'])

        train_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)
        if enable_valid_when_training:
            valid_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)

        self.logger.info(f'save model to {self.last_model_file_path} when meeting to the last epoch')

        train_losses = np.array(train_losses, dtype=np.float64)
        valid_losses = np.array(valid_losses, dtype=np.float64) if enable_valid_when_training else None
        self._save_after_train(num_epochs, train_losses, valid_losses, optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler)

    @classmethod
    def postprocess(self, targets: torch.Tensor, outputs: torch.Tensor):
        targets[targets >= 0.5] = 1
        targets[targets < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        return targets, outputs


    def _save_after_train(self, num_epochs: int, train_losses: np.ndarray, valid_losses: np.ndarray|None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()
        labels = c['classes']
        Recorder.record_loss(train_losses, self.output_dir, logger=self.logger)

        plot = Plot(1, 1)
        plot.subplot().epoch_loss(num_epochs, train_losses, labels, title='Train-Epoch-Loss').complete()
        plot.save(self.train_loss_image_path)

        self.logger.info(f'Save train-epoch-loss graph to {self.train_loss_image_path}')
        if valid_losses:
            Recorder.record_loss(valid_losses, self.output_dir, logger=self.logger)
            
            plot = Plot(1, 1)
            plot.subplot().epoch_loss(num_epochs, valid_losses, labels, title='Valid-Epoch-Loss').complete()
            plot.save(self.valid_loss_image_path)
            self.logger.info(f'Save valid-epoch-loss graph to {self.valid_loss_image_path}')


        if c['private']['wandb']:
            train_c = c['train']
            wandb.log({
                "train": {
                    "losses": train_losses,
                    "loss_image": self.train_loss_image_path,
                },
                "valid": {
                    "losses": valid_losses,
                    "loss_image": self.valid_loss_image_path,
                },
                "config": {
                    "batch_size": train_c['batch_size'],
                    "epoch": train_c['epoch'],
                    "dataset": train_c['dataset'],
                    "save_every_n_epoch": train_c['save_every_n_epoch'],
                    "early_stopping": {
                        "patience": train_c['early_stopping']['patience'],
                    } if train_c['early_stopping']['enabled'] else None,
                },
                "model_data": {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "lr_scheduler": {
                        'weights': lr_scheduler.state_dict(),
                        "warmup": train_c['lr_scheduler']['warmup'],
                        "warmup_lr": train_c['lr_scheduler']['warmup_lr'],
                    } if lr_scheduler else None,
                },
            })

class Tester:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('test')

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        c = get_config()
        device = torch.device(c['device'])
        class_labels = c['classes']

        self.model = self.model.to(device)
        metric_labels = ALL_METRIC_LABELS # TODO: load from c

        calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger)

        for inputs, targets in tqdm(test_dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            # (B, N, H, W) => N is n_classes
            outputs = self.model(inputs)

            targets, outputs = self.postprocess(targets, outputs)
            calculator.add_one_batch(
                targets.detach().cpu().numpy(), 
                outputs.detach().cpu().numpy())

        calculator.record_batches(self.output_dir)
    @classmethod
    def postprocess(self, targets: torch.Tensor, outputs: torch.Tensor):
        targets[targets >= 0.5] = 1
        targets[targets < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        return targets, outputs


class Vaildator:
    def __init__(self, output_dir: Path, model: nn.Module):
        super().__init__()
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('valid')

    @torch.no_grad()
    def valid(self, valid_dataloader: DataLoader):
        c = get_config()
        class_labels = c['classes']
        metric_labels = ALL_METRIC_LABELS # TODO: load from c

        calculator = ScoreCalculator(class_labels, metric_labels)

        for inputs, targets in valid_dataloader:
            outputs = self.model(inputs)

            self.postprocess(targets, outputs)
            calculator.add_one_batch(targets.cpu().detach().numpy(), 
                                     outputs.cpu().detach().numpy())

        calculator.record_batches(self.output_dir)

    @classmethod
    def postprocess(self, targets: torch.Tensor, outputs: torch.Tensor):
        targets[targets >= 0.5] = 1
        targets[targets < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        return targets, outputs

class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('predict')

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])

        self.model = self.model.to(device)
        self.model.eval()
        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name
            input = self.preprocess(input)

            input = input.to(device)

            pred = self.model(input)

            image = self.postprocess(pred)
            image.save(self.output_dir / input_filename)

    @classmethod
    def preprocess(self, input: Path):
        input = Image.open(input).convert('L')
        input = image_transform(input, size=(512, 512)).unsqueeze(0)
        # transform = image_transforms(resize=(512, 512), is_rgb=False, is_PIL_image=True)
        # input = transform(input).unsqueeze(0)
        return input

    @classmethod
    def postprocess(self, pred: torch.Tensor):
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.astype(np.uint8)
        image = Image.fromarray(pred, mode='L')
        return image
