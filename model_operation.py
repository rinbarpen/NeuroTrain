from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import wandb
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console

from config import get_config, ALL_STYLES
from utils.early_stopping import EarlyStopping
from utils.data_saver import DataSaver
from utils.util import save_model
from utils.timer import Timer
from utils.scores import ScoreCalculator
from utils.painter import Plot
from utils.transform import get_transforms
from utils.criterion import CombineCriterion

class Trainer:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.train_loss_image_path = output_dir / "train_epoch_loss.png"
        self.valid_loss_image_path = output_dir / "valid_epoch_loss.png"
        self.save_model_dir = output_dir / "weights"
        self.last_model_file_path = self.save_model_dir / "last.pt"
        self.best_model_file_path = self.save_model_dir / "best.pt"
        self.last_model_ext_file_path = self.save_model_dir / "last-ext.pt"
        self.best_model_ext_file_path = self.save_model_dir / "best-ext.pt"

        self.model = model

        self.save_model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger('train')
        self.data_saver = DataSaver(output_dir)
        
        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.train_calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger, saver=self.data_saver)
        self.valid_calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger, saver=self.data_saver)
        

    def train(self, num_epochs: int, 
              criterion: CombineCriterion, 
              optimizer: torch.optim.Optimizer,
              train_dataloader: DataLoader,
              valid_dataloader: DataLoader | None = None, 
              scaler: GradScaler | None = None, 
              lr_scheduler: LRScheduler | None = None,
              *, early_stop: bool = False, last_epoch: int=0):
        self.train_calculator.clear()
        self.valid_calculator.clear()

        c = get_config()
        device = torch.device(c['device'])
        save_every_n_epoch = c["train"]["save_every_n_epoch"]
        enable_valid_when_training = valid_dataloader is not None
        accumulation_steps = c["train"]["grad_accumulation_steps"]

        self.model = self.model.to(device)

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
 
        with tqdm(total=(num_epochs-last_epoch) * (len(train_dataloader) + (len(valid_dataloader) if valid_dataloader else 0)), desc='Training...') as pbar:
            for epoch in range(last_epoch+1, num_epochs+1):
                self.model.train()
                train_loss = 0.0

                for i, (inputs, targets) in enumerate(train_dataloader, 1):
                    inputs, targets = inputs.to(device), targets.to(device)

                    if scaler:
                        device_type = 'cuda' if 'cuda' in c['device'] else 'cpu'
                        compute_type = torch.float16 if c['train']['scaler']['compute_type'] != 'bfloat16' else torch.bfloat16
                        with autocast(device_type, dtype=compute_type):
                            outputs = self.model(inputs)
                        
                            if accumulation_steps <= 0:
                                losses = criterion(targets, outputs)
                                for loss in losses:
                                    scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                            elif accumulation_steps > 0:
                                losses = criterion(targets, outputs)
                                for loss in losses:
                                    loss /= accumulation_steps
                                    scaler.scale(loss).backward()
                                if i % accumulation_steps == 0:
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad()
                    else:
                        outputs = self.model(inputs)
                        if accumulation_steps <= 0:
                            losses = criterion(targets, outputs)
                            for loss in losses:
                                loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        elif accumulation_steps > 0:
                            losses = criterion(targets, outputs)
                            for loss in losses:
                                loss /= accumulation_steps
                                loss.backward()
                            if i % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()

                    batch_loss = loss.item()
                    train_loss += batch_loss
                    pbar.update()
                    pbar.set_postfix({'batch_loss': batch_loss})

                    targets, outputs = self.postprocess(targets, outputs)
                    self.train_calculator.add_one_batch(
                        targets.detach().cpu().numpy(), 
                        outputs.detach().cpu().numpy())

                if lr_scheduler:
                    lr_scheduler.step()

                train_loss /= len(train_dataloader)
                train_losses.append(train_loss)
                pbar.update(0)
                pbar.set_postfix({'epoch_loss': train_loss})

                self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')
                self.train_calculator.finish_one_epoch()

                # validate
                if valid_dataloader:
                    valid_loss = 0.0
                    self.model.eval()

                    with torch.no_grad():
                        for inputs, targets in valid_dataloader:
                            inputs, targets = inputs.to(device), targets.to(device)

                            outputs = self.model(inputs)
                            loss = criterion(targets, outputs)

                            batch_loss = loss.item()
                            valid_loss += batch_loss
                            pbar.update(1)
                            pbar.set_postfix({'valid_batch_loss': batch_loss})

                            targets, outputs = self.postprocess(targets, outputs)
                            self.valid_calculator.add_one_batch(
                                targets.detach().cpu().float().numpy(), 
                                outputs.detach().cpu().float().numpy())

                    valid_loss /= len(valid_dataloader)
                    valid_losses.append(valid_loss)

                    pbar.set_postfix({'valid_epoch_loss': valid_loss})

                    self.logger.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')
                    self.valid_calculator.finish_one_epoch()

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
                            ext_path=save_model_ext_filename, optimizer=optimizer,        
                            scaler=scaler, lr_scheduler=lr_scheduler,
                            epoch=epoch, version=c['run_id'])
                    self.logger.info(f'save model params to {save_model_filename} and ext params to {save_model_ext_filename} when {epoch=}, {train_loss=}')

                target_loss = valid_loss if enable_valid_when_training else train_loss
                if target_loss < best_loss:
                    best_loss = target_loss
                    save_model(self.best_model_file_path, self.model, 
                                ext_path=self.best_model_ext_file_path,
                                optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler,
                                epoch=epoch, version=c['run_id'])
                    self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')

                save_model(self.last_model_file_path, self.model,
                        ext_path=self.last_model_ext_file_path,
                        optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, 
                        epoch=num_epochs, version=c['run_id'])
                self.logger.info(f'save model params to {self.last_model_file_path} and ext params to {self.last_model_ext_file_path} while finishing training')

        self.train_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)
        if enable_valid_when_training:
            self.valid_calculator.record_epochs(self.output_dir, n_epochs=num_epochs)

        self._print_table(enable_valid_when_training)

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

    def _print_table(self, enable_valid_when_training=False):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']

        train_mean_scores = self.train_calculator.metric_record
        valid_mean_scores = self.valid_calculator.metric_record if enable_valid_when_training else None
        train_metric_class_scores = self.train_calculator.mean_record
        valid_metric_class_scores = self.valid_calculator.mean_record if enable_valid_when_training else None
        
        styles = ALL_STYLES
        console = Console()

        table = Table(title='Metric Class Mean Score(Train)')
        table.add_column("Class/Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        
        for class_label in class_labels:
            table.add_row("Train/" + class_label, *[str(train_metric_class_scores[class_label][metric]) for metric in metric_labels])
        if valid_metric_class_scores:
            for class_label in class_labels:
                table.add_row("Valid/" + class_label, *[str(valid_metric_class_scores[class_label][metric]) for metric in metric_labels])
        console.print(table)

        table = Table(title='Summary of Metric(Train)')
        table.add_column("Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        table.add_row("Train", *[str(score) for score in train_mean_scores.values()])
        if valid_mean_scores:
            table.add_row("Valid", *[str(score) for score in valid_mean_scores.values()])
        console.print(table)

    def _save_after_train(self, num_epochs: int, train_losses: np.ndarray, valid_losses: np.ndarray|None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()
        labels = c['classes']

        self.data_saver.save_train_loss(train_losses)
        self.logger.info(f'Save train loss info under the {self.output_dir}')

        plot = Plot(1, 1)
        plot.subplot().epoch_loss(num_epochs, train_losses, labels, title='Train/Epoch-Loss').complete()
        plot.save(self.train_loss_image_path)

        self.logger.info(f'Save train/epoch_loss graph to {self.train_loss_image_path}')
        if valid_losses:
            self.data_saver.save_valid_loss(valid_losses)
            self.logger.info(f'Save valid/loss info under the {self.output_dir}')
            
            plot = Plot(1, 1)
            plot.subplot().epoch_loss(num_epochs, valid_losses, labels, title='Valid/Epoch-Loss').complete()
            plot.save(self.valid_loss_image_path)
            self.logger.info(f'Save valid/epoch_loss graph to {self.valid_loss_image_path}')

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
                    "dataset": c['dataset'],
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
        self.logger = logging.getLogger('test')
        self.data_saver = DataSaver(output_dir)
        
        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger, saver=self.data_saver)

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader):
        self.calculator.clear()

        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)

        for inputs, targets in tqdm(test_dataloader, desc="Testing..."):
            inputs, targets = inputs.to(device), targets.to(device)
            # (B, N, H, W) => N is n_classes
            outputs = self.model(inputs)

            targets, outputs = self.postprocess(targets, outputs)
            self.calculator.add_one_batch(
                targets.detach().cpu().numpy(),
                outputs.detach().cpu().numpy())

        self.calculator.record_batches(self.output_dir)

        self._print_table()
    @classmethod
    def postprocess(self, targets: torch.Tensor, outputs: torch.Tensor):
        targets[targets >= 0.5] = 1
        targets[targets < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        return targets, outputs

    def _print_table(self):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']

        test_mean_scores = self.calculator.metric_record
        test_metric_class_scores = self.calculator.mean_record
        
        styles = ALL_STYLES
        console = Console()

        table = Table(title='Metric Class Mean Score(Test)')
        table.add_column("Class/Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        for class_label in class_labels:
            table.add_row("Test/" + class_label, *[str(test_metric_class_scores[class_label][metric]) for metric in metric_labels])
        console.print(table)

        table = Table(title='Summary of Metric(Test)')
        table.add_column("Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        table.add_row("Test", *[str(score) for score in test_mean_scores.values()])
        console.print(table)


class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('predict')
        self.timer = Timer()

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])

        self.model = self.model.to(device)
        self.model.eval()

        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name

            with self.timer.timeit(task=input_filename + '.preprocess'):
                input_tensor, original_size = self.preprocess(input)

            with self.timer.timeit(task=input_filename + '.inference'):
                input_tensor = input_tensor.to(device)
                pred_tensor = self.model(input_tensor)

            with self.timer.timeit(task=input_filename + '.postprocess'):
                pred_np = self.postprocess(pred_tensor)

            self.record(input, pred_np, filename=input_filename, original_size=original_size)

        cost = self.timer.total_elapsed_time()
        self.logger.info(f'Predicting had cost {cost}s')

    def record(self, input: Path, pred: np.ndarray, **kwargs):
        output_filename = self.output_dir / kwargs['filename']
        original_size = kwargs['original_size'] # (W, H)

        pred_image = Image.fromarray(pred, mode='L')
        pred_image = pred_image.resize(original_size)

        pred_image.save(output_filename)

    def preprocess(self, input: Path) -> (torch.Tensor, tuple[int, int]):
        input = Image.open(input).convert('L')
        size = input.size # (H, W)
        transforms = get_transforms()
        input = transforms(input).unsqueeze(0)
        return input, size
    def postprocess(self, pred: torch.Tensor) -> np.ndarray:
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.astype(np.uint8)
        return pred
