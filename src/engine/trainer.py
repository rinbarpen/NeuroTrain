from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console

from src.config import get_config, ALL_STYLES
from src.utils import EarlyStopping, DataSaver, ScoreAggregator, save_model, Timer, ScoreCalculator, Plot, CombineCriterion, select_postprocess_fn

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

        self.save_model_dir.mkdir()
        self.logger = logging.getLogger('train')
        self.data_saver = DataSaver(output_dir)

        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.train_calculator = ScoreCalculator(output_dir, class_labels, metric_labels, logger=self.logger, saver=self.data_saver)
        self.valid_calculator = ScoreCalculator(output_dir, class_labels, metric_labels, logger=self.logger, saver=self.data_saver)

        postprocess_name = c.get('postprocess', "")
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

        self.postprocess = select_postprocess_fn(postprocess_name)
        # TODO:
        # if is_continue_mode:
        #     recovery()

    def train(self, num_epochs: int, 
              criterion, 
              optimizer,
              train_dataloader,
              valid_dataloader = None, 
              scaler = None, 
              lr_scheduler = None,
              *, early_stop: bool = False, last_epoch: int=0):
        c = get_config()
        device = torch.device(c['device'])
        save_every_n_epoch = c["train"]["save_every_n_epoch"]
        accumulation_steps = c["train"]["grad_accumulation_steps"]
        enable_accumulation_step = accumulation_steps > 0

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

                if not enable_accumulation_step:
                    optimizer.zero_grad()

                for i, (inputs, targets) in enumerate(train_dataloader, 1):
                    inputs, targets = inputs.to(device), targets.to(device)

                    sum_loss = 0.0
                    if scaler:
                        device_type = 'cuda' if 'cuda' in c['device'] else 'cpu'
                        compute_type = torch.bfloat16 if c['train']['scaler']['compute_type'] == 'bfloat16' else torch.float16
                        with autocast(device_type, dtype=compute_type):
                            outputs = self.model(inputs)

                            if not enable_accumulation_step:
                                loss = criterion(targets, outputs)
                                scaler.scale(loss).backward()
                                sum_loss += loss.item()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss = criterion(targets, outputs)
                                loss /= accumulation_steps
                                scaler.scale(loss).backward()
                                sum_loss += loss.item()
                                if i % accumulation_steps == 0:
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad()
                    else:
                        outputs = self.model(inputs)
                        if not enable_accumulation_step:
                            loss = criterion(targets, outputs)
                            loss.backward()
                            sum_loss += loss.item()
                            optimizer.step()
                        else:
                            loss = criterion(targets, outputs)
                            loss /= accumulation_steps
                            loss.backward()
                            sum_loss += loss.item()
                            if i % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()

                    batch_loss = sum_loss
                    train_loss += batch_loss
                    pbar.update()
                    pbar.set_postfix({'batch_loss': batch_loss, 'epoch': epoch})

                    targets, outputs = self.postprocess(targets, outputs)
                    self.train_calculator.finish_one_batch(
                        targets.detach().cpu().numpy(), 
                        outputs.detach().cpu().numpy())

                if lr_scheduler:
                    lr_scheduler.step()

                train_loss /= len(train_dataloader)
                train_losses.append(train_loss)
                pbar.update(0)
                pbar.set_postfix({'epoch_loss': train_loss, 'epoch': epoch})

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
                            losses = criterion(targets, outputs)
                            valid_sum_loss = 0.0
                            for loss in losses:
                                valid_sum_loss += loss.item()

                            batch_loss = valid_sum_loss
                            valid_loss += batch_loss
                            pbar.update(1)
                            pbar.set_postfix({'valid_batch_loss': batch_loss, 'epoch': epoch})

                            targets, outputs = self.postprocess(targets, outputs)
                            self.valid_calculator.finish_one_batch(
                                targets.detach().cpu().numpy(), 
                                outputs.detach().cpu().numpy())

                    valid_loss /= len(valid_dataloader)
                    valid_losses.append(valid_loss)

                    pbar.update(0)
                    pbar.set_postfix({'valid_epoch_loss': valid_loss, 'epoch': epoch})

                    self.logger.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')
                    self.valid_calculator.finish_one_epoch()

                    if early_stopper:
                        early_stopper(valid_loss)
                        if early_stopper.is_stopped():
                            self.logger.info("Early stopping")
                            break

                # save every n epoch
                if save_every_n_epoch > 0 and epoch % save_every_n_epoch == 0:
                    save_model_filename = self.save_model_dir / f'{c["model"]["name"]}-{epoch}of{num_epochs}.pt'
                    save_model_ext_filename = self.save_model_dir / f'{c["model"]["name"]}-{epoch}of{num_epochs}-ext.pt'
                    save_model(save_model_filename, self.model,             
                            ext_path=save_model_ext_filename, optimizer=optimizer,        
                            scaler=scaler, lr_scheduler=lr_scheduler,
                            epoch=epoch, version=c['run_id'], loss=best_loss)
                    self.logger.info(f'save model params to {save_model_filename} and ext params to {save_model_ext_filename} when {epoch=}, {train_loss=}')

                # save best model
                if valid_dataloader:
                    target_loss = valid_loss
                    if target_loss < best_loss:
                        best_loss = target_loss
                        save_model(self.best_model_file_path, self.model, 
                                    ext_path=self.best_model_ext_file_path,
                                    optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler,
                                    epoch=epoch, version=c['run_id'], loss=best_loss)
                        self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')
                else:
                    best_loss = train_loss
                    save_model(self.best_model_file_path, self.model, 
                                ext_path=self.best_model_ext_file_path,
                                optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler,
                                epoch=epoch, version=c['run_id'], loss=best_loss)
                    self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')


                save_model(self.last_model_file_path, self.model,
                        ext_path=self.last_model_ext_file_path,
                        optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, 
                        epoch=num_epochs, version=c['run_id'], loss=train_loss)

            # accumulation_step
            if enable_accumulation_step:
                if (len(train_dataloader) * num_epochs) % accumulation_steps != 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

            save_model(self.last_model_file_path, self.model,
                    ext_path=self.last_model_ext_file_path,
                    optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler, 
                    epoch=num_epochs, version=c['run_id'], loss=best_loss)
            self.logger.info(f'save model params to {self.last_model_file_path} and ext params to {self.last_model_ext_file_path} while finishing training')

        self.train_calculator.record_epochs(n_epochs=num_epochs)
        if valid_dataloader:
            self.valid_calculator.record_epochs(n_epochs=num_epochs)

        self._print_table(valid_dataloader is not None)

        train_losses = np.array(train_losses, dtype=np.float64)
        valid_losses = np.array(valid_losses, dtype=np.float64) if valid_dataloader else None
        self._save_after_train(num_epochs, train_losses, valid_losses, optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler)

    def _print_table(self, enable_valid_when_training=False):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']

        train_scores = ScoreAggregator(self.train_calculator.epoch_metric_label_scores)
        train_mean_scores = train_scores.ml1_mean
        train_metric_class_scores = (train_scores.mc1_mean, train_scores.mc1_std)
        valid_scores = ScoreAggregator(self.valid_calculator.epoch_metric_label_scores) if enable_valid_when_training else None
        valid_mean_scores = valid_scores.ml1_mean if valid_scores else None
        valid_metric_class_scores = (valid_scores.mc1_mean, valid_scores.mc1_std) if valid_scores else None

        styles = ALL_STYLES
        console = Console()

        table = Table(title='Metric Class Mean Score(Train)')
        table.add_column("Class/Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        for class_label in class_labels:
            mean_scores = train_metric_class_scores[0]
            std_scores = train_metric_class_scores[1]
            table.add_row("Train/" + class_label, *[f'{mean_scores[metric][class_label]:.3f} ± {std_scores[metric][class_label]:.3f}' for metric in metric_labels])
        if valid_metric_class_scores:
            for class_label in class_labels:
                mean_scores = valid_metric_class_scores[0]
                std_scores = valid_metric_class_scores[1]
                table.add_row("Valid/" + class_label, *[f'{mean_scores[metric][class_label]:.3f} ± {std_scores[metric][class_label]:.3f}' for metric in metric_labels])
        console.print(table)

        table = Table(title='Summary of Metric(Train)')
        table.add_column("Metric", justify="center")
        for metric, style in zip(metric_labels, styles[:len(metric_labels)]):
            table.add_column(metric, justify="center", style=style)
        table.add_row("Train", *[f'{score:.3f}' for score in train_mean_scores.values()])
        if valid_mean_scores:
            table.add_row("Valid", *[f'{score:.3f}' for score in valid_mean_scores.values()])
        console.print(table)

    def _save_after_train(self, num_epochs: int, train_losses, valid_losses=None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()

        self.data_saver.save_train_loss(train_losses)

        plot = Plot(1, 1)
        plot = plot.subplot().epoch_loss(num_epochs, train_losses, 'train', title='Train/Epoch-Loss')
        if valid_losses is not None:
            self.data_saver.save_valid_loss(valid_losses)
            plot = plot.epoch_loss(num_epochs, valid_losses, 'valid', title='Train&Valid/Epoch-Loss').complete()
            self.logger.info(f'Save train & valid loss info under the {self.output_dir}')
        else:
            plot = plot.complete()
            self.logger.info(f'Save train loss info under the {self.output_dir}')
        plot.save(self.train_loss_image_path)

        self.logger.info(f'Save train/epoch_loss graph to {self.train_loss_image_path}')

        if c['private']['wandb']:
            train_c = c['train']
            import wandb
            wandb.log({
                "train": {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "loss_image": self.train_loss_image_path,
                },
                "config": {
                    "batch_size": train_c['batch_size'],
                    "epoch": train_c['epoch'],
                    "dataset": c['dataset'],
                    "save_every_n_epoch": train_c['save_every_n_epoch'],
                    "early_stopping": {
                        **{k: v for k, v in train_c['early_stopping'].items()},
                    } if train_c.get('early_stopping') else None,
                },
                "model_data": {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "lr_scheduler": {
                        'weights': lr_scheduler.state_dict(),
                        **{k: v for k, v in train_c['lr_scheduler'].items()},
                    } if lr_scheduler else None,
                },
            })
