from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console
import json
from functools import partial

from src.config import get_config, ALL_STYLES
from src.utils import EarlyStopping, DataSaver, save_model, select_postprocess_fn
from src.recorder.metric_recorder import MetricRecorder, ScoreAggregator
from src.visualizer.painter import Plot
from src.constants import TrainOutputFilenameEnv

class Trainer:
    def __init__(self, output_dir: Path, model: nn.Module, is_continue_mode: bool=False):
        self.output_dir = output_dir

        c = get_config()
        self.filename_env = TrainOutputFilenameEnv()
        self.filename_env.register(train_dir=self.output_dir, model=c["model"]["name"])
        self.filename_env.register(epoch=0, num_epochs=0)

        self.train_loss_image_path = self.filename_env.output_train_loss_filename
        self.valid_loss_image_path = self.filename_env.output_valid_loss_filename
        self.last_model_file_path = self.filename_env.output_last_model_filename
        self.best_model_file_path = self.filename_env.output_best_model_filename
        self.last_model_ext_file_path = self.filename_env.output_last_ext_model_filename
        self.best_model_ext_file_path = self.filename_env.output_best_ext_model_filename
        self.recovery_dir = self.filename_env.recovery_dir

        self.model = model

        self.filename_env.output_temp_model_filename.parent.mkdir(exist_ok=True, parents=True)

        self.recovery_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('train')
        self.data_saver = DataSaver(output_dir)

        class_labels = c['classes']
        metric_labels = c['metrics']
        self.train_metric_recorder = MetricRecorder(output_dir, class_labels, metric_labels, logger=self.logger, saver=self.data_saver)
        self.valid_metric_recorder = MetricRecorder(output_dir, class_labels, metric_labels, logger=self.logger, saver=self.data_saver)

        postprocess_name = c.get('postprocess', "")
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

        self.postprocess = select_postprocess_fn(postprocess_name)
        if is_continue_mode:
            self._recovery()

    def _recovery(self):
        with open(self.recovery_dir / 'train_metrics.json', 'r') as f:
            self.train_metric_recorder.epoch_metric_label_scores = json.load(f)
        with open(self.recovery_dir / 'valid_metrics.json', 'r') as f:
            self.valid_metric_recorder.epoch_metric_label_scores = json.load(f)

    def setup_trainer(self, criterion=None, optimizer=None, scaler: GradScaler | None=None, lr_scheduler=None, buildin_loss_calcuation: bool=False):
        self.criterion = criterion if not buildin_loss_calcuation else None
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch: int, num_epochs: int, train_dataloader: DataLoader, pbar: tqdm):
        c = get_config()
        device = torch.device(c['device'])
        accumulation_steps = c["train"]["grad_accumulation_steps"]
        enable_accumulation_step = accumulation_steps > 0
        if c['train'].get('lr_scheduler'):
            lr_scheduler_update_policy = c['train']['lr_scheduler'].get('update_policy', 'epoch')

        train_loss = 0.0
        self.model.train()

        if not enable_accumulation_step:
            self.optimizer.zero_grad()

        # Discard the last batch
        for batch, batch_inputs in enumerate(train_dataloader, 1):
            inputs, targets = batch_inputs['image'].to(device), batch_inputs['mask'].to(device)

            sum_loss = 0.0
            if self.scaler:
                device_type = 'cuda' if 'cuda' in c['device'] else 'cpu'
                compute_type = torch.bfloat16 if c['train']['scaler']['compute_type'] == 'bfloat16' else torch.float16
                with autocast(device_type, dtype=compute_type):
                    if self.criterion:
                        outputs = self.model(inputs)
                        loss = self.criterion(targets, outputs)
                    else:
                        outputs, loss = self.model(inputs)

                    if not enable_accumulation_step:
                        self.scaler.scale(loss).backward()
                        sum_loss += loss.item()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss /= accumulation_steps
                        self.scaler.scale(loss).backward()
                        sum_loss += loss.item()
                        if batch % accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
            else:
                if self.criterion:
                    outputs = self.model(inputs)
                    loss = self.criterion(targets, outputs)
                else:
                    outputs, loss = self.model(inputs)
                
                if not enable_accumulation_step:
                    loss.backward()
                    sum_loss += loss.item()
                    self.optimizer.step()
                else:
                    loss /= accumulation_steps
                    loss.backward()
                    sum_loss += loss.item()
                    if batch % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            # Add batch-wise scheduler update
            if self.lr_scheduler and lr_scheduler_update_policy == 'step':
                self.lr_scheduler.step()

            batch_loss = sum_loss
            train_loss += batch_loss
            pbar.update()
            pbar.set_postfix({'batch_loss': batch_loss, 'epoch': epoch})

            targets, outputs = self.postprocess(targets, outputs)
            self.train_metric_recorder.finish_one_batch(
                targets.detach().cpu().numpy(), 
                outputs.detach().cpu().numpy())

        train_loss /= len(train_dataloader)
        self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')
        self.train_metric_recorder.finish_one_epoch()
        return train_loss

    def _valid_epoch(self, epoch: int, num_epochs: int, valid_dataloader: DataLoader, pbar: tqdm):
        c = get_config()
        device = torch.device(c['device'])
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch_inputs in valid_dataloader:
                inputs, targets = batch_inputs['image'].to(device), batch_inputs['mask'].to(device)

                if self.scaler:
                    device_type = 'cuda' if 'cuda' in c['device'] else 'cpu'
                    compute_type = torch.bfloat16 if c['train']['scaler']['compute_type'] == 'bfloat16' else torch.float16
                    with autocast(device_type, dtype=compute_type):
                        if self.criterion:
                            outputs = self.model(inputs)
                            losses = self.criterion(targets, outputs)
                        else:
                            # buildin loss
                            outputs, losses = self.model(inputs)
                else:
                    if self.criterion:
                        outputs = self.model(inputs)
                        losses = self.criterion(targets, outputs)
                    else:
                        # buildin loss
                        outputs, losses = self.model(inputs)

                valid_sum_loss = losses.sum().item()
                batch_loss = valid_sum_loss
                valid_loss += batch_loss
                pbar.update(1)
                pbar.set_postfix({'valid_batch_loss': batch_loss, 'epoch': epoch})

                targets, outputs = self.postprocess(targets, outputs)
                self.valid_metric_recorder.finish_one_batch(
                    targets.detach().cpu().numpy(), 
                    outputs.detach().cpu().numpy())

        valid_loss /= len(valid_dataloader)
        self.logger.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')
        self.valid_metric_recorder.finish_one_epoch()
        return valid_loss

    def train(self, num_epochs: int, 
              train_dataloader,
              valid_dataloader = None, 
              *, early_stop: bool = False, last_epoch: int=0):
        c = get_config()
        device = torch.device(c['device'])
        save_period = c["train"]["save_period"] 
        if save_period >= 1:
            save_period = int(save_period)
        elif save_period <= 0:
            save_period = num_epochs
        else:
            save_period = int(save_period * num_epochs)

        save_recovery_period = c["train"]["save_recovery_period"] 
        if save_recovery_period >= 1:
            save_recovery_period = int(save_recovery_period)
        elif save_recovery_period <= 0:
            save_recovery_period = num_epochs
        else:
            save_recovery_period = int(save_recovery_period * num_epochs)

        accumulation_steps = c["train"]["grad_accumulation_steps"]
        enable_accumulation_step = accumulation_steps > 0

        self.model = self.model.to(device)

        save_model_preset = partial(save_model, model=self.model, optimizer=self.optimizer, scaler=self.scaler, lr_scheduler=self.lr_scheduler, version=c['run_id'])

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
                train_loss = self._train_epoch(epoch, num_epochs, train_dataloader, pbar)
                train_losses.append(train_loss)
                pbar.update(0)
                pbar.set_postfix({'epoch_loss': train_loss, 'epoch': epoch})

                # 调整 lr per one epoch
                if self.lr_scheduler:
                    # self.optimizer.step()
                    self.lr_scheduler.step()

                # recovery info
                if save_recovery_period > 0 and epoch % save_recovery_period == 0:
                    self.logger.info(f'save temporary recovery info when {epoch}/{num_epochs}')
                    with open(self.recovery_dir / 'train_metrics.json', 'w') as f:
                        json.dump(self.train_metric_recorder.epoch_metric_label_scores, f)
                    with open(self.recovery_dir / 'valid_metrics.json', 'w') as f:
                        json.dump(self.valid_metric_recorder.epoch_metric_label_scores, f)
                # checkpoint info
                if save_period > 0 and epoch % save_period == 0:
                    self.logger.info(f'save temporary checkpoint info when {epoch}/{num_epochs}')
                    self.filename_env.register(epoch=epoch, num_epochs=num_epochs)
                    save_model_filename = self.filename_env.output_temp_model_filename
                    save_model_ext_filename = self.filename_env.output_temp_ext_model_filename
                    save_model_preset(save_model_filename,             
                            ext_path=save_model_ext_filename,        
                            epoch=epoch, loss=train_loss)
                    self.logger.info(f'save model params to {save_model_filename} and ext params to {save_model_ext_filename} when {epoch=}, {train_loss=}')
                # metrics info
                if save_period > 0 and epoch % save_period == 0:
                    self.logger.info(f'save temporary losses and metrics info when {epoch}/{num_epochs}')
                    self._save_train_info(epoch, num_epochs, np.array(train_losses, dtype=np.float64), np.array(valid_losses, dtype=np.float64) if valid_dataloader else None)

                # validate
                if valid_dataloader:
                    valid_loss = self._valid_epoch(epoch, num_epochs, valid_dataloader, pbar)
                    valid_losses.append(valid_loss)

                    pbar.update(0)
                    pbar.set_postfix({'valid_epoch_loss': valid_loss, 'epoch': epoch})
                    
                    # save best model
                    target_loss = valid_loss
                    if target_loss < best_loss:
                        best_loss = target_loss
                        save_model_preset(self.best_model_file_path, 
                                    ext_path=self.best_model_ext_file_path,
                                    epoch=epoch, loss=best_loss)
                        self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')

                    if early_stopper:
                        early_stopper(valid_loss)
                        if early_stopper.is_stopped():
                            self.logger.info("Early stopping")
                            break
                # else:
                #     best_loss = train_loss
                #     save_model(self.best_model_file_path, self.model, 
                #                 ext_path=self.best_model_ext_file_path,
                #                 optimizer=self.optimizer, scaler=self.scaler, lr_scheduler=self.lr_scheduler,
                #                 epoch=epoch, version=c['run_id'], loss=best_loss)
                #     self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')

                save_model_preset(self.last_model_file_path, 
                        ext_path=self.last_model_ext_file_path,
                        epoch=num_epochs, loss=train_loss)

            # accumulation_step
            # if policy is to use the remain batch grad, it is accumulation_policy == 'remain leave'
            # if enable_accumulation_step:
            #     if (len(train_dataloader) * num_epochs) % accumulation_steps != 0:
            #         if self.scaler:
            #             self.scaler.step(self.optimizer)
            #             self.scaler.update()
            #             self.optimizer.zero_grad()
            #         else:
            #             self.optimizer.step()
            #             self.optimizer.zero_grad()

            #         save_model(self.last_model_file_path, **model_args,
            #                 ext_path=self.last_model_ext_file_path, **ext_model_args,
            #                 epoch=num_epochs, loss=best_loss)
            #         self.logger.info(f'save model params to {self.last_model_file_path} and ext params to {self.last_model_ext_file_path} while finishing training')

        self._print_table(valid_dataloader is not None)

        self._save_train_info(num_epochs, num_epochs, np.array(train_losses, dtype=np.float64), np.array(valid_losses, dtype=np.float64) if valid_dataloader else None)
        self._wandb_save(train_losses, valid_losses, self.optimizer, self.scaler, self.lr_scheduler)

    # TODO: Need to optimize 
    def _print_table(self, enable_valid_when_training=False):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']

        train_scores = ScoreAggregator(self.train_metric_recorder.epoch_metric_label_scores)
        train_mean_scores = train_scores.ml1_mean
        train_metric_class_scores = (train_scores.mc1_mean, train_scores.mc1_std)
        valid_scores = ScoreAggregator(self.valid_metric_recorder.epoch_metric_label_scores) if enable_valid_when_training else None
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

    # TODO: Need to optimize 
    def _save_train_info(self, epochs: int, num_epochs: int, train_losses, valid_losses=None):
        self.data_saver.save_train_loss(train_losses)

        plot = Plot(1, 1)
        plot = plot.subplot().epoch_loss(epochs, num_epochs, train_losses, 'train', title='Train/Epoch-Loss')
        if valid_losses is not None:
            self.data_saver.save_valid_loss(valid_losses)
            plot = plot.epoch_loss(epochs, num_epochs, valid_losses, 'valid', title='Valid/Epoch-Loss')
            self.logger.info(f'Save train & valid loss info under the {self.output_dir}')
        else:
            self.logger.info(f'Save train loss info under the {self.output_dir}')
        plot = plot.complete()
        plot.save(self.train_loss_image_path)

        self.train_metric_recorder.record_epochs(epochs, n_epochs=num_epochs)
        if valid_losses:
            self.valid_metric_recorder.record_epochs(epochs, n_epochs=num_epochs)

        self.logger.info(f'Save train/epoch_loss graph to {self.train_loss_image_path}')

    def _wandb_save(self, train_losses, valid_losses=None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()
        if c['private']['wandb']:
            train_c = c['train']
            import wandb
            wandb.log({
                "train": {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "loss_image": self.train_loss_image_path,
                },
                "config": c,
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
