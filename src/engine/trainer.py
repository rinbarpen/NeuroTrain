from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
import logging
from rich.table import Table
from rich.console import Console
import json
from functools import partial

from src.config import get_config, ALL_STYLES
from src.utils import EarlyStopping, save_model, select_postprocess_fn
from src.recorder import MeterRecorder, DataSaver
from src.visualizer.painter import Plot
from src.constants import TrainOutputFilenameEnv

class Trainer:
    def __init__(self, output_dir: Path, model: nn.Module, is_continue_mode: bool=False):
        self.output_dir = output_dir

        c = get_config()
        self.filename_env = TrainOutputFilenameEnv()
        self.filename_env.register(train_dir=self.output_dir, model=c["model"]["name"])
        self.filename_env.register(epoch=0, num_epochs=0)
        self.filename_env.prepare_dir()

        self.model = model

        self.logger = logging.getLogger('train')
        self.data_saver = DataSaver()

        class_labels = c['classes']
        metric_labels = c['metrics']
        self.train_metric_recorder = MeterRecorder(class_labels, metric_labels, logger=self.logger, saver=self.data_saver, prefix="train_")
        self.valid_metric_recorder = MeterRecorder(class_labels, metric_labels, logger=self.logger, saver=self.data_saver, prefix="valid_")

        postprocess_name = c.get('postprocess', "")
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

        self.postprocess = select_postprocess_fn(postprocess_name)
        if is_continue_mode:
            self._recovery()

    def _recovery(self):
        # MeterBasedRecorder uses epoch_meters format
        # Recovery mechanism for MeterBasedRecorder
        try:
            recovery_dir = self.filename_env.recovery_dir
            pointer_file = recovery_dir / "recovery_info.json"

            def _load_and_restore(target_file: Path):
                with target_file.open('r') as f:
                    recovery_data = json.load(f)
                # 恢复epoch_meters数据
                if 'train_epoch_scores' in recovery_data and recovery_data['train_epoch_scores'] is not None:
                    self._restore_epoch_meters(self.train_metric_recorder, recovery_data['train_epoch_scores'])
                if 'valid_epoch_scores' in recovery_data and recovery_data['valid_epoch_scores'] is not None:
                    self._restore_epoch_meters(self.valid_metric_recorder, recovery_data['valid_epoch_scores'])
                self.logger.info(f"Successfully recovered meter-based recorder data from {target_file}")

            if pointer_file.exists():
                _load_and_restore(pointer_file)
                return

            # 兼容旧版本：若指针文件不存在，则查找最新的 recovery_epoch_*.json
            candidates = list(recovery_dir.glob('recovery_epoch_*.json'))
            if candidates:
                def _extract_epoch(p: Path) -> int:
                    name = p.stem  # e.g., recovery_epoch_12
                    try:
                        return int(name.rsplit('_', 1)[1])
                    except Exception:
                        return -1
                latest = max(candidates, key=_extract_epoch)
                if latest.exists():
                    _load_and_restore(latest)
                    return

            self.logger.info("No recovery data found")
        except Exception as e:
            self.logger.warning(f"Recovery failed: {e}")
    
    def _restore_epoch_meters(self, recorder: MeterRecorder, epoch_scores_data):
        """从恢复数据中重建epoch_meters"""
        try:
            for metric_label in recorder.metric_labels:
                for class_label in recorder.class_labels:
                    if metric_label in epoch_scores_data and class_label in epoch_scores_data[metric_label]:
                        scores = epoch_scores_data[metric_label][class_label]
                        meter = recorder.epoch_meters[metric_label][class_label]
                        # 清空现有数据并重新添加
                        meter.reset()
                        for score in scores:
                            meter.update(score)
        except Exception as e:
            self.logger.warning(f"Failed to restore epoch meters: {e}")

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
        lr_scheduler_update_policy = c['train'].get('lr_scheduler', {}).get('update_policy', 'epoch')
        train_loss = 0.0
        step_losses = []
        step_lrs = []
        self.model.train()

        if not enable_accumulation_step and self.optimizer is not None:
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
                        if self.optimizer is not None:
                            self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss /= accumulation_steps
                        self.scaler.scale(loss).backward()
                        sum_loss += loss.item()
                        if batch % accumulation_steps == 0:
                            if self.optimizer is not None:
                                self.scaler.step(self.optimizer)
                            self.scaler.update()
                            if self.optimizer is not None:
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
                    if self.optimizer is not None:
                        self.optimizer.step()
                else:
                    loss /= accumulation_steps
                    loss.backward()
                    sum_loss += loss.item()
                    if batch % accumulation_steps == 0:
                        if self.optimizer is not None:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

            # Add batch-wise scheduler update
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler and lr_scheduler_update_policy == 'step':
                self.lr_scheduler.step()

            batch_loss = sum_loss
            train_loss += batch_loss
            step_losses.append(batch_loss)
            
            # 收集当前step的学习率
            if self.optimizer is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                step_lrs.append(current_lr)
            
            pbar.update()
            pbar.set_postfix({'batch_loss': batch_loss, 'epoch': epoch, 'step': batch + epoch * len(train_dataloader)})

            if self.postprocess is not None:
                targets, outputs = self.postprocess(targets, outputs)
            self.train_metric_recorder.finish_one_batch(
                targets.detach().cpu().numpy(), 
                outputs.detach().cpu().numpy())

        train_loss /= len(train_dataloader)
        self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')
        # 保存step级loss
        try:
            self.data_saver.save_train_step_loss(np.array(step_losses, dtype=np.float64), filename=self.filename_env.output_train_step_loss_filename.as_posix())
        except Exception as e:
            self.logger.warning(f"Failed to save step losses: {e}")
        self.train_metric_recorder.finish_one_epoch()
        return train_loss, step_losses, step_lrs

    def _valid_epoch(self, epoch: int, num_epochs: int, valid_dataloader: DataLoader, pbar: tqdm):
        c = get_config()
        device = torch.device(c['device'])
        self.model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            step_losses = []
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
                step_losses.append(batch_loss)
                pbar.update(1)
                pbar.set_postfix({'valid_batch_loss': batch_loss, 'epoch': epoch})

                if self.postprocess is not None:
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
        
        # 收集step级别的数据用于可视化
        all_step_losses = []
        all_step_lrs = []
        epoch_lrs = []

        with tqdm(total=(num_epochs-last_epoch) * (len(train_dataloader) + (len(valid_dataloader) if valid_dataloader else 0)), desc='Training...') as pbar:
            for epoch in range(last_epoch+1, num_epochs+1):
                train_loss, epoch_step_losses, epoch_step_lrs = self._train_epoch(epoch, num_epochs, train_dataloader, pbar)
                train_losses.append(train_loss)
                
                # 收集step级别的数据
                all_step_losses.extend(epoch_step_losses)
                all_step_lrs.extend(epoch_step_lrs)
                
                # 收集epoch级别的学习率
                if self.optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    epoch_lrs.append(current_lr)
                
                pbar.update(0)
                pbar.set_postfix({'epoch_loss': train_loss, 'epoch': epoch})

                # 调整 lr per one epoch
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    # self.optimizer.step()
                    self.lr_scheduler.step()

                # recovery info
                if save_recovery_period > 0 and epoch % save_recovery_period == 0:
                    self.logger.info(f'save temporary recovery info when {epoch}/{num_epochs}')
                    self._save_recovery_info(epoch, num_epochs, train_losses, valid_losses if valid_dataloader else None)
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
                    self._save_train_info(epoch, num_epochs, train_losses, valid_losses if valid_dataloader else None)

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
                        save_model_preset(self.filename_env.output_best_model_filename, 
                                    ext_path=self.filename_env.output_best_ext_model_filename,
                                    epoch=epoch, loss=best_loss)
                        self.logger.info(f'save model params to {self.filename_env.output_best_model_filename} and ext params to {self.filename_env.output_best_ext_model_filename} when {epoch=}, {best_loss=}')

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

                save_model_preset(self.filename_env.output_last_model_filename, 
                        ext_path=self.filename_env.output_last_ext_model_filename,
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

        self._draw_visualizations(all_step_losses, all_step_lrs, epoch_lrs, num_epochs)

        self._save_train_info(num_epochs, num_epochs, train_losses, valid_losses if valid_dataloader else None)
        self._wandb_save(train_losses, valid_losses, self.optimizer, self.scaler, self.lr_scheduler)

    def _draw_visualizations(self, all_step_losses, all_step_lrs, epoch_lrs, num_epochs):
        """生成可视化图表"""
        try:
            from src.visualizer.painter import Plot

            if isinstance(all_step_losses, (list, tuple)):
                all_step_losses = np.array(all_step_losses, dtype=np.float64)
            if isinstance(all_step_lrs, (list, tuple)):
                all_step_lrs = np.array(all_step_lrs, dtype=np.float64)
            if isinstance(epoch_lrs, (list, tuple)):
                epoch_lrs = np.array(epoch_lrs, dtype=np.float64)
            
            # 1. 生成step_loss图
            if all_step_losses and len(all_step_losses) > 0:
                step_loss_plot = Plot(1, 1)
                step_loss_plot.subplot().step_loss(
                    all_step_losses,
                    label='Train Loss',
                    title='Train/Step-Loss'
                ).complete()
                
                step_loss_path = self.filename_env.output_train_step_loss_image_filename
                step_loss_plot.save(step_loss_path)
                self.logger.info(f'Save step loss graph to {step_loss_path}')
            
            # 2. 生成step_lr图
            if all_step_lrs and len(all_step_lrs) > 0:
                step_lr_plot = Plot(1, 1)
                step_lr_plot.subplot().step_lr(
                    all_step_lrs,
                    label='Learning Rate',
                    title='Train/Step-LR'
                ).complete()
                
                step_lr_path = self.filename_env.output_lr_by_step_image_filename
                step_lr_plot.save(step_lr_path)
                self.logger.info(f'Save step lr graph to {step_lr_path}')
            
            # 3. 生成epoch_lr图
            if epoch_lrs and len(epoch_lrs) > 0:
                epoch_lr_plot = Plot(1, 1)
                epoch_lr_plot.subplot().epoch_lr(
                    len(epoch_lrs), num_epochs,
                    epoch_lrs,
                    label='Learning Rate',
                    title='Train/Epoch-LR'
                ).complete()
                
                epoch_lr_path = self.filename_env.output_lr_by_epoch_image_filename
                epoch_lr_plot.save(epoch_lr_path)
                self.logger.info(f'Save epoch lr graph to {epoch_lr_path}')
                
        except Exception as e:
            self.logger.warning(f"Failed to generate step visualizations: {e}")

    def _print_table(self, enable_valid_when_training=False):
        c = get_config()
        class_labels  = c['classes']
        metric_labels = c['metrics']

        # 使用MeterBasedRecorder的内置统计功能
        train_mean_scores = self.train_metric_recorder.get_epoch_metrics()
        train_metric_class_scores = (
            self.train_metric_recorder._compute_mc1(np.mean),
            self.train_metric_recorder._compute_mc1(np.std)
        )
        
        valid_mean_scores = None
        valid_metric_class_scores = None
        if enable_valid_when_training:
            valid_mean_scores = self.valid_metric_recorder.get_epoch_metrics()
            valid_metric_class_scores = (
                self.valid_metric_recorder._compute_mc1(np.mean),
                self.valid_metric_recorder._compute_mc1(np.std)
            )

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

    def _save_train_info(self, epochs: int, num_epochs: int, train_losses, valid_losses=None):
        """保存训练信息，包括损失图表和指标记录"""
        try:
            # 数据验证
            if train_losses is None or len(train_losses) == 0:
                self.logger.warning("No training losses to save")
                return
            if isinstance(train_losses, (list, tuple)):
                train_losses = np.array(train_losses, dtype=np.float64)
            if valid_losses is not None and isinstance(valid_losses, (list, tuple)):
                valid_losses = np.array(valid_losses, dtype=np.float64)
            
            self.data_saver.save_train_loss(train_losses, self.filename_env.output_train_epoch_loss_details_filename.as_posix())

            # 保存训练损失图表到train目录
            train_plot = Plot(1, 1)
            train_plot = train_plot.subplot().epoch_loss(epochs, num_epochs, train_losses, 'train', title='Train/Epoch-Loss')
            train_plot = train_plot.complete()
            train_loss_path = self.filename_env.output_train_epoch_loss_image_filename
            train_plot.save(train_loss_path)
            self.logger.info(f'Save train loss graph to {train_loss_path} under the {self.output_dir}')

            # 如果有验证损失，分别保存验证损失图表和组合图表
            if valid_losses is not None and len(valid_losses) > 0:
                self.data_saver.save_valid_loss(valid_losses, self.filename_env.output_valid_epoch_loss_details_filename.as_posix())
                
                # 保存验证损失图表到valid目录
                valid_plot = Plot(1, 1)
                valid_plot = valid_plot.subplot().epoch_loss(epochs, num_epochs, valid_losses, 'valid', title='Valid/Epoch-Loss')
                valid_plot = valid_plot.complete()
                valid_loss_path = self.filename_env.output_valid_epoch_loss_image_filename
                valid_plot.save(valid_loss_path)
                self.logger.info(f'Save valid loss graph to {valid_loss_path} under the {self.output_dir}')
                
                # 保存组合损失图表到公共目录（train目录）
                combined_plot = Plot(1, 1)
                combined_plot = combined_plot.subplot().epoch_loss(epochs, num_epochs, train_losses, 'train', title='Train&Valid/Epoch-Loss')
                combined_plot = combined_plot.epoch_loss(epochs, num_epochs, valid_losses, 'valid', title='Train&Valid/Epoch-Loss')
                combined_plot = combined_plot.complete()
                # 保存为 epoch_loss.png（名称修正）
                self.filename_env.register(epoch=epochs, num_epochs=num_epochs)
                epoch_loss_path = self.filename_env.output_epoch_loss_image_filename
                combined_plot.save(epoch_loss_path)
                self.logger.info(f'Save combined train & valid loss graph to {epoch_loss_path} under the {self.output_dir}')
            else:
                self.logger.info(f'Save train loss info under the {self.output_dir}')

            # 记录学习率（按epoch）
            try:
                if self.optimizer is not None:
                    current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                    self.data_saver.save_lr_by_epoch(np.array(current_lrs, dtype=np.float64), self.filename_env.output_lr_by_epoch_filename.as_posix())
            except Exception as e:
                self.logger.warning(f"Failed to save lr by epoch: {e}")

            # 记录训练指标
            try:
                self.train_metric_recorder.record_epochs(
                    epochs, 
                    n_epochs=num_epochs,
                    output_dir=self.output_dir / "train",
                    filenames={
                        'all_metric_by_class': str(self.output_dir / "train" / "{class_label}" / "all_metric.csv"),
                        'mean_metric_by_class': str(self.output_dir / "train" / "{class_label}" / "mean_metric.csv"),
                        'std_metric_by_class': str(self.output_dir / "train" / "{class_label}" / "std_metric.csv"),
                        'mean_metric': str(self.output_dir / "train" / "mean_metric.csv"),
                        'std_metric': str(self.output_dir / "train" / "std_metric.csv"),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to record train metrics: {e}")
            
            # 记录验证指标（如果存在验证数据）
            if valid_losses is not None:
                try:
                    self.valid_metric_recorder.record_epochs(
                        epochs, 
                        n_epochs=num_epochs,
                        output_dir=self.output_dir / "valid",
                        filenames={
                            'all_metric_by_class': str(self.output_dir / "valid" / "{class_label}" / "all_metric.csv"),
                            'mean_metric_by_class': str(self.output_dir  / "valid" / "{class_label}" / "mean_metric.csv"),
                            'std_metric_by_class': str(self.output_dir  / "valid" / "{class_label}" / "std_metric.csv"),
                            'mean_metric': str(self.output_dir  / "valid" / "mean_metric.csv"),
                            'std_metric': str(self.output_dir  / "valid" / "std_metric.csv"),
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to record valid metrics: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to save train info: {e}")
            # 不抛出异常，避免中断训练过程

    def _save_recovery_info(self, epoch: int, num_epochs: int, train_losses, valid_losses=None):
        """保存recovery信息，包括训练状态和指标数据"""
        try:
            # 数据验证
            if not isinstance(epoch, int) or epoch < 0:
                self.logger.warning(f"Invalid epoch value: {epoch}, skipping recovery save")
                return
            
            if not isinstance(num_epochs, int) or num_epochs <= 0:
                self.logger.warning(f"Invalid num_epochs value: {num_epochs}, skipping recovery save")
                return
            
            if isinstance(train_losses, np.ndarray):
                train_losses = train_losses.tolist()
            if valid_losses is not None and isinstance(valid_losses, np.ndarray):
                valid_losses = valid_losses.tolist()
            
            # 确保recovery目录存在
            # self.recovery_dir.mkdir(exist_ok=True, parents=True)
            
            recovery_data = {
                'epoch': epoch,
                'num_epochs': num_epochs,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_epoch_scores': self.train_metric_recorder.get_epoch_scores(),
                'valid_epoch_scores': self.valid_metric_recorder.get_epoch_scores() if valid_losses is not None else None,
            }
            
            # 保存到recovery目录（按epoch文件）
            recovery_file = self.filename_env.recovery_dir / f"recovery_epoch_{epoch}.json"
            with open(recovery_file, 'w') as f:
                json.dump(recovery_data, f, indent=2)

            # 同步写一份指针文件，供启动时快速恢复最新状态
            pointer_file = self.filename_env.recovery_dir / "recovery_info.json"
            with open(pointer_file, 'w') as f:
                json.dump(recovery_data, f, indent=2)
            
            # 保存指标数据
            try:
                # MetricRecorder没有save_meters方法，直接记录即可
                pass
            except Exception as e:
                self.logger.warning(f"Failed to save metric data for recovery: {e}")
            
            self.logger.info(f'Save recovery info to {recovery_file} (and updated {pointer_file})')
            
        except Exception as e:
            self.logger.error(f"Failed to save recovery info at epoch {epoch}: {e}")
            # 不抛出异常，避免中断训练过程

    def _wandb_save(self, train_losses, valid_losses=None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()
        if c['private']['wandb']:
            train_c = c['train']
            import wandb
            wandb.log({
                "train": {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
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
