from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
import logging
import json
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Tuple


class TrainingStopRequested(RuntimeError):
    """Internal exception to unwind control flow when stop is requested."""
    pass

from src.config import get_config
from src.utils.ddp_utils import is_main_process
from src.utils import (
    EarlyStopping,
    save_model,
    select_postprocess_fn,
    run_async_task,
    TrainingStopManager,
    reset_peak_memory_stats,
    log_memory_cost,
)
from src.utils.ndict import ModelOutput, NDict
from src.utils.metric_table import print_metric_scores_table
from src.utils.progress_bar import ProgressBar
from src.recorder import MeterRecorder, DataSaver, LossTracker
from src.visualizer.painter import Plot
from src.constants import TrainOutputFilenameEnv
from src.monitor import TrainingMonitor, ProgressTracker, MonitorApiClient
from src.monitor import create_web_monitor

class Trainer:
    def __init__(
        self,
        output_dir: Path,
        model: nn.Module,
        is_continue_mode: bool = False,
        *,
        recovery_dir: Path | None = None,
        async_executor: ThreadPoolExecutor | None = None,
    ):
        self.output_dir = output_dir

        c = get_config()
        self.device = torch.device(c['device'])
        
        class_labels = c['classes']
        all_metrics = c['metrics']
        main_metric = c.get('main_metric', 'loss')
        if main_metric == 'loss' or not main_metric:
            train_valid_metric_labels = []
        else:
            if main_metric not in all_metrics:
                logging.getLogger('train').warning(f"main_metric '{main_metric}' not in config metrics {all_metrics}; using loss only for train/valid")
                train_valid_metric_labels = []
            else:
                train_valid_metric_labels = [main_metric]
        
        self.filename_env = TrainOutputFilenameEnv()
        self.filename_env.register(train_dir=self.output_dir, model=c["model"]["name"], class_labels=class_labels)
        self.filename_env.register(epoch=0, num_epochs=0)
        self.filename_env.prepare_dir()

        self.model = model
        self.is_continue_mode = is_continue_mode
        self._recovery_dir = Path(recovery_dir) if recovery_dir is not None else None

        self.logger = logging.getLogger('train')
        self._stop_manager = TrainingStopManager()
        self._stop_manager.set_logger(self.logger)
        self._stop_manager.install_signal_handlers()
        self._stop_manager.register_stop_file(self.filename_env.stop_flag_file)
        self._stop_saved = False
        self._save_model_fn = None
        self.data_saver = DataSaver()
        self.train_metric_recorder = MeterRecorder(class_labels, train_valid_metric_labels, logger=self.logger, saver=self.data_saver, prefix="train_")
        self.valid_metric_recorder = MeterRecorder(class_labels, train_valid_metric_labels, logger=self.logger, saver=self.data_saver, prefix="valid_")
        self.loss_tracker = LossTracker(self.data_saver, self.filename_env, logger=self.logger)

        postprocess_name = c.get('postprocess', "")
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

        self.postprocess = select_postprocess_fn(postprocess_name)
        monitor_conf = c.get('monitor', {}) or {}
        self.monitor = None
        self.progress_tracker = None
        self.web_monitor = None
        self.monitor_api_client = MonitorApiClient.from_config(monitor_conf, logger=self.logger)
        self._monitor_run_info = self._build_monitor_run_info(c)
        self.log_interval = max(1, int(c['train'].get('log_interval', 10)))
        self._async_executor = async_executor
        progress_tracker_needed = (
            monitor_conf.get('enabled', False)
            or monitor_conf.get('web', False)
            or self.monitor_api_client is not None
        )

        if monitor_conf.get('enabled', False):
            try:
                self.monitor = TrainingMonitor(monitor_conf.get('config'))
                self.monitor.start_monitoring()
                self.progress_tracker = ProgressTracker()
                self.logger.info("Training monitor enabled")
            except Exception as e:
                self.logger.warning(f"Monitor initialization failed: {e}")
                self.monitor = None
                self.progress_tracker = None
        if monitor_conf.get('web'):
            try:
                self.web_monitor = create_web_monitor()
                if c.get("private", {}).get("wandb") and c.get("private", {}).get("wandb_alerts", True):
                    def _wandb_alert_callback(alert):
                        try:
                            import wandb
                            wandb.alert(title=alert.rule_name, text=alert.message)
                        except Exception:
                            pass
                    self.web_monitor.alert_system.alert_callbacks.append(_wandb_alert_callback)
                self.web_monitor.start(block=False)
                self.logger.info("Web monitor started")
            except Exception as e:
                self.logger.warning(f"Web monitor initialization failed: {e}")
                self.web_monitor = None

        if progress_tracker_needed and self.progress_tracker is None:
            self.progress_tracker = ProgressTracker()

        if self.monitor_api_client:
            self.monitor_api_client.update_run_info(self._monitor_run_info)
        if is_continue_mode:
            self._recovery()

    def _recovery(self):
        # MeterBasedRecorder uses epoch_meters format
        # Recovery mechanism for MeterBasedRecorder
        try:
            recovery_dir = self._recovery_dir if self._recovery_dir is not None else self.filename_env.recovery_dir
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

            self.logger.info("No recovery data found")
        except Exception as e:
            self.logger.warning(f"Recovery failed: {e}")

    def _build_monitor_run_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": config.get("task"),
            "entity": config.get("entity"),
            "run_id": config.get("run_id"),
            "output_dir": str(self.output_dir),
            "device": config.get("device"),
        }
    
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

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            if value.device == self.device:
                return value
            return value.to(self.device, non_blocking=True)
        if isinstance(value, NDict):
            return NDict({k: self._move_to_device(v) for k, v in value.items()})
        if isinstance(value, Mapping):
            return {k: self._move_to_device(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(v) for v in value)
        if isinstance(value, list):
            return [self._move_to_device(v) for v in value]
        if isinstance(value, set):
            return {self._move_to_device(v) for v in value}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._move_to_device(v) for v in value]
        return value

    def _forward_model(self, batch_inputs) -> tuple[ModelOutput, torch.Tensor]:
        inputs_obj = batch_inputs
        targets = None

        if isinstance(batch_inputs, tuple) and len(batch_inputs) == 2:
            inputs_obj, targets = batch_inputs

        model_input = inputs_obj
        if isinstance(inputs_obj, Mapping):
            # 规范化：从 dict 中取出模型输入张量（支持 inputs / image / images）
            if 'inputs' in inputs_obj:
                model_input = inputs_obj['inputs']
            elif 'image' in inputs_obj:
                model_input = inputs_obj['image']
            elif 'images' in inputs_obj:
                model_input = inputs_obj['images']
            else:
                model_input = inputs_obj
            targets = inputs_obj.get('targets', targets)
            if targets is None and 'metadata' in inputs_obj:
                meta = inputs_obj['metadata']
                dev = model_input.device if isinstance(model_input, torch.Tensor) else None
                if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], Mapping) and 'label' in meta[0]:
                    targets = torch.tensor([m['label'] for m in meta], device=dev)
                elif isinstance(meta, Mapping) and 'label' in meta:
                    lbl = meta['label']
                    if isinstance(lbl, torch.Tensor):
                        targets = lbl if dev is None else lbl.to(dev)
                        if targets.dtype != torch.long:
                            targets = targets.long()
                    else:
                        # 单样本为标量，batch 为 list；保证 targets 为 1D long
                        flat = [lbl] if (np.ndim(lbl) == 0 or isinstance(lbl, (int, float))) else lbl
                        targets = torch.tensor(flat, device=dev, dtype=torch.long)
        else:
            model_input = inputs_obj

        result = self.model(model_input)
        loss: torch.Tensor | None = None
        if isinstance(result, tuple) and len(result) == 2:
            outputs_raw, loss = result
        else:
            outputs_raw = result

        if isinstance(outputs_raw, ModelOutput):
            outputs = outputs_raw
        elif isinstance(outputs_raw, Mapping):
            outputs = ModelOutput(**outputs_raw)
        else:
            outputs = ModelOutput(preds=outputs_raw)

        if 'targets' not in outputs and targets is not None:
            outputs['targets'] = targets

        if self.criterion is not None:
            preds = outputs.get('preds')
            targets = outputs.get('targets')
            if isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor):
                loss = self.criterion(targets, preds)

        if loss is None:
            raise ValueError("Unable to compute loss: ensure model or criterion provides it.")

        assert isinstance(loss, torch.Tensor)
        return outputs, loss

    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            with torch.no_grad():
                return float(value.detach().mean().item())
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.mean())
        return None

    def _extract_loss_components(self, outputs: ModelOutput, loss_tensor: torch.Tensor | None):
        losses = {}
        total_loss_value = self._to_float(loss_tensor)
        if total_loss_value is None:
            return losses
        losses['total_loss'] = total_loss_value
        for key, value in outputs.items():
            if not isinstance(key, str):
                continue
            key_lower = key.lower()
            if key_lower == 'loss':
                continue
            if 'loss' not in key_lower:
                continue
            scalar_value = self._to_float(value)
            if scalar_value is None:
                continue
            losses[key] = scalar_value
        return losses

    def _record_loss_step(self, stage: str, epoch: int, step: int, total_steps: int | None, outputs: ModelOutput, loss_tensor: torch.Tensor | None):
        if not hasattr(self, 'loss_tracker') or self.loss_tracker is None:
            return
        losses = self._extract_loss_components(outputs, loss_tensor)
        if not losses:
            return
        global_step = step
        if total_steps is not None:
            global_step = step + epoch * total_steps
        self.loss_tracker.record_step(stage, epoch, step, global_step, losses)

    def _train_epoch(
        self,
        epoch: int,
        num_epochs: int,
        train_dataloader: DataLoader,
        pbar: ProgressBar,
        train_losses_ref: list[float],
        valid_losses_ref: list[float],
        total_pbar_steps: int,
    ):
        c = get_config()
        accumulation_steps = c["train"].get("grad_accumulation_steps", 1)
        enable_accumulation_step = accumulation_steps > 0
        lr_scheduler_update_policy = c['train'].get('lr_scheduler', {}).get('update_policy', 'epoch')
        train_loss = 0.0
        step_losses = []
        step_lrs = []
        self.model.train()
        if self.loss_tracker:
            self.loss_tracker.begin_epoch('train')

        total_steps = len(train_dataloader)
        processed_batches = 0

        if not enable_accumulation_step and self.optimizer is not None:
            self.optimizer.zero_grad()

        for batch, batch_inputs in enumerate(train_dataloader, 1):
            batch_inputs = self._move_to_device(batch_inputs)
            processed_batches += 1

            sum_loss = 0.0
            if self.scaler:
                device_type = 'cuda' if 'cuda' in self.device.type else 'cpu'
                compute_type = torch.bfloat16 if c['train']['scaler']['compute_type'] == 'bfloat16' else torch.float16
                with autocast(device_type, dtype=compute_type):
                    outputs, loss = self._forward_model(batch_inputs)

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
                outputs, loss = self._forward_model(batch_inputs)
                
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
            self._record_loss_step('train', epoch, batch, total_steps, outputs, loss)
            
            # 收集当前step的学习率
            if self.optimizer is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                step_lrs.append(current_lr)
            
            metric_targets = outputs['targets']
            metric_preds = outputs['preds']

            if (
                self.postprocess is not None
                and isinstance(metric_targets, torch.Tensor)
                and isinstance(metric_preds, torch.Tensor)
            ):
                metric_targets, metric_preds = self.postprocess(metric_targets, metric_preds)

            if (
                isinstance(metric_targets, torch.Tensor)
                and isinstance(metric_preds, torch.Tensor)
            ):
                self.train_metric_recorder.finish_one_batch(
                    metric_targets.detach().cpu().numpy(),
                    metric_preds.detach().cpu().numpy(),
                )

            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 0.0
            if self.monitor:
                run_async_task(
                    self._async_executor,
                    self.monitor.update_training_metrics,
                    epoch=epoch,
                    step=batch,
                    loss=batch_loss,
                    learning_rate=current_lr,
                    batch_size=c['train']['batch_size'],
                    logger=self.logger,
                )
            snapshot_for_api = None
            if self.progress_tracker:
                if batch == 1:
                    self.progress_tracker.start_epoch(epoch)
                if batch % self.log_interval == 0 or batch == total_steps:
                    self.progress_tracker.start_step(batch)
                    self.progress_tracker.end_step(c['train']['batch_size'])
                    snapshot_for_api = self.progress_tracker.get_current_progress()
                    if self.monitor_api_client and snapshot_for_api:
                        run_async_task(
                            self._async_executor,
                            self.monitor_api_client.send_progress_snapshot,
                            snapshot_for_api,
                        )
            step_time_meter = getattr(self, 'step_time_meter', None)
            throughput = 0.0
            if step_time_meter is not None and hasattr(step_time_meter, 'avg'):
                throughput = c['train']['batch_size'] / max(step_time_meter.avg, 1e-6)
            if self.web_monitor:
                try:
                    run_async_task(
                        self._async_executor,
                        self.web_monitor.update_metrics,
                        epoch=epoch,
                        step=batch,
                        loss=batch_loss,
                        learning_rate=current_lr,
                        batch_size=c['train']['batch_size'],
                        throughput=throughput,
                        logger=self.logger,
                    )
                except Exception as exc:
                    self.logger.warning(f"Web monitor update failed: {exc}")
                    self.web_monitor = None

            eta_total_seconds = None
            if self.progress_tracker:
                current_snapshot = snapshot_for_api or self.progress_tracker.get_current_progress()
                if current_snapshot and current_snapshot.eta_total:
                    eta_total_seconds = current_snapshot.eta_total.total_seconds()

            if self.monitor_api_client:
                run_async_task(
                    self._async_executor,
                    self.monitor_api_client.send_training_metrics,
                    epoch=epoch,
                    step=batch,
                    loss=batch_loss,
                    learning_rate=current_lr,
                    batch_size=c['train']['batch_size'],
                    throughput=throughput,
                    eta_total_seconds=eta_total_seconds,
                )

            pbar.update()

            main_metric = c.get('main_metric', 'loss')
            train_metrics = None
            if main_metric and main_metric != 'loss' and self.train_metric_recorder.metric_labels:
                all_m = self.train_metric_recorder.get_current_metrics()
                if main_metric in all_m:
                    train_metrics = {main_metric: all_m[main_metric]}
            pbar.update_step(
                epoch=epoch,
                current_step=batch,
                total_steps=total_steps,
                loss_label='batch_loss',
                loss_value=batch_loss,
                global_step=batch + epoch * total_steps,
                force=batch == total_steps,
                prefix='Training...',
                total_global_steps=total_pbar_steps,
                total_epochs=num_epochs,
                metrics=train_metrics,
            )

            if self._stop_manager.should_stop():
                stopped_losses = train_losses_ref.copy()
                if processed_batches > 0:
                    stopped_losses.append(train_loss / processed_batches)
                self._handle_stop_request(
                    epoch,
                    num_epochs,
                    stopped_losses,
                    valid_losses_ref.copy(),
                )
                raise TrainingStopRequested

        if processed_batches > 0:
            train_loss /= processed_batches
        else:
            train_loss = 0.0
        self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')
        # 保存step级loss
        try:
            self.data_saver.save_train_step_loss(np.array(step_losses, dtype=np.float64), filename=self.filename_env.output_train_step_loss_filename.as_posix())
        except Exception as e:
            self.logger.warning(f"Failed to save step losses: {e}")
        self.train_metric_recorder.finish_one_epoch()
        if self.loss_tracker:
            self.loss_tracker.finish_epoch('train', epoch)
        return train_loss, step_losses, step_lrs

    def _valid_epoch(
        self,
        epoch: int,
        num_epochs: int,
        valid_dataloader: DataLoader,
        pbar: ProgressBar,
        train_losses_ref: list[float],
        valid_losses_ref: list[float],
        total_pbar_steps: int,
    ):
        c = get_config()
        self.model.eval()
        valid_loss = 0.0
        processed_batches = 0
        total_steps = len(valid_dataloader)
        if self.loss_tracker:
            self.loss_tracker.begin_epoch('valid')

        with torch.no_grad():
            step_losses = []
            total_steps = len(valid_dataloader)
            for batch_idx, batch_inputs in enumerate(valid_dataloader, 1):
                batch_inputs = self._move_to_device(batch_inputs)
                processed_batches += 1

                outputs, loss = self._forward_model(batch_inputs)
        
                batch_loss = loss.item()
                valid_loss += batch_loss
                step_losses.append(batch_loss)
                self._record_loss_step('valid', epoch, batch_idx, total_steps, outputs, loss)

                metric_targets = outputs['targets']
                metric_preds = outputs['preds']

                if (
                    self.postprocess is not None
                    and isinstance(metric_targets, torch.Tensor)
                    and isinstance(metric_preds, torch.Tensor)
                ):
                    metric_targets, metric_preds = self.postprocess(metric_targets, metric_preds)

                if (
                    isinstance(metric_targets, torch.Tensor)
                    and isinstance(metric_preds, torch.Tensor)
                ):
                    self.valid_metric_recorder.finish_one_batch(
                        metric_targets.detach().cpu().numpy(),
                        metric_preds.detach().cpu().numpy(),
                    )

                pbar.update()

                main_metric = c.get("main_metric", "loss")
                valid_metrics = None
                if main_metric and main_metric != "loss" and self.valid_metric_recorder.metric_labels:
                    all_m = self.valid_metric_recorder.get_current_metrics()
                    if main_metric in all_m:
                        valid_metrics = {main_metric: all_m[main_metric]}
                pbar.update_step(
                    epoch=epoch,
                    current_step=batch_idx,
                    total_steps=total_steps,
                    loss_label='valid_batch_loss',
                    loss_value=batch_loss,
                    force=batch_idx == total_steps,
                    prefix='Training...',
                    total_global_steps=total_pbar_steps,
                    total_epochs=num_epochs,
                    metrics=valid_metrics,
                )

                if self._stop_manager.should_stop():
                    stopped_valid_losses = valid_losses_ref.copy()
                    if processed_batches > 0:
                        stopped_valid_losses.append(valid_loss / processed_batches)
                    self._handle_stop_request(
                        epoch,
                        num_epochs,
                        train_losses_ref.copy(),
                        stopped_valid_losses,
                    )
                    raise TrainingStopRequested

        if processed_batches > 0:
            valid_loss /= processed_batches
        else:
            valid_loss = 0.0
        self.logger.info(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss}')
        self.valid_metric_recorder.finish_one_epoch()
        if self.loss_tracker:
            self.loss_tracker.finish_epoch('valid', epoch)
        return valid_loss

    def train(self, num_epochs: int,
              train_dataloader,
              valid_dataloader = None,
              *, early_stop: bool = False, last_epoch: int=0):
        c = get_config()
        if last_epoch == 0 and self.is_continue_mode:
            try:
                recovery_dir = self._recovery_dir if self._recovery_dir is not None else self.filename_env.recovery_dir
                pointer_file = recovery_dir / "recovery_info.json"
                if pointer_file.exists():
                    with pointer_file.open("r") as f:
                        recovery_data = json.load(f)
                    last_epoch = int(recovery_data.get("epoch", 0))
                    self.logger.info(f"Resumed last_epoch from recovery_info: {last_epoch}")
            except Exception as e:
                self.logger.warning(f"Could not infer last_epoch from recovery_info: {e}")
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

        self.model = self.model.to(self.device)
        reset_peak_memory_stats()

        save_model_preset = partial(save_model, model=self.model, optimizer=self.optimizer, scaler=self.scaler, lr_scheduler=self.lr_scheduler, version=c['run_id'])
        self._save_model_fn = save_model_preset
        self._stop_manager.reset()
        self._stop_saved = False

        if early_stop and valid_dataloader is None:
            self.logger.warning("Validate isn't launched, early_stop will be cancelled")
            early_stopper = None
        elif early_stop and valid_dataloader:
            early_stopper = EarlyStopping(c['train']['early_stopping']['patience'])
        else:
            early_stopper = None

        best_by = c['train'].get('best_by', 'loss')
        metric_direction = c['train'].get('metric_direction', 'max')
        if best_by == 'main_metric':
            if c.get('main_metric') in (None, 'loss') or not self.valid_metric_recorder.metric_labels:
                self.logger.warning("best_by=main_metric but main_metric is loss or no valid metrics; using best_by=loss")
                best_by = 'loss'
        if best_by == 'main_metric':
            best_metric = float('-inf') if metric_direction == 'max' else float('inf')
        else:
            best_metric = None
        best_loss = float('inf')

        train_losses = []
        valid_losses = []
        
        # 收集step级别的数据用于可视化
        all_step_losses = []
        all_step_lrs = []
        epoch_lrs = []

        steps_per_epoch = len(train_dataloader) if train_dataloader is not None else 0
        if self.progress_tracker:
            self.progress_tracker.start_training(num_epochs - last_epoch, steps_per_epoch)
        if self.monitor_api_client:
            self.monitor_api_client.start_monitoring(self._monitor_run_info)
        if self.web_monitor:
            try:
                self.web_monitor.start_training(num_epochs - last_epoch, steps_per_epoch)
            except Exception as exc:
                self.logger.warning(f"Failed to start web monitor training session: {exc}")
                self.web_monitor = None
        total_pbar_steps = (num_epochs - last_epoch) * (len(train_dataloader) + (len(valid_dataloader) if valid_dataloader else 0))
        stop_triggered = False
        completed_epoch = last_epoch

        try:
            with ProgressBar(total=total_pbar_steps, desc='Training...', log_interval=self.log_interval) as pbar:
                for epoch in range(last_epoch+1, num_epochs+1):
                    train_loss, epoch_step_losses, epoch_step_lrs = self._train_epoch(
                        epoch,
                        num_epochs,
                        train_dataloader,
                        pbar,
                        train_losses,
                        valid_losses,
                        total_pbar_steps,
                    )
                    train_losses.append(train_loss)
                    completed_epoch = epoch
                    
                    # 收集step级别的数据
                    all_step_losses.extend(epoch_step_losses)
                    all_step_lrs.extend(epoch_step_lrs)
                    
                    # 收集epoch级别的学习率
                    if self.optimizer is not None:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        epoch_lrs.append(current_lr)
                    
                    pbar.update(0)
                    epoch_postfix: dict[str, Any] = {'epoch_loss': train_loss, 'epoch': epoch}
                    main_metric = c.get('main_metric', 'loss')
                    if main_metric and main_metric != 'loss':
                        train_epoch_metrics = self.train_metric_recorder.get_epoch_metrics()
                        if main_metric in train_epoch_metrics:
                            epoch_postfix[main_metric] = f"{float(train_epoch_metrics[main_metric]):.4f}"
                    pbar.set_postfix(epoch_postfix)

                    # 调整 lr per one epoch
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
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
                        valid_loss = self._valid_epoch(
                            epoch,
                            num_epochs,
                            valid_dataloader,
                            pbar,
                            train_losses,
                            valid_losses,
                            total_pbar_steps,
                        )
                        valid_losses.append(valid_loss)

                        pbar.update(0)
                        valid_postfix: dict[str, Any] = {'valid_epoch_loss': valid_loss, 'epoch': epoch}
                        main_metric = c.get('main_metric', 'loss')
                        if main_metric and main_metric != 'loss':
                            valid_epoch_metrics = self.valid_metric_recorder.get_epoch_metrics()
                            if main_metric in valid_epoch_metrics:
                                valid_postfix[main_metric] = f"{float(valid_epoch_metrics[main_metric]):.4f}"
                        pbar.set_postfix(valid_postfix)

                        # save best model and early_stop by best_by (loss or main_metric)
                        # EarlyStopping expects "quantity to minimize" and does score = -val internally
                        if best_by == 'loss':
                            target = valid_loss
                            if target < best_loss:
                                best_loss = target
                                save_model_preset(self.filename_env.output_best_model_filename,
                                            ext_path=self.filename_env.output_best_ext_model_filename,
                                            epoch=epoch, loss=best_loss)
                                self.logger.info(f'save best model (loss) to {self.filename_env.output_best_model_filename} when {epoch=}, {best_loss=}')
                            early_score = valid_loss
                        else:
                            valid_metrics = self.valid_metric_recorder.get_epoch_metrics()
                            main_metric_name = self.valid_metric_recorder.metric_labels[0] if self.valid_metric_recorder.metric_labels else str(c.get('main_metric', ''))
                            default_val = float('-inf') if metric_direction == 'max' else float('inf')
                            target = valid_metrics.get(main_metric_name, default_val)
                            current_best = best_metric if best_metric is not None else default_val
                            if metric_direction == 'max':
                                is_better = target > current_best
                                early_score = -target
                            else:
                                is_better = target < current_best
                                early_score = target
                            if is_better:
                                best_metric = target
                                save_model_preset(self.filename_env.output_best_model_filename,
                                            ext_path=self.filename_env.output_best_ext_model_filename,
                                            epoch=epoch, loss=valid_loss, best_metric=best_metric, best_by=best_by)
                                self.logger.info(f'save best model (main_metric={main_metric_name}) to {self.filename_env.output_best_model_filename} when {epoch=}, {best_metric=}')

                        if early_stopper:
                            early_stopper(early_score)
                            if early_stopper.is_stopped():
                                self.logger.info("Early stopping")
                                break

                    # per-epoch wandb full-recording
                    current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else None
                    train_metrics = self.train_metric_recorder.get_epoch_metrics()
                    valid_metrics = self.valid_metric_recorder.get_epoch_metrics() if valid_dataloader else None
                    self._wandb_log_epoch(
                        epoch, train_loss,
                        valid_loss if valid_dataloader else None,
                        current_lr, train_metrics, valid_metrics,
                    )

                    # else:
                    #     best_loss = train_loss
                    #     save_model(self.best_model_file_path, self.model, 
                    #                 ext_path=self.best_model_ext_file_path,
                    #                 optimizer=self.optimizer, scaler=self.scaler, lr_scheduler=self.lr_scheduler,
                    #                 epoch=epoch, version=c['run_id'], loss=best_loss)
                    #     self.logger.info(f'save model params to {self.best_model_file_path} and ext params to {self.best_model_ext_file_path} when {epoch=}, {best_loss=}')

                    save_model_preset(
                        self.filename_env.output_last_model_filename,
                        ext_path=self.filename_env.output_last_ext_model_filename,
                        epoch=num_epochs,
                        loss=train_loss,
                    )
                    if self.progress_tracker:
                        self.progress_tracker.end_epoch()
                    if self.web_monitor:
                        try:
                            self.web_monitor.end_epoch()
                        except Exception as exc:
                            self.logger.warning(f"Web monitor end_epoch failed: {exc}")

        except TrainingStopRequested:
            stop_triggered = True

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

        except KeyboardInterrupt:
            stop_triggered = True
            self._stop_manager.request_stop(reason="keyboard-interrupt")
            self.logger.warning("检测到键盘中断，正在安全终止训练")
        finally:
            if stop_triggered and not self._stop_saved:
                self._handle_stop_request(completed_epoch, num_epochs, train_losses, valid_losses if valid_dataloader else None)

        if self.progress_tracker and self.progress_tracker.is_training:
            self.progress_tracker.end_training()
        if self.monitor:
            self.monitor.stop_monitoring()
        if self.web_monitor:
            try:
                self.web_monitor.end_training()
                self.web_monitor.stop()
            except Exception as exc:
                self.logger.warning(f"Failed to stop web monitor cleanly: {exc}")
        if self.monitor_api_client:
            self.monitor_api_client.stop_monitoring()

        if not c.get("ddp", {}).get("enabled", False) or is_main_process():
            self._print_table(valid_dataloader is not None)

        self._draw_visualizations(all_step_losses, all_step_lrs, epoch_lrs, num_epochs)

        self._save_train_info(completed_epoch, num_epochs, train_losses, valid_losses if valid_dataloader else None)
        self._wandb_save(train_losses, valid_losses, self.optimizer, self.scaler, self.lr_scheduler)
        log_memory_cost("Train", self.logger)

    def _handle_stop_request(self, epoch: int, num_epochs: int, train_losses, valid_losses=None):
        """在收到停止信号时，统一保存checkpoint、扩展状态以及恢复信息。"""
        if self._stop_saved:
            return

        self._stop_saved = True
        reason = self._stop_manager.reason or "manual"
        safe_epoch = max(epoch, 0)
        self.logger.warning(f"检测到训练终止请求（原因：{reason}），将在 epoch={safe_epoch} 保存临时检查点")

        last_loss = train_losses[-1] if train_losses else None
        try:
            if self._save_model_fn is not None:
                stop_model_path = self.filename_env.output_stop_model_filename
                stop_ext_path = self.filename_env.output_stop_ext_model_filename
                self._save_model_fn(
                    stop_model_path,
                    ext_path=stop_ext_path,
                    epoch=safe_epoch,
                    loss=last_loss,
                )
                self.logger.info(f"Stop checkpoint saved to {stop_model_path} / {stop_ext_path}")
        except Exception as exc:
            self.logger.error(f"Failed to save stop checkpoint: {exc}")

        try:
            self._save_recovery_info(safe_epoch, num_epochs, train_losses, valid_losses)
        except Exception as exc:
            self.logger.warning(f"Failed to save recovery info on stop: {exc}")

        try:
            self._save_train_info(safe_epoch, num_epochs, train_losses, valid_losses)
        except Exception as exc:
            self.logger.warning(f"Failed to save metric info on stop: {exc}")

        c = get_config()
        if c.get("private", {}).get("wandb") and c.get("private", {}).get("wandb_alerts", True):
            try:
                import wandb
                wandb.alert(title="Run stopped", text=str(reason))
            except Exception as exc:
                self.logger.debug(f"wandb alert on stop failed: {exc}")

        try:
            self._stop_manager.acknowledge()
        except Exception as exc:
            self.logger.debug(f"Failed to cleanup stop flag: {exc}")

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
        class_labels = c['classes']
        metric_labels = self.train_metric_recorder.metric_labels
        if not metric_labels:
            self.logger.info("Train/Valid: main_metric is loss, no per-batch metrics table.")
            return

        train_mean_scores = self.train_metric_recorder.get_epoch_metrics()
        train_mc1_mean = self.train_metric_recorder._compute_mc1(np.mean)
        train_mc1_std = self.train_metric_recorder._compute_mc1(np.std)
        train_std_scores = {}
        for m in metric_labels:
            vals = list(train_mc1_mean.get(m, {}).values())
            train_std_scores[m] = float(np.std(vals)) if len(vals) > 1 else 0.0
        class_table_rows: list = [("Train", train_mc1_mean, train_mc1_std)]
        summary_rows: list = [("Train", train_mean_scores, train_std_scores)]

        if enable_valid_when_training:
            valid_mean_scores = self.valid_metric_recorder.get_epoch_metrics()
            valid_mc1_mean = self.valid_metric_recorder._compute_mc1(np.mean)
            valid_mc1_std = self.valid_metric_recorder._compute_mc1(np.std)
            valid_std_scores = {}
            for m in metric_labels:
                vals = list(valid_mc1_mean.get(m, {}).values())
                valid_std_scores[m] = float(np.std(vals)) if len(vals) > 1 else 0.0
            class_table_rows.append(("Valid", valid_mc1_mean, valid_mc1_std))
            summary_rows.append(("Valid", valid_mean_scores, valid_std_scores))

        print_metric_scores_table(
            class_labels,
            metric_labels,
            class_table_rows,
            summary_rows,
            style_key="train",
            title_class="Metric Class Mean Score(Train)",
            title_summary="Summary of Metric(Train)",
        )

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

    def _wandb_log_epoch(self, epoch: int, train_loss: float, valid_loss=None, lr=None,
                         train_metrics=None, valid_metrics=None):
        """Log per-epoch metrics to wandb for full-recording."""
        c = get_config()
        if not c.get('private', {}).get('wandb'):
            return
        try:
            import wandb
            log_dict = {"epoch": epoch, "train/loss": train_loss}
            if lr is not None:
                log_dict["lr"] = lr
            if valid_loss is not None:
                log_dict["valid/loss"] = valid_loss
            if train_metrics:
                for k, v in train_metrics.items():
                    log_dict[f"train/{k}"] = v
            if valid_metrics:
                for k, v in valid_metrics.items():
                    log_dict[f"valid/{k}"] = v
            wandb.log(log_dict, step=epoch)
        except Exception as e:
            self.logger.warning(f"wandb log epoch failed: {e}")

    def _wandb_save(self, train_losses, valid_losses=None, optimizer=None, scaler=None, lr_scheduler=None):
        c = get_config()
        if c.get('private', {}).get('wandb'):
            train_c = c.get('train', {})
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
                        **{k: v for k, v in train_c.get('lr_scheduler', {}).items()},
                    } if lr_scheduler else None,
                },
            })
