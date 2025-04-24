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

from config import get_config, ALL_METRIC_LABELS
from utils.early_stopping import EarlyStopping
from utils.data_saver import DataSaver
from utils.util import get_logger, save_model
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
        self.logger = get_logger('train')
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

        pbar_total = tqdm(total=num_epochs-last_epoch, desc='Training(Epoch)...')
        for epoch in range(last_epoch+1, num_epochs+1):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(total=len(train_dataloader), desc='Training(Batch)...')
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
                    targets.detach().cpu().float().numpy(), 
                    outputs.detach().cpu().float().numpy())

            if lr_scheduler:
                lr_scheduler.step()

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            pbar_total.update()
            pbar_total.set_postfix({'epoch_loss': train_loss})

            self.logger.info(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}')
            self.train_calculator.finish_one_epoch()

            # validate
            if valid_dataloader:
                valid_loss = 0.0
                self.model.eval()

                with torch.no_grad():
                    pbar = tqdm(total=len(valid_dataloader), desc='Validating(Batch)...')
                    for inputs, targets in valid_dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = self.model(inputs)
                        loss = criterion(targets, outputs)

                        batch_loss = loss.item()
                        valid_loss += batch_loss
                        pbar.update()
                        pbar.set_postfix({'batch_loss': batch_loss})

                        targets, outputs = self.postprocess(targets, outputs)
                        self.valid_calculator.add_one_batch(
                            targets.detach().cpu().float().numpy(), 
                            outputs.detach().cpu().float().numpy())

                valid_loss /= len(valid_dataloader)
                valid_losses.append(valid_loss)

                pbar_total.set_postfix({'valid_epoch_loss': valid_loss})

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
        self.logger = get_logger('test')
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
        self.data_saver = DataSaver(output_dir) 
        
        c = get_config()
        class_labels = c['classes']
        metric_labels = c['metrics']
        self.calculator = ScoreCalculator(class_labels, metric_labels, logger=self.logger, saver=self.data_saver)

    @torch.no_grad()
    def valid(self, valid_dataloader: DataLoader):
        self.calculator.clear()
        for inputs, targets in valid_dataloader:
            outputs = self.model(inputs)

            self.postprocess(targets, outputs)
            self.calculator.add_one_batch(
                targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()
            )

        self.calculator.record_batches(self.output_dir)

    @classmethod
    def postprocess(self, targets: torch.Tensor, outputs: torch.Tensor):
        targets[targets >= 0.5] = 1
        targets[targets < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        return targets, outputs

# TODO: Easy to inherit to be used
class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('predict')
        self.timer = Timer()

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])

        self.model = self.model.to(device)
        self.model.eval()

        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name

            self.timer.start(input_filename + '.preprocess')
            input_tensor, original_size = self.preprocess(input)
            self.timer.stop(input_filename + '.preprocess')

            self.timer.start(input_filename + '.inference')
            input_tensor = input_tensor.to(device)
            pred_tensor = self.model(input_tensor)
            self.timer.stop(input_filename + '.inference')

            self.timer.start(input_filename + '.postprocess')
            pred_np = self.postprocess(pred_tensor)
            self.timer.stop(input_filename + '.postprocess')

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
