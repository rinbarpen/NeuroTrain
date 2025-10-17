"""
DeepSpeed 训练器

提供基于 DeepSpeed 的分布式训练支持，包括：
- ZeRO 优化器状态分片
- 梯度累积和同步
- 混合精度训练
- 模型并行和数据并行
- 内存优化和检查点管理
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from rich.table import Table
from rich.console import Console

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

from src.config import get_config
from src.utils import EarlyStopping, select_postprocess_fn
from src.recorder import MeterRecorder, DataSaver
from src.visualizer.painter import Plot
from src.constants import TrainOutputFilenameEnv


class DeepSpeedTrainer:
    """DeepSpeed 训练器类"""
    
    def __init__(
        self,
        output_dir: Path,
        model: nn.Module,
        deepspeed_config: Union[str, Dict[str, Any]],
        is_continue_mode: bool = False,
        local_rank: int = -1
    ):
        """
        初始化 DeepSpeed 训练器
        
        Args:
            output_dir: 输出目录
            model: 要训练的模型
            deepspeed_config: DeepSpeed 配置文件路径或配置字典
            is_continue_mode: 是否为继续训练模式
            local_rank: 本地进程排名
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available. Please install deepspeed: pip install deepspeed")
        
        self.output_dir = output_dir
        self.model = model
        self.local_rank = local_rank
        self.is_continue_mode = is_continue_mode
        
        # 获取配置
        self.c = get_config()
        
        # 设置输出环境
        self.filename_env = TrainOutputFilenameEnv()
        self.filename_env.register(train_dir=self.output_dir, model=self.c["model"]["name"])
        self.filename_env.register(epoch=0, num_epochs=0)
        self.filename_env.prepare_dir()
        
        # 初始化日志记录器
        self.logger = logging.getLogger('deepspeed_train')
        self.data_saver = DataSaver()
        
        # 设置指标记录器
        class_labels = self.c['classes']
        metric_labels = self.c['metrics']
        self.train_metric_recorder = MeterRecorder(
            class_labels, metric_labels, 
            logger=self.logger, saver=self.data_saver, prefix="train_"
        )
        self.valid_metric_recorder = MeterRecorder(
            class_labels, metric_labels, 
            logger=self.logger, saver=self.data_saver, prefix="valid_"
        )
        
        # 设置后处理函数
        postprocess_name = self.c.get('postprocess', "")
        if postprocess_name:
            self.postprocess = select_postprocess_fn(postprocess_name)
        else:
            self.postprocess = None
        
        # 初始化 DeepSpeed
        self._init_deepspeed(deepspeed_config)
        
        # 恢复模式
        if is_continue_mode:
            self._recovery()
    
    def _init_deepspeed(self, deepspeed_config: Union[str, Dict[str, Any]]):
        """初始化 DeepSpeed 引擎"""
        # 加载 DeepSpeed 配置
        if isinstance(deepspeed_config, str):
            config_path = Path(deepspeed_config)
            if not config_path.exists():
                raise FileNotFoundError(f"DeepSpeed config file not found: {config_path}")
            with open(config_path, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = deepspeed_config
        
        # 更新配置中的自动值
        ds_config = self._update_ds_config(ds_config)
        
        # 初始化 DeepSpeed 引擎
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            local_rank=self.local_rank
        )
        
        self.logger.info(f"DeepSpeed initialized with config: {ds_config}")
        self.logger.info(f"Model engine: {self.model_engine}")
        self.logger.info(f"Optimizer: {self.optimizer}")
        self.logger.info(f"LR Scheduler: {self.lr_scheduler}")
    
    def _update_ds_config(self, ds_config: Dict[str, Any]) -> Dict[str, Any]:
        """更新 DeepSpeed 配置中的自动值"""
        train_config = self.c['train']
        
        # 更新批次大小
        if ds_config.get('train_batch_size') == 'auto':
            ds_config['train_batch_size'] = train_config['batch_size']
        
        if ds_config.get('train_micro_batch_size_per_gpu') == 'auto':
            ds_config['train_micro_batch_size_per_gpu'] = train_config['batch_size']
        
        if ds_config.get('gradient_accumulation_steps') == 'auto':
            ds_config['gradient_accumulation_steps'] = train_config.get('grad_accumulation_steps', 1)
        
        # 更新优化器配置
        if 'optimizer' in ds_config and ds_config['optimizer'].get('params'):
            opt_params = ds_config['optimizer']['params']
            if opt_params.get('lr') == 'auto':
                opt_params['lr'] = train_config['optimizer']['learning_rate']
            if opt_params.get('weight_decay') == 'auto':
                opt_params['weight_decay'] = train_config['optimizer']['weight_decay']
        
        # 更新学习率调度器配置
        if 'scheduler' in ds_config and ds_config['scheduler'].get('params'):
            sched_params = ds_config['scheduler']['params']
            if sched_params.get('warmup_min_lr') == 'auto':
                sched_params['warmup_min_lr'] = train_config.get('lr_scheduler', {}).get('warmup_start_lr', 0.0)
            if sched_params.get('warmup_max_lr') == 'auto':
                sched_params['warmup_max_lr'] = train_config['optimizer']['learning_rate']
            if sched_params.get('warmup_num_steps') == 'auto':
                warmup_epochs = train_config.get('lr_scheduler', {}).get('warmup_epochs', 0)
                sched_params['warmup_num_steps'] = warmup_epochs * train_config['batch_size']
        
        # 更新梯度裁剪
        if ds_config.get('gradient_clipping') == 'auto':
            ds_config['gradient_clipping'] = train_config.get('max_grad_norm', 1.0)
        
        return ds_config
    
    def _recovery(self):
        """恢复训练状态"""
        try:
            # 查找最新的检查点
            checkpoint_dirs = list(self.output_dir.glob("checkpoint-*"))
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split('-')[1]))
                self.logger.info(f"Loading checkpoint from {latest_checkpoint}")
                
                # 加载 DeepSpeed 检查点
                _, client_state = self.model_engine.load_checkpoint(latest_checkpoint)
                
                if client_state:
                    self.logger.info(f"Loaded client state: {client_state}")
                    return client_state.get('epoch', 0)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
        
        return 0
    
    def setup_trainer(self, criterion=None):
        """设置训练器"""
        self.criterion = criterion
        self.logger.info("DeepSpeed trainer setup completed")
    
    def train(
        self,
        num_epochs: int,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
        early_stop: bool = False,
        last_epoch: int = 0
    ):
        """
        执行训练
        
        Args:
            num_epochs: 训练轮数
            train_dataloader: 训练数据加载器
            valid_dataloader: 验证数据加载器
            early_stop: 是否启用早停
            last_epoch: 起始轮数
        """
        self.logger.info(f"Starting DeepSpeed training for {num_epochs} epochs")
        
        # 设置早停
        early_stopping = None
        if early_stop and valid_dataloader is not None:
            patience = self.c['train'].get('early_stopping', {}).get('patience', 10)
            early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # 训练循环
        for epoch in range(last_epoch, num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练一个epoch
            train_metrics = self._train_epoch(epoch, train_dataloader)
            
            # 验证
            if valid_dataloader is not None:
                valid_metrics = self._validate_epoch(epoch, valid_dataloader)
                
                # 早停检查
                if early_stopping:
                    if early_stopping(valid_metrics.get('loss', float('inf'))):
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # 保存检查点
            if (epoch + 1) % self.c['train'].get('save_period', 10) == 0:
                self._save_checkpoint(epoch + 1, train_metrics)
            
            # 学习率调度
            if self.lr_scheduler:
                self.lr_scheduler.step()
        
        # 保存最终模型
        self._save_final_model()
        
        self.logger.info("DeepSpeed training completed")
    
    def _train_epoch(self, epoch: int, train_dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model_engine.train()
        
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        # 创建进度条
        if self.local_rank <= 0:  # 只在主进程显示进度条
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataloader
        
        for batch_idx, batch_data in enumerate(pbar):
            # 准备数据
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                inputs, targets = batch_data
            else:
                inputs = batch_data
                targets = None
            
            # 前向传播
            if self.criterion and targets is not None:
                outputs = self.model_engine(inputs)
                loss = self.criterion(targets, outputs)
            else:
                outputs, loss = self.model_engine(inputs)
            
            # 反向传播
            self.model_engine.backward(loss)
            self.model_engine.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            if self.local_rank <= 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })
            
            # 记录指标
            if targets is not None and self.postprocess:
                targets, outputs = self.postprocess(targets, outputs)
                self.train_metric_recorder.finish_one_batch(
                    targets.detach().cpu().numpy(),
                    outputs.detach().cpu().numpy()
                )
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss}
    
    def _validate_epoch(self, epoch: int, valid_dataloader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model_engine.eval()
        
        total_loss = 0.0
        num_batches = len(valid_dataloader)
        
        with torch.no_grad():
            for batch_data in valid_dataloader:
                # 准备数据
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    inputs, targets = batch_data
                else:
                    inputs = batch_data
                    targets = None
                
                # 前向传播
                if self.criterion and targets is not None:
                    outputs = self.model_engine(inputs)
                    loss = self.criterion(targets, outputs)
                else:
                    outputs, loss = self.model_engine(inputs)
                
                total_loss += loss.item()
                
                # 记录指标
                if targets is not None and self.postprocess:
                    targets, outputs = self.postprocess(targets, outputs)
                    self.valid_metric_recorder.finish_one_batch(
                        targets.detach().cpu().numpy(),
                        outputs.detach().cpu().numpy()
                    )
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch + 1} - Valid Loss: {avg_loss:.4f}")
        
        return {'loss': avg_loss}
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_dir = self.output_dir / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存 DeepSpeed 检查点
        client_state = {
            'epoch': epoch,
            'metrics': metrics
        }
        
        self.model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_final_model(self):
        """保存最终模型"""
        # 保存 DeepSpeed 模型
        model_dir = self.output_dir / "final_model"
        model_dir.mkdir(exist_ok=True)
        
        self.model_engine.save_checkpoint(model_dir)
        self.logger.info(f"Final model saved to {model_dir}")
    
    def get_model_engine(self):
        """获取 DeepSpeed 模型引擎"""
        return self.model_engine
    
    def get_optimizer(self):
        """获取优化器"""
        return self.optimizer
    
    def get_lr_scheduler(self):
        """获取学习率调度器"""
        return self.lr_scheduler
