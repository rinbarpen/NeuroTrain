"""
LLM/VLM 训练管线
支持预训练、SFT、DPO、PPO、GRPO等多阶段训练
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from src.engine.llm.config import (
    ModelConfig,
    DatasetConfig,
    StageConfig,
    TrainingPlan,
    TrainingStageType,
)
from src.engine.llm.utils import (
    load_model_and_tokenizer,
    save_model_and_tokenizer,
    resolve_callable,
    get_reward_function,
)

logger = logging.getLogger(__name__)


class LLMTrainer:
    """
    LLM/VLM 训练器
    支持预训练、SFT、DPO、PPO、GRPO等多阶段训练流程
    
    使用示例:
        # 方式1: 使用训练计划执行多阶段训练
        >>> from src.engine.llm import LLMTrainer, load_training_plan_from_dict
        >>> config_dict = {...}  # 训练配置
        >>> plan = load_training_plan_from_dict(config_dict)
        >>> trainer = LLMTrainer(plan)
        >>> trainer.train()
        
        # 方式2: 直接调用单个训练方法
        >>> trainer = LLMTrainer(plan)
        >>> trainer.sft()  # 执行SFT训练
        >>> trainer.dpo()  # 执行DPO训练
    """
    
    def __init__(self, plan: TrainingPlan):
        """
        初始化训练器
        
        Args:
            plan: 训练计划配置
        """
        self.plan = plan
        self.logger = logger
        
        # 设置随机种子
        if plan.seed is not None:
            torch.manual_seed(plan.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(plan.seed)
        
        # 创建基础输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(plan.base_output_dir) / plan.task_name / timestamp
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"LLM Trainer initialized for task: {plan.task_name}")
        self.logger.info(f"Output directory: {self.base_output_dir}")
        
        # 阶段间传递的模型路径
        self._current_model_path: Optional[str] = None
    
    def train(self):
        """执行完整训练计划"""
        self.logger.info(f"Starting training plan with {len(self.plan.stages)} stages")
        
        for stage_idx, stage_config in enumerate(self.plan.stages, 1):
            self.logger.info(f"=" * 80)
            self.logger.info(f"Stage {stage_idx}/{len(self.plan.stages)}: {stage_config.stage_name} ({stage_config.stage_type.value})")
            self.logger.info(f"=" * 80)
            
            try:
                self._run_stage(stage_config)
            except Exception as e:
                self.logger.error(f"Stage {stage_config.stage_name} failed: {e}", exc_info=True)
                raise
        
        self.logger.info("Training plan completed successfully!")
        self.logger.info(f"All outputs saved to: {self.base_output_dir}")
    
    # ==================== 公开的训练方法 ====================
    
    def sft(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        stage_config: Optional[StageConfig] = None,
        **kwargs
    ):
        """
        执行 SFT (Supervised Fine-Tuning) 训练
        
        Args:
            dataset_config: 数据集配置，如果为None则从plan中获取
            stage_config: 阶段配置，如果为None则创建默认配置
            **kwargs: 传递给StageConfig的其他参数
        """
        if stage_config is None:
            stage_config = StageConfig(
                stage_type=TrainingStageType.SFT,
                stage_name="sft",
                **kwargs
            )
        if dataset_config is None:
            dataset_config = self.plan.dataset_configs.get("sft") or DatasetConfig()
            self.plan.dataset_configs["sft"] = dataset_config
        
        self._run_sft(stage_config)
    
    def dpo(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        stage_config: Optional[StageConfig] = None,
        **kwargs
    ):
        """
        执行 DPO (Direct Preference Optimization) 训练
        
        Args:
            dataset_config: 数据集配置，如果为None则从plan中获取
            stage_config: 阶段配置，如果为None则创建默认配置
            **kwargs: 传递给StageConfig的其他参数
        """
        if stage_config is None:
            stage_config = StageConfig(
                stage_type=TrainingStageType.DPO,
                stage_name="dpo",
                **kwargs
            )
        if dataset_config is None:
            dataset_config = self.plan.dataset_configs.get("dpo") or DatasetConfig()
            self.plan.dataset_configs["dpo"] = dataset_config
        
        self._run_dpo(stage_config)
    
    def pretrain(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        stage_config: Optional[StageConfig] = None,
        **kwargs
    ):
        """
        执行预训练
        
        Args:
            dataset_config: 数据集配置，如果为None则从plan中获取
            stage_config: 阶段配置，如果为None则创建默认配置
            **kwargs: 传递给StageConfig的其他参数
        """
        if stage_config is None:
            stage_config = StageConfig(
                stage_type=TrainingStageType.PRETRAIN,
                stage_name="pretrain",
                **kwargs
            )
        if dataset_config is None:
            dataset_config = self.plan.dataset_configs.get("pretrain") or DatasetConfig()
            self.plan.dataset_configs["pretrain"] = dataset_config
        
        self._run_pretrain(stage_config)
    
    def ppo(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        stage_config: Optional[StageConfig] = None,
        **kwargs
    ):
        """
        执行 PPO (Proximal Policy Optimization) 训练
        
        Args:
            dataset_config: 数据集配置，如果为None则从plan中获取
            stage_config: 阶段配置，如果为None则创建默认配置
            **kwargs: 传递给StageConfig的其他参数
        """
        if stage_config is None:
            stage_config = StageConfig(
                stage_type=TrainingStageType.PPO,
                stage_name="ppo",
                **kwargs
            )
        if dataset_config is None:
            dataset_config = self.plan.dataset_configs.get("ppo") or DatasetConfig()
            self.plan.dataset_configs["ppo"] = dataset_config
        
        self._run_ppo(stage_config)
    
    def grpo(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        stage_config: Optional[StageConfig] = None,
        **kwargs
    ):
        """
        执行 GRPO (Group Relative Policy Optimization) 训练
        
        Args:
            dataset_config: 数据集配置，如果为None则从plan中获取
            stage_config: 阶段配置，如果为None则创建默认配置
            **kwargs: 传递给StageConfig的其他参数
        """
        if stage_config is None:
            stage_config = StageConfig(
                stage_type=TrainingStageType.GRPO,
                stage_name="grpo",
                **kwargs
            )
        if dataset_config is None:
            dataset_config = self.plan.dataset_configs.get("grpo") or DatasetConfig()
            self.plan.dataset_configs["grpo"] = dataset_config
        
        self._run_grpo(stage_config)
    
    # ==================== 内部方法 ====================
    
    def _run_stage(self, stage_config: StageConfig):
        """执行单个训练阶段"""
        stage_type = stage_config.stage_type
        
        if stage_type == TrainingStageType.PRETRAIN:
            self._run_pretrain(stage_config)
        elif stage_type == TrainingStageType.SFT:
            self._run_sft(stage_config)
        elif stage_type == TrainingStageType.DPO:
            self._run_dpo(stage_config)
        elif stage_type == TrainingStageType.PPO:
            self._run_ppo(stage_config)
        elif stage_type == TrainingStageType.GRPO:
            self._run_grpo(stage_config)
        else:
            raise ValueError(f"Unsupported stage type: {stage_type}")
    
    def _determine_model_path(self, stage_config: StageConfig) -> str:
        """确定当前阶段加载的模型路径"""
        if stage_config.load_from:
            return stage_config.load_from
        if self._current_model_path:
            return self._current_model_path
        return self.plan.model_config.model_name_or_path
    
    def _get_output_dir(self, stage_config: StageConfig) -> Path:
        """获取阶段输出目录"""
        if stage_config.output_dir:
            return Path(stage_config.output_dir)
        return self.base_output_dir / stage_config.stage_name
    
    def _load_dataset(self, stage_config: StageConfig) -> Dataset:
        """加载数据集"""
        # 获取数据集配置
        dataset_config = self.plan.dataset_configs.get(stage_config.stage_name)
        if dataset_config is None:
            raise ValueError(f"No dataset config found for stage: {stage_config.stage_name}")
        
        self.logger.info(f"Loading dataset for stage: {stage_config.stage_name}")
        
        # 从 HuggingFace 加载
        if dataset_config.dataset_name:
            dataset = load_dataset(
                dataset_config.dataset_name,
                split=dataset_config.dataset_split,
                streaming=dataset_config.streaming,
                **dataset_config.dataset_kwargs,
            )
        # 从本地路径加载
        elif dataset_config.dataset_path:
            dataset_path = Path(dataset_config.dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
            # 支持多种格式
            if dataset_path.is_dir():
                dataset = load_dataset(
                    "json",
                    data_dir=str(dataset_path),
                    split=dataset_config.dataset_split,
                    **dataset_config.dataset_kwargs,
                )
            else:
                suffix = dataset_path.suffix.lower()
                if suffix == ".json" or suffix == ".jsonl":
                    dataset = load_dataset(
                        "json",
                        data_files=str(dataset_path),
                        split=dataset_config.dataset_split,
                        **dataset_config.dataset_kwargs,
                    )
                elif suffix == ".csv":
                    dataset = load_dataset(
                        "csv",
                        data_files=str(dataset_path),
                        split=dataset_config.dataset_split,
                        **dataset_config.dataset_kwargs,
                    )
                elif suffix == ".txt":
                    # 支持自定义分隔符的txt文件
                    if dataset_config.txt_delimiter and dataset_config.txt_field_names:
                        dataset = self._load_txt_with_delimiter(
                            dataset_path, dataset_config
                        )
                    else:
                        # 默认使用tab分隔符，尝试作为CSV加载
                        dataset = load_dataset(
                            "csv",
                            data_files=str(dataset_path),
                            split=dataset_config.dataset_split,
                            delimiter="\t",
                            **dataset_config.dataset_kwargs,
                        )
                else:
                    raise ValueError(f"Unsupported dataset format: {suffix}")
        else:
            raise ValueError("Either dataset_name or dataset_path must be specified")
        
        # 限制样本数
        if dataset_config.num_samples:
            dataset = dataset.select(range(min(dataset_config.num_samples, len(dataset))))
        
        self.logger.info(f"Dataset loaded: {len(dataset)} samples")
        return dataset
    
    def _load_txt_with_delimiter(
        self, dataset_path: Path, dataset_config
    ) -> Dataset:
        """
        加载使用自定义分隔符的txt文件
        
        Args:
            dataset_path: 数据集文件路径
            dataset_config: 数据集配置
            
        Returns:
            Dataset对象
        """
        delimiter = dataset_config.txt_delimiter
        field_names = dataset_config.txt_field_names
        
        self.logger.info(
            f"Loading txt file with delimiter '{delimiter}' and fields: {field_names}"
        )
        
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                parts = line.split(delimiter)
                if len(parts) != len(field_names):
                    self.logger.warning(
                        f"Line {line_idx}: Expected {len(field_names)} fields, "
                        f"got {len(parts)}. Skipping."
                    )
                    continue
                
                example = {field: part.strip() for field, part in zip(field_names, parts)}
                examples.append(example)
        
        self.logger.info(f"Loaded {len(examples)} examples from txt file")
        
        # 转换为HuggingFace Dataset
        dataset = Dataset.from_list(examples)
        return dataset
    
    def _build_training_arguments(
        self,
        stage_config: StageConfig,
        output_dir: Path,
    ) -> TrainingArguments:
        """构建 TrainingArguments"""
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=stage_config.num_train_epochs,
            per_device_train_batch_size=stage_config.per_device_train_batch_size,
            per_device_eval_batch_size=stage_config.per_device_eval_batch_size,
            gradient_accumulation_steps=stage_config.gradient_accumulation_steps,
            learning_rate=stage_config.learning_rate,
            warmup_steps=stage_config.warmup_steps,
            logging_steps=stage_config.logging_steps,
            save_steps=stage_config.save_steps,
            eval_steps=stage_config.eval_steps,
            max_steps=stage_config.max_steps,
            optim=stage_config.optimizer,
            weight_decay=stage_config.weight_decay,
            max_grad_norm=stage_config.max_grad_norm,
            lr_scheduler_type=stage_config.lr_scheduler_type,
            fp16=stage_config.fp16,
            bf16=stage_config.bf16,
            do_eval=stage_config.do_eval,
            evaluation_strategy=stage_config.eval_strategy,
            save_strategy=stage_config.save_strategy,
            save_total_limit=stage_config.save_total_limit,
            local_rank=stage_config.local_rank,
            ddp_find_unused_parameters=stage_config.ddp_find_unused_parameters,
            deepspeed=stage_config.deepspeed,
            report_to=["wandb"] if self.plan.use_wandb else ["none"],
            run_name=f"{self.plan.task_name}_{stage_config.stage_name}",
            logging_dir=self.plan.logging_dir,
            seed=self.plan.seed,
            remove_unused_columns=False,
        )
        
        return args
    
    def _run_pretrain(self, stage_config: StageConfig):
        """运行预训练阶段"""
        self.logger.info("Starting pretraining stage")
        
        # 加载模型
        model_path = self._determine_model_path(stage_config)
        model, tokenizer = load_model_and_tokenizer(self.plan.model_config, model_path)
        
        # 加载数据集
        dataset = self._load_dataset(stage_config)
        
        # 数据预处理
        dataset_config = self.plan.dataset_configs[stage_config.stage_name]
        
        def tokenize_function(examples):
            return tokenizer(
                examples[dataset_config.text_field],
                truncation=True,
                max_length=dataset_config.max_length,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=dataset_config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 训练参数
        output_dir = self._get_output_dir(stage_config)
        training_args = self._build_training_arguments(stage_config, output_dir)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 训练
        trainer.train(resume_from_checkpoint=self.plan.resume_from_checkpoint)
        
        # 保存
        save_model_and_tokenizer(model, tokenizer, output_dir)
        self._current_model_path = str(output_dir)
        
        self.logger.info(f"Pretraining completed. Model saved to {output_dir}")
    
    def _run_sft(self, stage_config: StageConfig):
        """运行 SFT(Supervised Fine-Tuning) 阶段"""
        self.logger.info("Starting SFT stage")
        
        try:
            from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        except ImportError:
            raise ImportError("TRL is required for SFT training. Install: pip install trl")
        
        # 加载模型
        model_path = self._determine_model_path(stage_config)
        model, tokenizer = load_model_and_tokenizer(self.plan.model_config, model_path)
        
        # 加载数据集
        dataset = self._load_dataset(stage_config)
        dataset_config = self.plan.dataset_configs[stage_config.stage_name]
        
        # 格式化函数
        formatting_func = None
        if dataset_config.formatting_func:
            formatting_func = resolve_callable(dataset_config.formatting_func)
        else:
            # 默认格式化
            def default_formatting_func(example):
                prompt = example.get(dataset_config.prompt_field, "")
                response = example.get(dataset_config.response_field, "")
                return f"{prompt}\n{response}"
            
            formatting_func = default_formatting_func
        
        # 训练参数
        output_dir = self._get_output_dir(stage_config)
        training_args = self._build_training_arguments(stage_config, output_dir)
        
        # SFT Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            formatting_func=formatting_func,
            max_seq_length=dataset_config.max_length,
            packing=stage_config.stage_kwargs.get("packing", False),
        )
        
        # 训练
        trainer.train(resume_from_checkpoint=self.plan.resume_from_checkpoint)
        
        # 保存
        save_model_and_tokenizer(model, tokenizer, output_dir)
        self._current_model_path = str(output_dir)
        
        self.logger.info(f"SFT completed. Model saved to {output_dir}")
    
    def _run_dpo(self, stage_config: StageConfig):
        """运行 DPO(Direct Preference Optimization) 阶段"""
        self.logger.info("Starting DPO stage")
        
        try:
            from trl import DPOTrainer
        except ImportError:
            raise ImportError("TRL is required for DPO training. Install: pip install trl")
        
        # 加载模型
        model_path = self._determine_model_path(stage_config)
        model, tokenizer = load_model_and_tokenizer(self.plan.model_config, model_path)
        
        # 加载 reference 模型
        ref_model = None
        if stage_config.reference_model_path:
            ref_model, _ = load_model_and_tokenizer(
                self.plan.model_config,
                stage_config.reference_model_path,
            )
        
        # 加载数据集
        dataset = self._load_dataset(stage_config)
        dataset_config = self.plan.dataset_configs[stage_config.stage_name]
        
        # 确保数据集包含必要字段
        required_fields = ["prompt", "chosen", "rejected"]
        for field in required_fields:
            if field not in dataset.column_names:
                # 尝试从配置映射
                if field == "prompt" and dataset_config.prompt_field in dataset.column_names:
                    dataset = dataset.rename_column(dataset_config.prompt_field, "prompt")
                elif field == "chosen" and dataset_config.chosen_field in dataset.column_names:
                    dataset = dataset.rename_column(dataset_config.chosen_field, "chosen")
                elif field == "rejected" and dataset_config.rejected_field in dataset.column_names:
                    dataset = dataset.rename_column(dataset_config.rejected_field, "rejected")
                else:
                    raise ValueError(f"Dataset missing required field: {field}")
        
        # 训练参数
        output_dir = self._get_output_dir(stage_config)
        training_args = self._build_training_arguments(stage_config, output_dir)
        
        # DPO Trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            beta=stage_config.dpo_beta,
            max_length=dataset_config.max_length,
            max_prompt_length=dataset_config.max_length // 2,
        )
        
        # 训练
        trainer.train(resume_from_checkpoint=self.plan.resume_from_checkpoint)
        
        # 保存
        save_model_and_tokenizer(model, tokenizer, output_dir)
        self._current_model_path = str(output_dir)
        
        self.logger.info(f"DPO completed. Model saved to {output_dir}")
    
    def _run_ppo(self, stage_config: StageConfig):
        """运行 PPO(Proximal Policy Optimization) 阶段"""
        self.logger.info("Starting PPO stage")
        
        try:
            from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        except ImportError:
            raise ImportError("TRL is required for PPO training. Install: pip install trl")
        
        # 加载模型(需要 value head)
        model_path = self._determine_model_path(stage_config)
        model, tokenizer = load_model_and_tokenizer(
            self.plan.model_config,
            model_path,
            requires_value_head=True,
        )
        
        # 获取 reward 函数
        reward_fn = get_reward_function(
            reward_function_path=stage_config.reward_function,
            reward_model_path=stage_config.reward_model_path,
            model_config=self.plan.model_config,
        )
        
        # 加载数据集
        dataset = self._load_dataset(stage_config)
        dataset_config = self.plan.dataset_configs[stage_config.stage_name]
        
        # PPO 配置
        output_dir = self._get_output_dir(stage_config)
        ppo_config = PPOConfig(
            model_name=model_path,
            learning_rate=stage_config.learning_rate,
            batch_size=stage_config.per_device_train_batch_size,
            mini_batch_size=stage_config.per_device_train_batch_size,
            gradient_accumulation_steps=stage_config.gradient_accumulation_steps,
            ppo_epochs=stage_config.ppo_epochs,
            init_kl_coef=stage_config.init_kl_coef,
            adap_kl_ctrl=stage_config.adap_kl_ctrl,
            log_with="wandb" if self.plan.use_wandb else None,
            project_kwargs={"project_name": self.plan.wandb_project} if self.plan.use_wandb else None,
        )
        
        # PPO Trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
        
        # 训练循环
        generation_kwargs = {
            "max_new_tokens": stage_config.stage_kwargs.get("max_new_tokens", 128),
            "temperature": stage_config.stage_kwargs.get("temperature", 0.7),
            "top_p": stage_config.stage_kwargs.get("top_p", 0.9),
            "do_sample": True,
        }
        
        for epoch in range(stage_config.num_train_epochs):
            for batch in ppo_trainer.dataloader:
                # 获取 prompts
                query_tensors = batch[dataset_config.text_field]
                
                # 生成响应
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs,
                )
                
                # 解码
                prompts = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
                responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                
                # 计算 rewards
                rewards = reward_fn(prompts, responses)
                rewards = [torch.tensor(r) for r in rewards]
                
                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)
        
        # 保存
        save_model_and_tokenizer(model, tokenizer, output_dir)
        self._current_model_path = str(output_dir)
        
        self.logger.info(f"PPO completed. Model saved to {output_dir}")
    
    def _run_grpo(self, stage_config: StageConfig):
        """运行 GRPO(Group Relative Policy Optimization) 阶段"""
        self.logger.info("Starting GRPO stage")
        
        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            raise ImportError("TRL is required for GRPO training. Install: pip install trl")
        
        # 加载模型
        model_path = self._determine_model_path(stage_config)
        model, tokenizer = load_model_and_tokenizer(self.plan.model_config, model_path)
        
        # 获取 reward 函数
        reward_fn = get_reward_function(
            reward_function_path=stage_config.reward_function,
            reward_model_path=stage_config.reward_model_path,
            model_config=self.plan.model_config,
        )
        
        # 加载数据集
        dataset = self._load_dataset(stage_config)
        
        # 训练参数
        output_dir = self._get_output_dir(stage_config)
        training_args = self._build_training_arguments(stage_config, output_dir)
        
        # GRPO Trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )
        
        # 训练
        trainer.train(resume_from_checkpoint=self.plan.resume_from_checkpoint)
        
        # 保存
        save_model_and_tokenizer(model, tokenizer, output_dir)
        self._current_model_path = str(output_dir)
        
        self.logger.info(f"GRPO completed. Model saved to {output_dir}")


# 为了向后兼容，保留 LLMVLMTrainingPipeline 作为别名
LLMVLMTrainingPipeline = LLMTrainer

__all__ = ["LLMTrainer", "LLMVLMTrainingPipeline"]

