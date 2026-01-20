"""
LLM/VLM 训练配置数据类
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from enum import Enum

from src.constants import PRETRAINED_MODEL_DIR, DATASET_ROOT_DIR


class TrainingStageType(str, Enum):
    """训练阶段类型"""
    PRETRAIN = "pretrain"
    SFT = "sft"
    DPO = "dpo"
    PPO = "ppo"
    GRPO = "grpo"


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型配置
    model_name_or_path: str  # HF model ID 或本地路径
    model_type: Literal["llm", "vlm"] = "llm"  # 模型类型
    
    # 设备与精度
    device: str = "auto"
    dtype: Optional[str] = None  # float16, bfloat16, float32
    trust_remote_code: bool = True
    
    # LoRA/PEFT 配置
    use_lora: bool = False
    lora_config: Dict[str, Any] = field(default_factory=dict)
    
    # 量化配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_config: Dict[str, Any] = field(default_factory=dict)
    
    # 其他模型参数
    gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 缓存目录
    cache_dir: str = PRETRAINED_MODEL_DIR


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 数据来源
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    dataset_path: Optional[Union[str, Path]] = None  # 本地路径
    dataset_split: str = "train"  # train, validation, test
    
    # 数据处理
    text_field: str = "text"  # 文本字段名
    image_field: Optional[str] = None  # 图像字段名(VLM)
    max_length: int = 2048
    
    # 特定阶段字段
    # SFT
    prompt_field: Optional[str] = "prompt"
    response_field: Optional[str] = "response"
    
    # DPO
    chosen_field: Optional[str] = "chosen"
    rejected_field: Optional[str] = "rejected"
    
    # PPO/GRPO
    reward_field: Optional[str] = None
    
    # 数据预处理
    preprocessing_num_workers: int = 4
    formatting_func: Optional[str] = None  # 可调用函数路径
    
    # 数据加载
    streaming: bool = False
    num_samples: Optional[int] = None  # 限制样本数
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # 自定义文本文件格式（用于txt文件）
    txt_delimiter: Optional[str] = None  # 分隔符，如 "|", "\t", ","
    txt_field_names: Optional[List[str]] = None  # 字段名称列表，如 ["prompt", "response"]


@dataclass
class StageConfig:
    """训练阶段配置"""
    # 阶段类型
    stage_type: TrainingStageType
    stage_name: str  # 自定义阶段名称
    
    # 输入输出
    load_from: Optional[str] = None  # 加载检查点路径(None则从ModelConfig加载)
    output_dir: Optional[str] = None  # 输出目录(None则自动生成)
    
    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    max_steps: int = -1  # -1 表示使用 num_train_epochs
    
    # 优化器
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    
    # DeepSpeed 配置
    deepspeed: Optional[str] = None  # DeepSpeed config path
    
    # DDP 配置
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    
    # 混合精度
    fp16: bool = False
    bf16: bool = False
    
    # 评估
    do_eval: bool = True
    eval_strategy: str = "steps"
    
    # 保存
    save_strategy: str = "steps"
    save_total_limit: int = 3
    
    # 阶段特定参数
    stage_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # DPO 专用
    dpo_beta: float = 0.1
    reference_model_path: Optional[str] = None
    
    # PPO/GRPO 专用
    reward_model_path: Optional[str] = None
    reward_function: Optional[str] = None  # 自定义 reward 函数路径
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True


@dataclass
class TrainingPlan:
    """完整训练计划"""
    # 基础配置
    task_name: str
    model_config: ModelConfig
    
    # 训练阶段列表
    stages: List[StageConfig] = field(default_factory=list)
    
    # 数据集配置(每个阶段可覆盖)
    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=dict)
    
    # 全局设置
    base_output_dir: str = "runs"
    seed: int = 42
    logging_dir: Optional[str] = None
    
    # 分布式训练
    use_torchrun: bool = False
    world_size: int = 1
    
    # 监控
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # 其他全局参数
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False


def load_training_plan_from_dict(config_dict: Dict[str, Any]) -> TrainingPlan:
    """从字典加载训练计划"""
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    
    dataset_configs = {}
    for stage_name, ds_cfg in config_dict.get("datasets", {}).items():
        dataset_configs[stage_name] = DatasetConfig(**ds_cfg)
    
    stages = []
    for stage_cfg in config_dict.get("stages", []):
        stage_type_str = stage_cfg.get("stage_type")
        stage_cfg["stage_type"] = TrainingStageType(stage_type_str)
        stages.append(StageConfig(**stage_cfg))
    
    plan = TrainingPlan(
        task_name=config_dict["task_name"],
        model_config=model_cfg,
        stages=stages,
        dataset_configs=dataset_configs,
        **{k: v for k, v in config_dict.items() 
           if k not in ("task_name", "model", "stages", "datasets")}
    )
    return plan


__all__ = [
    "TrainingStageType",
    "ModelConfig",
    "DatasetConfig",
    "StageConfig",
    "TrainingPlan",
    "load_training_plan_from_dict",
]

