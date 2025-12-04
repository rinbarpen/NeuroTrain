"""
LLM/VLM 训练模块
支持预训练、SFT、DPO、PPO、GRPO等多阶段训练流程
"""

from .config import (
    ModelConfig,
    DatasetConfig,
    StageConfig,
    TrainingPlan,
    TrainingStageType,
    load_training_plan_from_dict,
)
from .pipeline import LLMVLMTrainingPipeline

__all__ = [
    "ModelConfig",
    "DatasetConfig", 
    "StageConfig",
    "TrainingPlan",
    "TrainingStageType",
    "load_training_plan_from_dict",
    "LLMVLMTrainingPipeline",
]
