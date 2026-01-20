"""
LLM/VLM 训练模块
支持预训练、SFT、DPO、PPO、GRPO等多阶段训练流程

LLMTrainer 提供以下公开方法:
- trainer.sft()      - 监督微调 (Supervised Fine-Tuning)
- trainer.dpo()      - 直接偏好优化 (Direct Preference Optimization)
- trainer.pretrain() - 预训练
- trainer.ppo()      - 近端策略优化 (Proximal Policy Optimization)
- trainer.grpo()     - 群体相对策略优化 (Group Relative Policy Optimization)
- trainer.train()    - 执行完整的多阶段训练计划
"""

from .config import (
    ModelConfig,
    DatasetConfig,
    StageConfig,
    TrainingPlan,
    TrainingStageType,
    load_training_plan_from_dict,
)
from .trainer import LLMTrainer, LLMVLMTrainingPipeline

__all__ = [
    "ModelConfig",
    "DatasetConfig", 
    "StageConfig",
    "TrainingPlan",
    "TrainingStageType",
    "load_training_plan_from_dict",
    "LLMTrainer",
    "LLMVLMTrainingPipeline",  # 向后兼容
]

