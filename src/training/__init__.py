"""
NeuroTrain 训练模块
支持传统监督学习与大模型(LLM/VLM)的预训练、SFT、DPO、PPO、GRPO等多阶段训练
"""
from src.training.llm import load_training_plan_from_dict, LLMVLMTrainingPipeline, ModelConfig, DatasetConfig, StageConfig, TrainingPlan, TrainingStageType

__all__ = [
    "load_training_plan_from_dict",
    "LLMVLMTrainingPipeline",
    "ModelConfig",
    "DatasetConfig",
    "StageConfig",
    "TrainingPlan",
    "TrainingStageType",
]
