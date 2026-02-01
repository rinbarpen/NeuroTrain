# from .inference import SegmentPredictor
from .trainer import Trainer
from .predictor import Predictor, get_predictor
from .inferencer import Inferencer
from .tester import Tester
try:
    from .deepspeed_trainer import DeepSpeedTrainer
except ImportError:
    DeepSpeedTrainer = None
from .multi_predictor import MultiModelPredictor

# LLM/VLM 训练器（可选，需要 datasets 等）
try:
    from .llm import (
        LLMTrainer,
        LLMVLMTrainingPipeline,
        load_training_plan_from_dict,
        ModelConfig,
        DatasetConfig,
        StageConfig,
        TrainingPlan,
        TrainingStageType,
    )
except ImportError:
    LLMTrainer = None
    LLMVLMTrainingPipeline = None
    load_training_plan_from_dict = None
    ModelConfig = None
    DatasetConfig = None
    StageConfig = None
    TrainingPlan = None
    TrainingStageType = None

__all__ = [
    "Trainer",
    "Predictor",
    "get_predictor",
    "Inferencer",
    "Tester",
    "DeepSpeedTrainer",  # None when deepspeed not installed
    "MultiModelPredictor",
    "LLMTrainer",
    "LLMVLMTrainingPipeline",
    "load_training_plan_from_dict",
    "ModelConfig",
    "DatasetConfig",
    "StageConfig",
    "TrainingPlan",
    "TrainingStageType",
]
