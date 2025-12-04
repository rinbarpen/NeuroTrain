#!/usr/bin/env python3
"""
LLM/VLM 训练脚本
支持单机、多卡、DeepSpeed 等多种训练模式

用法:
    # 单卡训练
    python scripts/train_llm.py --config configs/llm_training_example.toml
    
    # 多卡 torchrun + DeepSpeed
    torchrun --nproc_per_node=4 scripts/train_llm.py \\
        --config configs/llm_training_example.toml \\
        --deepspeed configs/deepspeed/ds_config_zero2.json
    
    # 使用代理下载模型
    proxy_on
    python scripts/train_llm.py --config configs/llm_training_example.toml
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.training.llm import load_training_plan_from_dict, LLMVLMTrainingPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM/VLM Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file (.toml, .yaml, .json)",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config (overrides config file)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torchrun)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("NeuroTrain LLM/VLM Training Pipeline")
    logger.info("=" * 80)
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config from: {config_path}")
    config_dict = load_config(config_path)
    
    # 覆盖命令行参数
    if args.deepspeed:
        # 为所有阶段设置 DeepSpeed
        for stage in config_dict.get("stages", []):
            stage["deepspeed"] = args.deepspeed
    
    if args.local_rank != -1:
        # 为所有阶段设置 local_rank
        for stage in config_dict.get("stages", []):
            stage["local_rank"] = args.local_rank
    
    if args.resume:
        config_dict["resume_from_checkpoint"] = args.resume
    
    # 构建训练计划
    training_plan = load_training_plan_from_dict(config_dict)
    
    logger.info(f"Task: {training_plan.task_name}")
    logger.info(f"Model: {training_plan.model_config.model_name_or_path}")
    logger.info(f"Model Type: {training_plan.model_config.model_type}")
    logger.info(f"Training Stages: {len(training_plan.stages)}")
    
    for idx, stage in enumerate(training_plan.stages, 1):
        logger.info(f"  Stage {idx}: {stage.stage_name} ({stage.stage_type.value})")
    
    # 执行训练
    pipeline = LLMVLMTrainingPipeline(training_plan)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

