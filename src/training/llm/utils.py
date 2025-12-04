"""
LLM/VLM 训练工具函数
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.training.llm.config import ModelConfig

logger = logging.getLogger(__name__)


def resolve_callable(func_path: str) -> Callable:
    """
    从字符串路径解析可调用对象
    
    Args:
        func_path: 格式为 "module.path:function_name"
        
    Returns:
        可调用对象
        
    Example:
        >>> func = resolve_callable("src.utils.data:my_formatter")
    """
    if ":" not in func_path:
        raise ValueError(f"Invalid function path: {func_path}. Expected 'module:function'")
    
    module_path, func_name = func_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    
    if not callable(func):
        raise ValueError(f"{func_path} is not callable")
    
    return func


def load_model_and_tokenizer(
    config: ModelConfig,
    model_name_or_path: Optional[str] = None,
    *,
    is_reward_model: bool = False,
    requires_value_head: bool = False,
) -> tuple[torch.nn.Module, Any]:
    """
    加载模型和分词器/处理器
    
    Args:
        config: 模型配置
        model_name_or_path: 覆盖配置中的模型路径
        is_reward_model: 是否为 reward 模型(用于 RL)
        requires_value_head: 是否需要 value head(PPO)
        
    Returns:
        (model, tokenizer/processor)
    """
    model_path = model_name_or_path or config.model_name_or_path
    logger.info(f"Loading model from {model_path}")
    
    # 构建量化配置
    quantization_config = None
    if config.load_in_4bit or config.load_in_8bit:
        bnb_config = config.bnb_config or {}
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=bnb_config.get("use_double_quant", True),
            bnb_4bit_quant_type=bnb_config.get("quant_type", "nf4"),
        )
    
    # 设备映射
    device_map = config.device if config.device != "auto" else "auto"
    
    # 数据类型
    torch_dtype = None
    if config.dtype:
        if config.dtype == "float16":
            torch_dtype = torch.float16
        elif config.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif config.dtype == "float32":
            torch_dtype = torch.float32
    
    # 模型加载参数
    model_kwargs = {
        "cache_dir": config.cache_dir,
        "trust_remote_code": config.trust_remote_code,
        "device_map": device_map,
        **config.model_kwargs,
    }
    
    if torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    # 根据需求选择模型类
    if requires_value_head:
        # PPO 需要 value head
        from trl import AutoModelForCausalLMWithValueHead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            **model_kwargs,
        )
    elif is_reward_model:
        # Reward 模型
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            **model_kwargs,
        )
    else:
        # 标准因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
    
    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # 量化训练准备
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_config.get("r", 8),
            lora_alpha=config.lora_config.get("lora_alpha", 16),
            target_modules=config.lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=config.lora_config.get("lora_dropout", 0.05),
            bias=config.lora_config.get("bias", "none"),
            task_type=config.lora_config.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_config)
        logger.info("LoRA enabled with config: %s", lora_config)
    
    # 加载 tokenizer/processor
    if config.model_type == "vlm":
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        tokenizer = processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded successfully: {model.__class__.__name__}")
    return model, tokenizer


def get_reward_function(
    reward_function_path: Optional[str] = None,
    reward_model_path: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> Callable[[list[str], list[str]], list[float]]:
    """
    获取 reward 函数
    
    Args:
        reward_function_path: 自定义 reward 函数路径
        reward_model_path: reward 模型路径
        model_config: 模型配置
        
    Returns:
        reward_fn(prompts, responses) -> scores
    """
    if reward_function_path:
        logger.info(f"Using custom reward function: {reward_function_path}")
        return resolve_callable(reward_function_path)
    
    if reward_model_path:
        logger.info(f"Loading reward model from: {reward_model_path}")
        
        # 加载 reward 模型
        if model_config is None:
            from src.training.llm.config import ModelConfig
            model_config = ModelConfig(model_name_or_path=reward_model_path)
        
        reward_model, reward_tokenizer = load_model_and_tokenizer(
            model_config,
            model_name_or_path=reward_model_path,
            is_reward_model=True,
        )
        reward_model.eval()
        
        def reward_fn(prompts: list[str], responses: list[str]) -> list[float]:
            """使用模型计算 reward"""
            texts = [p + r for p, r in zip(prompts, responses)]
            inputs = reward_tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(reward_model.device)
            
            with torch.no_grad():
                outputs = reward_model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            return scores
        
        return reward_fn
    
    # 默认简单 reward(长度奖励)
    logger.warning("No reward function or model specified, using default length-based reward")
    
    def default_reward(prompts: list[str], responses: list[str]) -> list[float]:
        return [float(len(r.split())) / 100.0 for r in responses]
    
    return default_reward


def save_model_and_tokenizer(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Union[str, Path],
):
    """
    保存模型和分词器
    
    Args:
        model: 模型
        tokenizer: 分词器/处理器
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    
    # 保存模型
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        # 对于 PEFT 模型
        model.module.save_pretrained(output_dir)
    
    # 保存 tokenizer/processor
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)
    
    logger.info("Model and tokenizer saved successfully")


__all__ = [
    "resolve_callable",
    "load_model_and_tokenizer",
    "get_reward_function",
    "save_model_and_tokenizer",
]

