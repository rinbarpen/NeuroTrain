# LLM/VLM 训练指南

## 概述

NeuroTrain 支持大语言模型(LLM)和视觉语言模型(VLM)的完整训练流程，包括：

- **预训练 (Pretrain)**: 从头训练或继续预训练
- **监督微调 (SFT)**: 指令微调、对话微调
- **偏好对齐**:
  - **DPO (Direct Preference Optimization)**: 直接偏好优化
  - **PPO (Proximal Policy Optimization)**: 近端策略优化
  - **GRPO (Group Relative Policy Optimization)**: 群体相对策略优化

## 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate ntrain

# 安装依赖(使用代理)
proxy_on
uv pip install transformers datasets trl accelerate peft bitsandbytes deepspeed
```

### 2. 准备数据

数据应放置在 `data/` 目录下,支持多种格式:

**SFT 数据格式** (jsonl):
```json
{"instruction": "用户问题", "output": "模型回答"}
{"instruction": "What is AI?", "output": "AI stands for..."}
```

**DPO 数据格式** (jsonl):
```json
{"prompt": "用户问题", "chosen": "好的回答", "rejected": "差的回答"}
```

**VLM 数据格式** (jsonl):
```json
{"conversations": [{"role": "user", "content": "描述这张图片"}, {"role": "assistant", "content": "这是..."}], "image": "path/to/image.jpg"}
```

### 3. 配置训练计划

参考示例配置:
- `configs/llm_training_example.toml` - LLM 训练(SFT+DPO)
- `configs/vlm_training_example.toml` - VLM 训练(SFT)

关键配置项:

```toml
# 基础配置
task_name = "my_llm_training"
seed = 42

# 模型配置
[model]
model_name_or_path = "meta-llama/Llama-2-7b-hf"
model_type = "llm"  # 或 "vlm"
dtype = "bfloat16"

# LoRA 配置
use_lora = true
[model.lora_config]
r = 16
lora_alpha = 32

# 量化(可选)
load_in_4bit = true

# 训练阶段
[[stages]]
stage_type = "sft"
stage_name = "sft_stage"
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-5
```

### 4. 开始训练

#### 单卡训练

```bash
python scripts/train_llm.py --config configs/llm_training_example.toml
```

#### 多卡训练 (torchrun)

```bash
# 4卡训练
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml
```

#### DeepSpeed 训练

```bash
# ZeRO Stage 2
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json

# ZeRO Stage 3 (大模型)
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero3.json
```

#### 恢复训练

```bash
python scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --resume runs/llm/my_task/20240101_120000/sft_stage/checkpoint-1000
```

## 训练阶段详解

### 预训练 (Pretrain)

从头训练或继续预训练语言模型。

**配置示例**:
```toml
[[stages]]
stage_type = "pretrain"
stage_name = "pretrain"
num_train_epochs = 1
per_device_train_batch_size = 8
learning_rate = 3e-4

[datasets.pretrain]
dataset_path = "data/pretrain_corpus.jsonl"
text_field = "text"
max_length = 2048
```

**数据格式**:
```json
{"text": "大量的原始文本数据..."}
```

### 监督微调 (SFT)

使用指令-响应对进行有监督微调。

**配置示例**:
```toml
[[stages]]
stage_type = "sft"
stage_name = "sft"
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-5

[datasets.sft]
dataset_path = "data/sft_instructions.jsonl"
prompt_field = "instruction"
response_field = "output"
max_length = 2048
```

**自定义格式化函数**:
```python
# src/utils/data_formatters.py
def my_formatting_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
```

配置中指定:
```toml
[datasets.sft]
formatting_func = "src.utils.data_formatters:my_formatting_func"
```

### DPO 训练

使用偏好对进行直接偏好优化。

**配置示例**:
```toml
[[stages]]
stage_type = "dpo"
stage_name = "dpo"
num_train_epochs = 2
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
learning_rate = 5e-6
dpo_beta = 0.1

[datasets.dpo]
dataset_path = "data/preference_pairs.jsonl"
prompt_field = "prompt"
chosen_field = "chosen"
rejected_field = "rejected"
```

**数据格式**:
```json
{
  "prompt": "用户问题",
  "chosen": "更好的回答",
  "rejected": "较差的回答"
}
```

### PPO 训练

使用强化学习进行策略优化。

**配置示例**:
```toml
[[stages]]
stage_type = "ppo"
stage_name = "ppo"
num_train_epochs = 1
per_device_train_batch_size = 4
learning_rate = 1e-5
ppo_epochs = 4
init_kl_coef = 0.2

# 使用 reward 模型
reward_model_path = "path/to/reward/model"

# 或使用自定义 reward 函数
reward_function = "src.utils.rewards:my_reward_function"

[datasets.ppo]
dataset_path = "data/ppo_prompts.jsonl"
text_field = "prompt"
```

**自定义 Reward 函数**:
```python
# src/utils/rewards.py
def my_reward_function(prompts: list[str], responses: list[str]) -> list[float]:
    """
    计算每个响应的奖励分数
    
    Args:
        prompts: 提示列表
        responses: 响应列表
        
    Returns:
        奖励分数列表
    """
    scores = []
    for prompt, response in zip(prompts, responses):
        # 自定义评分逻辑
        score = len(response.split()) / 100.0  # 示例:基于长度
        scores.append(score)
    return scores
```

### GRPO 训练

群体相对策略优化。

**配置示例**:
```toml
[[stages]]
stage_type = "grpo"
stage_name = "grpo"
num_train_epochs = 1
per_device_train_batch_size = 4
learning_rate = 1e-5
reward_function = "src.utils.rewards:group_reward_function"

[datasets.grpo]
dataset_path = "data/grpo_data.jsonl"
text_field = "prompt"
```

## VLM 训练

视觉语言模型训练支持图像+文本的多模态数据。

**配置示例**:
```toml
[model]
model_name_or_path = "llava-hf/llava-1.5-7b-hf"
model_type = "vlm"

[datasets.vlm_sft]
dataset_path = "data/vlm_instructions.jsonl"
image_field = "image"
prompt_field = "conversations"
```

**数据格式**:
```json
{
  "image": "data/images/img001.jpg",
  "conversations": [
    {"role": "user", "content": "描述这张图片"},
    {"role": "assistant", "content": "这是一张..."}
  ]
}
```

## 高级功能

### LoRA 微调

在配置中启用 LoRA:

```toml
[model]
use_lora = true
[model.lora_config]
r = 16
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_dropout = 0.05
```

### 量化训练

支持 4-bit 和 8-bit 量化:

```toml
[model]
load_in_4bit = true
[model.bnb_config]
use_double_quant = true
quant_type = "nf4"
```

### Gradient Checkpointing

节省显存:

```toml
[model]
gradient_checkpointing = true
```

### DeepSpeed ZeRO

配置文件在 `configs/deepspeed/`:
- `ds_config_zero2.json` - ZeRO Stage 2 (推荐用于中等模型)
- `ds_config_zero3.json` - ZeRO Stage 3 (大模型,需要CPU卸载)

使用方式:
```bash
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json
```

### WandB 监控

在配置中启用:

```toml
use_wandb = true
wandb_project = "my-llm-project"
wandb_entity = "my-team"
```

## 多阶段训练流程

配置文件支持定义多个连续阶段,每个阶段自动从上一阶段加载模型:

```toml
# Stage 1: SFT
[[stages]]
stage_type = "sft"
stage_name = "sft"
# ... 配置 ...

# Stage 2: DPO (自动从 SFT 阶段加载)
[[stages]]
stage_type = "dpo"
stage_name = "dpo"
load_from = ""  # 留空则从上一阶段加载
# ... 配置 ...

# Stage 3: PPO (自动从 DPO 阶段加载)
[[stages]]
stage_type = "ppo"
stage_name = "ppo"
# ... 配置 ...
```

## 输出结构

训练输出保存在 `runs/` 目录:

```
runs/
└── llm/
    └── my_task/
        └── 20240101_120000/
            ├── sft_stage/
            │   ├── checkpoint-500/
            │   ├── checkpoint-1000/
            │   └── ...
            ├── dpo_stage/
            │   └── ...
            └── ppo_stage/
                └── ...
```

每个阶段目录包含:
- `checkpoint-*/` - 训练检查点
- `pytorch_model.bin` - 最终模型权重
- `config.json` - 模型配置
- `tokenizer*` - 分词器文件

## 常见问题

### 1. CUDA Out of Memory

**解决方案**:
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`
- 使用量化 (`load_in_4bit`)
- 使用 DeepSpeed ZeRO Stage 3

### 2. 训练速度慢

**优化建议**:
- 使用 `bf16=true` (如果硬件支持)
- 启用 Flash Attention (如果可用)
- 使用 DeepSpeed
- 增加 `per_device_train_batch_size`

### 3. 模型下载失败

**解决方案**:
```bash
# 使用代理
proxy_on
python scripts/train_llm.py --config ...
```

### 4. 多卡训练同步问题

**确保**:
- 使用 `torchrun` 而非 `python`
- 所有进程看到相同的配置
- 检查 `CUDA_VISIBLE_DEVICES`

## 参考资源

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 示例脚本

完整的训练流程示例:

```bash
#!/bin/bash
# train_full_pipeline.sh

# 1. 准备环境
conda activate ntrain
proxy_on

# 2. 准备数据(假设已经准备好)
# python scripts/prepare_data.py

# 3. SFT 训练
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json

# 4. 评估
# python scripts/evaluate_llm.py \
#     --model runs/llm/my_task/latest/sft_stage \
#     --test_data data/test.jsonl

echo "Training pipeline completed!"
```

