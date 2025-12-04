# NeuroTrain LLM/VLM 训练模块

## 🚀 快速开始

```bash
# 1. 安装依赖
proxy_on
uv pip install transformers datasets trl accelerate peft bitsandbytes deepspeed

# 2. 单卡训练
python scripts/train_llm.py --config configs/llm_training_example.toml

# 3. 多卡训练 + DeepSpeed
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json
```

## ✨ 主要特性

- ✅ **多阶段训练**: 预训练 → SFT → DPO → PPO → GRPO
- ✅ **LLM & VLM**: 同时支持纯文本和多模态模型
- ✅ **高效训练**: LoRA/QLoRA、Gradient Checkpointing、Flash Attention
- ✅ **分布式**: DeepSpeed ZeRO、torchrun DDP
- ✅ **灵活配置**: TOML/YAML/JSON 配置文件

## 📁 项目结构

```
src/training/llm/
├── __init__.py           # 模块导出
├── config.py             # 配置数据类 (ModelConfig, StageConfig, TrainingPlan)
├── utils.py              # 工具函数 (模型加载、LoRA、量化、reward)
└── pipeline.py           # 训练管线 (LLMVLMTrainingPipeline)

configs/
├── llm_training_example.toml    # LLM 训练示例 (SFT+DPO)
├── vlm_training_example.toml    # VLM 训练示例
└── deepspeed/
    ├── ds_config_zero2.json     # DeepSpeed ZeRO Stage 2
    └── ds_config_zero3.json     # DeepSpeed ZeRO Stage 3

scripts/
└── train_llm.py          # 训练入口脚本

docs/
└── LLM_VLM_TRAINING_GUIDE.md    # 详细文档
```

## 🎯 支持的训练阶段

| 阶段 | 说明 | 数据格式 |
|------|------|---------|
| **Pretrain** | 预训练/继续预训练 | `{"text": "..."}` |
| **SFT** | 监督微调/指令微调 | `{"instruction": "...", "output": "..."}` |
| **DPO** | 直接偏好优化 | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| **PPO** | 近端策略优化 | `{"prompt": "..."}` + reward_fn |
| **GRPO** | 群体相对策略优化 | `{"prompt": "..."}` + reward_fn |

## 📝 配置示例

```toml
task_name = "llama2_sft_dpo"
seed = 42

[model]
model_name_or_path = "meta-llama/Llama-2-7b-hf"
model_type = "llm"
dtype = "bfloat16"
use_lora = true
load_in_4bit = true

[model.lora_config]
r = 16
lora_alpha = 32

[datasets.sft]
dataset_path = "data/sft_instructions.jsonl"
max_length = 2048

[[stages]]
stage_type = "sft"
stage_name = "sft"
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-5
bf16 = true

[[stages]]
stage_type = "dpo"
stage_name = "dpo"
num_train_epochs = 2
dpo_beta = 0.1
```

## 🔧 高级功能

### LoRA/QLoRA 微调

```toml
[model]
use_lora = true
load_in_4bit = true
[model.lora_config]
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### DeepSpeed ZeRO

```bash
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/llm_training_example.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json
```

### 自定义 Reward 函数 (PPO/GRPO)

```python
# src/utils/rewards.py
def my_reward_function(prompts: list[str], responses: list[str]) -> list[float]:
    scores = []
    for prompt, response in zip(prompts, responses):
        # 自定义评分逻辑
        score = compute_score(prompt, response)
        scores.append(score)
    return scores
```

```toml
[[stages]]
stage_type = "ppo"
reward_function = "src.utils.rewards:my_reward_function"
```

### VLM 训练

```toml
[model]
model_name_or_path = "llava-hf/llava-1.5-7b-hf"
model_type = "vlm"

[datasets.vlm_sft]
dataset_path = "data/vlm_instructions.jsonl"
image_field = "image"
```

## 📊 训练输出

```
runs/llm/{task_name}/{timestamp}/
├── sft_stage/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer*
├── dpo_stage/
│   └── ...
└── ppo_stage/
    └── ...
```

## 🐛 常见问题

### CUDA OOM
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing = true`
- 使用 `load_in_4bit = true`

### 训练速度慢
- 使用 `bf16 = true`
- 启用 DeepSpeed
- 检查数据加载是否为瓶颈

### 模型下载失败
```bash
proxy_on
python scripts/train_llm.py --config ...
```

## 📖 详细文档

查看 [LLM_VLM_TRAINING_GUIDE.md](docs/LLM_VLM_TRAINING_GUIDE.md) 获取完整文档。

## 🎓 示例

### 1. LLaMA-2 SFT 训练

```bash
python scripts/train_llm.py --config configs/llm_training_example.toml
```

### 2. LLaVA VLM 训练

```bash
python scripts/train_llm.py --config configs/vlm_training_example.toml
```

### 3. 完整 RLHF 流程 (SFT→DPO→PPO)

编辑配置文件添加所有阶段，然后：

```bash
torchrun --nproc_per_node=4 scripts/train_llm.py \
    --config configs/full_rlhf_pipeline.toml \
    --deepspeed configs/deepspeed/ds_config_zero2.json
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

本项目遵循与 NeuroTrain 主项目相同的许可协议。

