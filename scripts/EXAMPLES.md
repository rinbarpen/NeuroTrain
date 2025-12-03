# 脚本使用示例

## 🚀 训练示例

### 单GPU训练

```bash
# 基础训练
bash scripts/train.sh -c configs/single/train.toml --train

# 指定设备
bash scripts/train.sh -c configs/single/train.toml --train -d cuda:1

# 使用统一脚本
bash scripts/run_all.sh train -c configs/single/train.toml
```

### DDP多卡训练

```bash
# 2卡训练
bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 2

# 4卡训练
bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 4

# 8卡训练
bash scripts/train.sh -c configs/ddp_example.toml -t ddp -g 8

# 使用统一脚本
bash scripts/run_all.sh train -c configs/ddp_example.toml -t ddp -g 4
```

### DeepSpeed训练

```bash
# 2卡DeepSpeed训练
bash scripts/train.sh -c configs/deepspeed_example.yaml -t deepspeed -g 2

# 4卡DeepSpeed训练
bash scripts/train.sh -c configs/deepspeed_example.yaml -t deepspeed -g 4

# 使用统一脚本
bash scripts/run_all.sh train -c configs/deepspeed_example.yaml -t deepspeed -g 2
```

### 测试和预测

```bash
# 测试
bash scripts/train.sh -c configs/single/train.toml --test
# 或
bash scripts/run_all.sh test -c configs/single/train.toml

# 预测
bash scripts/train.sh -c configs/single/train.toml --predict
# 或
bash scripts/run_all.sh predict -c configs/single/train.toml
```

## 📊 分析器示例

### 指标分析器

```bash
# 分析指定run_id的指标
bash scripts/analyze.sh metrics --run_id experiment_001

# 分析多个run_id
bash scripts/analyze.sh metrics --run_id experiment_001 experiment_002

# 使用统一脚本
bash scripts/run_all.sh analyze metrics --run_id experiment_001
```

### 数据集分析器

```bash
# 分析CIFAR-10数据集
bash scripts/analyze.sh dataset --root_dir data/cifar10

# 分析指定数据集并保存结果
bash scripts/analyze.sh dataset --root_dir data/cifar10 --output_dir outputs/analysis

# 使用统一脚本
bash scripts/run_all.sh analyze dataset --root_dir data/cifar10
```

### 注意力分析器

```bash
# 分析模型注意力
bash scripts/analyze.sh attention --model_path runs/model.pth --input_dir data/images

# 分析并可视化
bash scripts/analyze.sh attention --model_path runs/model.pth --input_dir data/images --visualize

# 使用统一脚本
bash scripts/run_all.sh analyze attention --model_path runs/model.pth
```

### 掩码分析器

```bash
# 分析掩码
bash scripts/analyze.sh mask --input_dir data/images --output_dir outputs/masks

# 分析并生成统计
bash scripts/analyze.sh mask --input_dir data/images --output_dir outputs/masks --stats

# 使用统一脚本
bash scripts/run_all.sh analyze mask --input_dir data/images --output_dir outputs/masks
```

### 关系分析器

```bash
# 使用配置文件分析关系
bash scripts/analyze.sh relation --config configs/relation.yaml

# 直接指定参数
bash scripts/analyze.sh relation --input_file data/relations.json --output_dir outputs/relations

# 使用统一脚本
bash scripts/run_all.sh analyze relation --config configs/relation.yaml
```

### LoRA分析器

```bash
# 分析LoRA权重
bash scripts/analyze.sh lora --model_path runs/model.pth --lora_path runs/lora.pt

# 分析并比较
bash scripts/analyze.sh lora --model_path runs/model.pth --lora_path runs/lora.pt --compare

# 使用统一脚本
bash scripts/run_all.sh analyze lora --model_path runs/model.pth --lora_path runs/lora.pt
```

## 🔧 高级用法

### 自定义环境

```bash
# 使用不同的conda环境
bash scripts/train.sh -c configs/single/train.toml --train -e myenv

bash scripts/analyze.sh metrics --run_id experiment_001 -e myenv
```

### 传递额外参数

```bash
# 训练时传递额外参数
bash scripts/train.sh -c configs/single/train.toml --train -- --batch_size 64 --epoch 200

# 分析器传递额外参数
bash scripts/analyze.sh metrics --run_id experiment_001 -- --output_format json
```

### 组合使用

```bash
# 训练后立即测试
bash scripts/train.sh -c configs/single/train.toml --train && \
bash scripts/train.sh -c configs/single/train.toml --test

# 训练后分析指标
bash scripts/train.sh -c configs/single/train.toml --train && \
bash scripts/analyze.sh metrics --run_id $(cat runs/latest_run_id.txt)
```

## 📝 配置文件示例

### DDP训练配置

确保配置文件中包含：

```yaml
ddp:
  enabled: true
  log_level: "INFO"
```

### DeepSpeed训练配置

确保配置文件中包含：

```yaml
deepspeed:
  enabled: true
  zero_stage: 2
  fp16: false
  bf16: true
```

## ⚠️ 常见问题

### 问题1: DDP训练失败

**解决方案**:
- 确保配置文件中有 `ddp.enabled: true`
- 检查GPU数量是否正确
- 确保所有GPU可见：`nvidia-smi`

### 问题2: DeepSpeed未找到

**解决方案**:
```bash
pip install deepspeed
```

### 问题3: 分析器找不到模块

**解决方案**:
- 确保在项目根目录运行
- 检查conda环境是否正确激活
- 确保所有依赖已安装

## 🔗 相关文档

- [脚本README](README.md)
- [训练文档](../docs/)
- [数据集采样配置](../docs/dataset_sampling.md)

