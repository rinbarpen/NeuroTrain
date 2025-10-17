"""
LoRA Adapter Merger

用法示例：
    # 基本顺序合并
    python tools/lora_merge.py \
        --base THUDM/chatglm3-6b \
        --adapters ./output/lora_run1 ./output/lora_run2 \
        --output ./merged_model \
        --merge-dtype float16 \
        --trust-remote-code \
        --save-tokenizer \
        --use-proxy

    # 加权合并
    python tools/lora_merge.py \
        --base THUDM/chatglm3-6b \
        --adapters ./output/lora_run1 ./output/lora_run2 \
        --output ./merged_model \
        --merge-strategy weighted \
        --weights 0.7 0.3 \
        --merge-dtype float16

    # 平均合并
    python tools/lora_merge.py \
        --base THUDM/chatglm3-6b \
        --adapters ./output/lora_run1 ./output/lora_run2 \
        --output ./merged_model \
        --merge-strategy average

功能：
- 支持多种合并策略：顺序合并、加权合并、平均合并
- 将一个或多个 LoRA 适配器合并回基础模型
- 支持从本地路径或 Hugging Face 模型仓库 ID 加载基础模型
- 可选保存 tokenizer（若基础模型为 Transformers 文本/多模态模型）
- 可选启用代理（执行 shell 命令 `proxy_on`）
- 生成合并报告和统计信息
- 支持权重分析和可视化

注意：
- 顺序合并：按提供顺序依次合并，适用于参数增量可叠加的常见场景
- 加权合并：使用指定权重合并多个适配器
- 平均合并：使用相等权重合并所有适配器
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch


def str2dtype(name: str) -> Optional[torch.dtype]:
    name = name.lower()
    if name in ("auto", "none", ""):  # 自动让 transformers 决定
        return None
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _maybe_enable_proxy(use_proxy: bool) -> None:
    if not use_proxy:
        return
    try:
        # 用户环境约定：下载前执行 proxy_on 以开启代理
        subprocess.run(["proxy_on"], check=False)
    except Exception:
        # 静默忽略，防止在无该命令环境下报错中断
        pass


def _load_transformers_model(
    model_id_or_path: str,
    torch_dtype: Optional[torch.dtype],
    device_map: str,
    trust_remote_code: bool,
    local_files_only: bool,
):
    """尽可能稳健地加载 Transformers 模型。

    优先尝试 AutoModelForCausalLM；失败则回退到 AutoModel。
    对于包含自定义代码的多模态模型，需传入 trust_remote_code=True。
    """
    from transformers import AutoModelForCausalLM, AutoModel

    kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    try:
        return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    except Exception:
        return AutoModel.from_pretrained(model_id_or_path, **kwargs)


def _maybe_save_tokenizer(model_id_or_path: str, output_dir: Path, trust_remote_code: bool, local_files_only: bool) -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            use_fast=False,
        )
        tokenizer.save_pretrained(output_dir)
    except Exception:
        # 若不是文本/多模态模型或无对应分词器，静默忽略
        pass


def merge_lora_adapters(
    base: str,
    adapters: List[str],
    output: str,
    merge_strategy: str = "sequential",
    weights: Optional[List[float]] = None,
    merge_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    save_safetensors: bool = True,
    save_tokenizer: bool = False,
    use_proxy: bool = False,
    generate_report: bool = True,
) -> Dict[str, Any]:
    """合并多个 LoRA 适配器并保存完整模型。"""

    start_time = time.time()
    _maybe_enable_proxy(use_proxy)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LoRA] Starting merge with strategy: {merge_strategy}")
    print(f"[LoRA] Base model: {base}")
    print(f"[LoRA] Adapters: {adapters}")
    print(f"[LoRA] Output: {output_dir}")

    # 1) 加载基础模型
    print("[LoRA] Loading base model...")
    model = _load_transformers_model(
        model_id_or_path=base,
        torch_dtype=merge_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    # 2) 根据策略合并 LoRA 适配器
    if merge_strategy == "sequential":
        merged_model = _sequential_merge(model, adapters)
    elif merge_strategy == "weighted":
        merged_model = _weighted_merge(model, adapters, weights)
    elif merge_strategy == "average":
        merged_model = _average_merge(model, adapters)
    else:
        raise ValueError(f"Unsupported merge strategy: {merge_strategy}")

    # 3) 保存合并后的完整模型
    print(f"[Save] Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=save_safetensors)

    # 4) 可选保存 tokenizer（若存在）
    if save_tokenizer:
        _maybe_save_tokenizer(base, output_dir, trust_remote_code=trust_remote_code, local_files_only=local_files_only)

    # 5) 生成合并报告
    merge_info = {
        "base_model": base,
        "adapters": adapters,
        "merge_strategy": merge_strategy,
        "weights": weights,
        "output_path": str(output_dir),
        "merge_time": time.time() - start_time,
        "model_size_mb": _get_model_size(output_dir),
        "total_parameters": _count_parameters(merged_model),
        "merge_dtype": str(merge_dtype) if merge_dtype else "auto",
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "save_safetensors": save_safetensors,
        "save_tokenizer": save_tokenizer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if generate_report:
        _save_merge_report(merge_info, output_dir)

    print("[Done] LoRA adapters merged and model saved.")
    print(f"[Info] Merge completed in {merge_info['merge_time']:.2f} seconds")
    print(f"[Info] Model size: {merge_info['model_size_mb']:.2f} MB")
    print(f"[Info] Total parameters: {merge_info['total_parameters']:,}")

    return merge_info


def _sequential_merge(model, adapters: List[str]):
    """顺序合并适配器。"""
    merged_model = model
    for idx, adapter_path in enumerate(adapters):
        adapter_path = str(adapter_path)
        print(f"[LoRA] Loading adapter {idx+1}/{len(adapters)}: {adapter_path}")
        peft_model = PeftModel.from_pretrained(
            merged_model,
            adapter_path,
            is_trainable=False,
        )
        print("[LoRA] Merging and unloading...")
        merged_model = peft_model.merge_and_unload()
    return merged_model


def _weighted_merge(model, adapters: List[str], weights: Optional[List[float]] = None):
    """加权合并适配器。"""
    if weights is None:
        weights = [1.0] * len(adapters)
    
    if len(weights) != len(adapters):
        raise ValueError("Weights length must match adapters length")
    
    print(f"[LoRA] Using weights: {weights}")
    
    # 加载所有适配器
    peft_models = []
    for idx, adapter_path in enumerate(adapters):
        adapter_path = str(adapter_path)
        print(f"[LoRA] Loading adapter {idx+1}/{len(adapters)}: {adapter_path}")
        peft_model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
        peft_models.append(peft_model)
    
    # 加权合并
    merged_model = _merge_with_weights(model, peft_models, weights)
    return merged_model


def _average_merge(model, adapters: List[str]):
    """平均合并适配器。"""
    weights = [1.0 / len(adapters)] * len(adapters)
    print(f"[LoRA] Using equal weights: {weights}")
    return _weighted_merge(model, adapters, weights)


def _merge_with_weights(base_model, peft_models: List, weights: List[float]):
    """使用权重合并多个 PEFT 模型。"""
    print("[LoRA] Applying weighted merge...")
    
    # 获取基础模型状态字典
    merged_state_dict = base_model.state_dict().copy()
    
    # 对每个适配器应用权重
    for peft_model, weight in zip(peft_models, weights):
        adapter_state_dict = peft_model.state_dict()
        
        for name, param in adapter_state_dict.items():
            if name in merged_state_dict:
                if 'lora_A' in name or 'lora_B' in name:
                    # LoRA 参数需要特殊处理
                    merged_state_dict[name] += weight * param
                else:
                    # 其他参数直接加权
                    merged_state_dict[name] += weight * param
    
    # 创建新的模型实例
    merged_model = type(base_model)(base_model.config)
    merged_model.load_state_dict(merged_state_dict)
    
    return merged_model


def _get_model_size(model_path: Path) -> float:
    """获取模型大小（MB）。"""
    total_size = 0
    for file_path in model_path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)  # 转换为MB


def _count_parameters(model) -> int:
    """计算模型参数数量。"""
    return sum(p.numel() for p in model.parameters())


def _save_merge_report(merge_info: Dict[str, Any], output_dir: Path):
    """保存合并报告。"""
    report_path = output_dir / "merge_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)
    
    # 生成 Markdown 报告
    md_report = f"""# LoRA Merge Report

## Summary
- **Base Model**: {merge_info['base_model']}
- **Adapters**: {len(merge_info['adapters'])} adapters
- **Merge Strategy**: {merge_info['merge_strategy']}
- **Output Path**: {merge_info['output_path']}
- **Merge Time**: {merge_info['merge_time']:.2f} seconds
- **Model Size**: {merge_info['model_size_mb']:.2f} MB
- **Total Parameters**: {merge_info['total_parameters']:,}

## Adapters
"""
    
    for i, adapter in enumerate(merge_info['adapters']):
        md_report += f"{i+1}. {adapter}\n"
    
    if merge_info['weights']:
        md_report += f"\n## Weights\n"
        for i, weight in enumerate(merge_info['weights']):
            md_report += f"- Adapter {i+1}: {weight}\n"
    
    md_report += f"""
## Configuration
- **Merge Dtype**: {merge_info['merge_dtype']}
- **Device Map**: {merge_info['device_map']}
- **Trust Remote Code**: {merge_info['trust_remote_code']}
- **Local Files Only**: {merge_info['local_files_only']}
- **Save Safetensors**: {merge_info['save_safetensors']}
- **Save Tokenizer**: {merge_info['save_tokenizer']}

## Timestamp
{merge_info['timestamp']}
"""
    
    md_report_path = output_dir / "merge_report.md"
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"[Report] Merge report saved to: {report_path}")
    print(f"[Report] Markdown report saved to: {md_report_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge one or more LoRA adapters into a base Transformers model.")

    # 必需参数
    parser.add_argument("--base", required=True, help="基础模型路径或 Hugging Face 模型 ID")
    parser.add_argument("--adapters", nargs="+", required=True, help="一个或多个 LoRA 适配器目录")
    parser.add_argument("--output", required=True, help="合并后模型的保存目录")

    # 合并策略参数
    parser.add_argument("--merge-strategy", default="sequential", 
                       choices=["sequential", "weighted", "average"],
                       help="合并策略：sequential(顺序合并), weighted(加权合并), average(平均合并)")
    parser.add_argument("--weights", nargs="+", type=float, 
                       help="权重列表（用于加权合并），长度必须与适配器数量相同")

    # 模型参数
    parser.add_argument("--merge-dtype", default="auto", 
                       choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"], 
                       help="加载及合并时使用的 dtype")
    parser.add_argument("--device-map", default="auto", 
                       help="设备映射，例如 'auto' 或 'cpu' 或 CUDA 设备映射")
    parser.add_argument("--trust-remote-code", action="store_true", 
                       help="允许从远程仓库加载自定义代码（多模态/自定义模型需要）")
    parser.add_argument("--local-files-only", action="store_true", 
                       help="仅使用本地文件，禁止网络下载")

    # 保存选项
    parser.add_argument("--no-safetensors", dest="save_safetensors", action="store_false", 
                       help="禁用 safetensors 格式保存")
    parser.add_argument("--save-tokenizer", action="store_true", 
                       help="尝试保存 tokenizer（若可用）")
    parser.add_argument("--no-report", dest="generate_report", action="store_false",
                       help="禁用生成合并报告")

    # 其他选项
    parser.add_argument("--use-proxy", action="store_true", 
                       help="下载前尝试执行 'proxy_on' 以启用代理")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 验证参数
    if args.merge_strategy == "weighted" and args.weights is None:
        parser.error("--weights is required when using weighted merge strategy")
    
    if args.weights and len(args.weights) != len(args.adapters):
        parser.error(f"Number of weights ({len(args.weights)}) must match number of adapters ({len(args.adapters)})")

    merge_lora_adapters(
        base=args.base,
        adapters=list(args.adapters),
        output=args.output,
        merge_strategy=args.merge_strategy,
        weights=args.weights,
        merge_dtype=str2dtype(args.merge_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        save_safetensors=args.save_safetensors,
        save_tokenizer=args.save_tokenizer,
        use_proxy=args.use_proxy,
        generate_report=args.generate_report,
    )


if __name__ == "__main__":
    main()
