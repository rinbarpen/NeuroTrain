"""
Qwen模型推理模块
提供Qwen-Med和Qwen-VL-Med模型的加载和推理功能，支持数据集批量推理
"""

import torch
from typing import Optional, List, Union, Dict, Any, Literal
from PIL import Image
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def load_qwen_med(
    model_path: str = "Echelon-AI/Med-Qwen2-7B",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
):
    """
    加载Qwen-Med纯文本模型
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备，如 "cuda:3" 或 "cpu"
        cache_dir: 缓存目录，如果为None则使用环境变量HF_HUB_CACHE
        dtype: 模型数据类型
        
    Returns:
        tuple: (model, tokenizer)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE", "cache/models/pretrained")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir
    )
    
    return model, tokenizer


def load_qwen_vl_med(
    model_path: str = "AdaptLLM/biomed-Qwen2-VL-2B-Instruct",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
):
    """
    加载Qwen-VL-Med视觉语言模型
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备，如 "cuda:3" 或 "cpu"
        cache_dir: 缓存目录，如果为None则使用环境变量HF_HUB_CACHE
        dtype: 模型数据类型
        
    Returns:
        tuple: (model, processor)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE", "cache/models/pretrained")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path, cache_dir=cache_dir
    )
    
    return model, processor


def preprocess_image(image: Union[torch.Tensor, Image.Image]) -> Image.Image:
    """
    预处理图像，将tensor转换为PIL Image
    
    Args:
        image: 输入图像，可以是torch.Tensor或PIL.Image
        
    Returns:
        PIL.Image: 处理后的PIL图像
    """
    if isinstance(image, torch.Tensor):
        # 如果是tensor，转换为PIL Image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # 如果像素值是0~1，乘255再转为uint8
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype("uint8")
        image = Image.fromarray(image_np)
    return image


def qwen_med_chat(
    model,
    tokenizer,
    new_text: str,
    history: Optional[List] = None,
    max_new_tokens: int = 200,
    **kwargs,
) -> str:
    """
    Qwen-Med纯文本对话推理
    
    Args:
        model: Qwen模型
        tokenizer: 分词器
        new_text: 新的文本输入
        history: 对话历史（可选）
        max_new_tokens: 最大生成token数
        **kwargs: 传递给model.generate的其他参数
        
    Returns:
        str: 模型生成的回复文本
    """
    if history is None:
        history = []
    
    messages = history + [
        {"role": "user", "content": new_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def qwen_vl_med_chat(
    model,
    processor,
    new_text: str,
    new_image: Union[torch.Tensor, Image.Image],
    history: Optional[List] = None,
    max_new_tokens: int = 200,
    **kwargs,
) -> Union[str, List[str]]:
    """
    Qwen-VL-Med视觉语言对话推理
    
    Args:
        model: Qwen-VL模型
        processor: 处理器
        new_text: 新的文本输入
        new_image: 输入图像（可以是torch.Tensor或PIL.Image）
        history: 对话历史（可选）
        max_new_tokens: 最大生成token数
        **kwargs: 传递给model.generate的其他参数
        
    Returns:
        str or List[str]: 模型生成的回复文本（单张图像返回str，多张返回List[str]）
    """
    if history is None:
        history = []
    
    # 处理图像
    new_image = preprocess_image(new_image)
    
    messages = history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": new_image,
                },
                {"type": "text", "text": new_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    # 如果只有一张图像，返回单个字符串
    if len(output_text) == 1:
        return output_text[0]
    return output_text


def qwen_med_inference(
    model_path: str = "Echelon-AI/Med-Qwen2-7B",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    **kwargs,
) -> str:
    """
    完整的Qwen-Med推理流程（包含模型加载，纯文本）
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备
        cache_dir: 缓存目录
        prompt: 输入提示文本
        **kwargs: 传递给qwen_med_chat的其他参数
        
    Returns:
        str: 模型生成的回复文本
    """
    # 加载模型
    model, tokenizer = load_qwen_med(
        model_path=model_path,
        device=device,
        cache_dir=cache_dir,
    )
    
    # 执行推理
    output = qwen_med_chat(
        model=model,
        tokenizer=tokenizer,
        new_text=prompt,
        **kwargs,
    )
    
    return output


def qwen_vl_med_inference(
    model_path: str = "AdaptLLM/biomed-Qwen2-VL-2B-Instruct",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    image: Union[torch.Tensor, Image.Image, str] = None,
    **kwargs,
) -> Union[str, List[str]]:
    """
    完整的Qwen-VL-Med推理流程（包含模型加载，视觉语言）
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备
        cache_dir: 缓存目录
        prompt: 输入提示文本
        image: 输入图像（可以是torch.Tensor、PIL.Image或图像路径）
        **kwargs: 传递给qwen_vl_med_chat的其他参数
        
    Returns:
        str or List[str]: 模型生成的回复文本
    """
    # 加载模型
    model, processor = load_qwen_vl_med(
        model_path=model_path,
        device=device,
        cache_dir=cache_dir,
    )
    
    # 如果image是路径字符串，加载图像
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # 检查图像是否提供
    if image is None:
        raise ValueError("必须提供图像参数（可以是torch.Tensor、PIL.Image或图像路径）")
    
    # 执行推理
    output = qwen_vl_med_chat(
        model=model,
        processor=processor,
        new_text=prompt,
        new_image=image,
        **kwargs,
    )
    
    return output


def load_dataset(dataset_name: str, root_dir: Union[str, Path], split: Literal["train", "valid", "test"] = "train", **kwargs):
    """
    加载医学图像数据集
    
    Args:
        dataset_name: 数据集名称，支持 "DDR" 或 "PMC-OA"
        root_dir: 数据集根目录
        split: 数据集分割，如 "train", "valid", "test"
        **kwargs: 传递给数据集的额外参数
        
    Returns:
        dataset: 数据集对象
    """
    if dataset_name.upper() == "DDR":
        from src.dataset.medical.ddr_dataset import DDRDataset
        dataset = DDRDataset(root_dir=root_dir, split=split, **kwargs)
    elif dataset_name.upper() == "PMC-OA" or dataset_name.upper() == "PMCOA":
        from src.dataset.medical.pmc_oa_dataset import PMCOADataset
        dataset = PMCOADataset(root_dir=root_dir, split=split)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}，支持的数据集: DDR, PMC-OA")
    
    return dataset


def batch_inference_from_dataset(
    model,
    processor_or_tokenizer,
    dataset,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    is_vl_model: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    从数据集中批量进行推理（支持Qwen-Med和Qwen-VL-Med）
    
    Args:
        model: Qwen模型
        processor_or_tokenizer: 处理器（VL模型）或分词器（纯文本模型）
        dataset: 数据集对象
        prompt: 推理提示文本
        batch_size: 批次大小
        max_samples: 最大处理样本数，如果为None则处理整个数据集
        is_vl_model: 是否为视觉语言模型（True为Qwen-VL，False为Qwen-Med）
        **kwargs: 传递给推理函数的其他参数
        
    Returns:
        List[Dict]: 推理结果列表，每个元素包含图像、真实标注和模型输出
    """
    dataloader = dataset.dataloader(batch_size=batch_size, shuffle=False)
    
    results = []
    processed = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"]
        captions = batch.get("caption", [None] * len(images))
        severities = batch.get("severity", [None] * len(images))
        
        # 处理批次中的每张图像
        for img_idx, img in enumerate(images):
            if max_samples is not None and processed >= max_samples:
                return results
            
            try:
                # 执行推理
                if is_vl_model:
                    output = qwen_vl_med_chat(
                        model=model,
                        processor=processor_or_tokenizer,
                        new_text=prompt,
                        new_image=img,
                        **kwargs,
                    )
                    # Qwen-VL可能返回列表，转换为字符串
                    if isinstance(output, list):
                        output = output[0] if len(output) > 0 else ""
                else:
                    # 纯文本模型不支持图像，跳过
                    print(f"警告: Qwen-Med是纯文本模型，不支持图像输入，跳过图像 {img_idx}")
                    continue
                
                result = {
                    "batch_idx": batch_idx,
                    "image_idx": img_idx,
                    "model_output": output,
                    "ground_truth_caption": captions[img_idx] if captions[img_idx] is not None else None,
                    "ground_truth_severity": severities[img_idx] if severities[img_idx] is not None else None,
                }
                results.append(result)
                processed += 1
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 图像 {img_idx} 时出错: {e}")
                continue
        
        if max_samples is not None and processed >= max_samples:
            break
    
    return results


def inference_dataset(
    dataset_name: str,
    root_dir: Union[str, Path],
    split: Literal["train", "valid", "test"] = "train",
    model_path: str = "AdaptLLM/biomed-Qwen2-VL-2B-Instruct",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    is_vl_model: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    完整的数据集推理流程（包含模型和数据集加载）
    
    Args:
        dataset_name: 数据集名称，支持 "DDR" 或 "PMC-OA"
        root_dir: 数据集根目录
        split: 数据集分割
        model_path: 模型路径或HuggingFace模型ID
        device: 设备
        cache_dir: 缓存目录
        prompt: 推理提示文本
        batch_size: 批次大小
        max_samples: 最大处理样本数
        is_vl_model: 是否为视觉语言模型（True为Qwen-VL，False为Qwen-Med）
        **kwargs: 传递给推理函数的其他参数
        
    Returns:
        List[Dict]: 推理结果列表
    """
    # 加载模型
    if is_vl_model:
        model, processor = load_qwen_vl_med(
            model_path=model_path,
            device=device,
            cache_dir=cache_dir,
        )
        processor_or_tokenizer = processor
    else:
        model, tokenizer = load_qwen_med(
            model_path=model_path,
            device=device,
            cache_dir=cache_dir,
        )
        processor_or_tokenizer = tokenizer
    
    # 加载数据集
    dataset = load_dataset(dataset_name=dataset_name, root_dir=root_dir, split=split)
    
    # 批量推理
    results = batch_inference_from_dataset(
        model=model,
        processor_or_tokenizer=processor_or_tokenizer,
        dataset=dataset,
        prompt=prompt,
        batch_size=batch_size,
        max_samples=max_samples,
        is_vl_model=is_vl_model,
        **kwargs,
    )
    
    return results


if __name__ == "__main__":
    # 示例用法
    import torch
    
    # 设置环境变量（如果需要）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_CACHE", "cache/models/pretrained")
    
    # 方式1: Qwen-Med纯文本推理
    # model, tokenizer = load_qwen_med(device="cuda:3")
    # output = qwen_med_chat(
    #     model, tokenizer,
    #     new_text="请解释一下糖尿病视网膜病变。"
    # )
    # print("模型输出：", output)
    
    # 方式2: Qwen-VL-Med视觉语言推理
    # model, processor = load_qwen_vl_med(device="cuda:3")
    # image = ...  # 你的图像
    # output = qwen_vl_med_chat(
    #     model, processor,
    #     new_text="请根据这张视网膜图片进行医学诊断。",
    #     new_image=image
    # )
    # print("模型输出：", output)
    
    # 方式3: 一键推理（包含模型加载）
    # output = qwen_vl_med_inference(
    #     prompt="请根据这张视网膜图片进行医学诊断。",
    #     image="path/to/image.jpg"
    # )
    # print("模型输出：", output)
    
    # 方式4: 从数据集批量推理
    # results = inference_dataset(
    #     dataset_name="PMC-OA",
    #     root_dir="/media/yons/Datasets/PMC-OA/",
    #     split="train",
    #     device="cuda:3",
    #     prompt="请根据这张视网膜图片进行医学诊断。",
    #     max_samples=10,  # 只处理前10个样本
    #     is_vl_model=True,  # 使用视觉语言模型
    # )
    # for result in results:
    #     print(f"模型输出: {result['model_output']}")
    #     print(f"真实标注: {result['ground_truth_caption']}")
    #     print("-" * 50)

