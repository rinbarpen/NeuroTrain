"""
LLaVA模型推理模块
提供LLaVA-Med模型的加载和推理功能，支持数据集批量推理
"""

import torch
from typing import Optional, List, Union, Dict, Any, Literal
from PIL import Image
import os
from pathlib import Path

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)


def load_llava_med(
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
):
    """
    加载LLaVA-Med模型
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备，如 "cuda:3" 或 "cpu"
        cache_dir: 缓存目录，如果为None则使用环境变量HF_HUB_CACHE
        
    Returns:
        tuple: (model, tokenizer, image_processor)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE", "cache/models/pretrained")
    
    device_name = device.split(":")[0] if ":" in device else device
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        device=device_name,
    )
    
    return model, tokenizer, image_processor


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


def llava_med_chat(
    model,
    tokenizer,
    processor,
    new_text: str,
    new_image: Union[torch.Tensor, Image.Image],
    history: Optional[List] = None,
    conv_mode: str = "mistral_instruct",
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    use_cache: bool = True,
) -> str:
    """
    LLaVA-Med模型对话推理
    
    Args:
        model: LLaVA模型
        tokenizer: 分词器
        processor: 图像处理器
        new_text: 新的文本输入
        new_image: 输入图像（可以是torch.Tensor或PIL.Image）
        history: 对话历史（可选）
        conv_mode: 对话模板模式，默认为"mistral_instruct"
        max_new_tokens: 最大生成token数
        do_sample: 是否使用采样
        use_cache: 是否使用缓存
        
    Returns:
        str: 模型生成的回复文本
    """
    if history is None:
        history = []
    
    # 使用对话模板
    conv = conv_templates[conv_mode].copy()
    
    # 处理图像
    new_image = preprocess_image(new_image)
    
    # 处理图像张量
    image_tensor = process_images([new_image], processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(
            [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        )
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 构建prompt，添加图像token
    if model.config.mm_use_im_start_end:
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + new_text
        )
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + new_text
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids_tensor = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    if isinstance(input_ids_tensor, torch.Tensor):
        input_ids = input_ids_tensor.unsqueeze(0).to(model.device)
    else:
        # 如果是list，转换为tensor
        input_ids = torch.tensor(input_ids_tensor).unsqueeze(0).to(model.device)
    
    # 设置停止条件
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # 创建attention_mask
    attention_mask = torch.ones_like(input_ids)
    
    # 生成回复
    # 注意：LLaVA-Mistral的generate方法使用inputs作为第一个位置参数
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,  # 位置参数，会被映射到inputs
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            stopping_criteria=[stopping_criteria],
        )
    
    # 处理GenerateOutput对象（如果返回的是GenerateOutput）
    if hasattr(output_ids, 'sequences'):
        output_ids = output_ids.sequences
    
    # 确保output_ids是tensor
    if not isinstance(output_ids, torch.Tensor):
        output_ids = torch.tensor(output_ids)
    
    # 确保output_ids至少是2D的
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    
    # 提取新生成的token（标准LLaVA做法）
    # output_ids包含输入+输出，提取新生成的部分
    if output_ids.shape[1] > input_ids.shape[1]:
        new_token_ids = output_ids[0, input_ids.shape[1]:]
        num_generated_tokens = new_token_ids.shape[0]
        
        # 检查是否达到了最大token数（可能被截断）
        if num_generated_tokens >= max_new_tokens:
            print(f"提示: 生成了 {num_generated_tokens} 个token（达到最大限制 {max_new_tokens}），输出可能不完整")
        
        # 解码新生成的token（完整输出，不截断）
        outputs = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return outputs
    else:
        # 如果output_ids长度 <= input_ids长度，说明生成失败
        print(
            f"警告: 生成失败。input_ids长度: {input_ids.shape[1]}, output_ids长度: {output_ids.shape[1]}"
        )
        # 尝试解码整个output_ids看看是什么（完整输出，不截断）
        try:
            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"完整output_ids解码: {full_output}")
        except Exception as e:
            print(f"解码失败: {e}")
        return ""


def llava_med_inference(
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    image: Optional[Union[torch.Tensor, Image.Image, str]] = None,
    **kwargs,
) -> str:
    """
    完整的LLaVA-Med推理流程（包含模型加载）
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        device: 设备
        cache_dir: 缓存目录
        prompt: 输入提示文本
        image: 输入图像（可以是torch.Tensor、PIL.Image或图像路径）
        **kwargs: 传递给llava_med_chat的其他参数
        
    Returns:
        str: 模型生成的回复文本
    """
    # 加载模型
    model, tokenizer, processor = load_llava_med(
        model_path=model_path,
        device=device,
        cache_dir=cache_dir,
    )
    
    # 检查图像是否提供
    if image is None:
        raise ValueError("必须提供图像参数（可以是torch.Tensor、PIL.Image或图像路径）")
    
    # 如果image是路径字符串，加载图像
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # 执行推理
    output = llava_med_chat(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        new_text=prompt,
        new_image=image,
        **kwargs,
    )
    
    return output


def load_dataset(
    dataset_name: str, 
    root_dir: Union[str, Path], 
    split: Literal["train", "valid", "test"] = "train", 
    **kwargs
):
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
    tokenizer,
    processor,
    dataset,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    从数据集中批量进行推理
    
    Args:
        model: LLaVA模型
        tokenizer: 分词器
        processor: 图像处理器
        dataset: 数据集对象
        prompt: 推理提示文本
        batch_size: 批次大小（注意：LLaVA推理通常是单张图像）
        max_samples: 最大处理样本数，如果为None则处理整个数据集
        **kwargs: 传递给llava_med_chat的其他参数
        
    Returns:
        List[Dict]: 推理结果列表，每个元素包含图像、真实标注和模型输出
    """
    dataloader = dataset.dataloader(batch_size=batch_size, shuffle=False)
    
    results = []
    processed = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"]
        # 安全地获取captions和severities
        captions = batch.get("caption", None)
        if captions is None:
            captions = [None] * len(images)
        elif not isinstance(captions, list):
            captions = [captions] if len(images) == 1 else [None] * len(images)
        
        severities = batch.get("severity", None)
        if severities is None:
            severities = [None] * len(images)
        elif not isinstance(severities, list):
            severities = [severities] if len(images) == 1 else [None] * len(images)
        
        # 处理批次中的每张图像
        for img_idx, img in enumerate(images):
            if max_samples is not None and processed >= max_samples:
                return results
            
            try:
                # 执行推理
                output = llava_med_chat(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    new_text=prompt,
                    new_image=img,
                    **kwargs,
                )
                
                # 安全地获取对应的caption和severity
                caption = captions[img_idx] if img_idx < len(captions) else None
                severity = severities[img_idx] if img_idx < len(severities) else None
                
                result = {
                    "batch_idx": batch_idx,
                    "image_idx": img_idx,
                    "model_output": output,
                    "ground_truth_caption": caption,
                    "ground_truth_severity": severity,
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
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    device: str = "cuda:3",
    cache_dir: Optional[str] = None,
    prompt: str = "请根据这张视网膜图片进行医学诊断。",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
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
        **kwargs: 传递给llava_med_chat的其他参数
        
    Returns:
        List[Dict]: 推理结果列表
    """
    # 加载模型
    model, tokenizer, processor = load_llava_med(
        model_path=model_path,
        device=device,
        cache_dir=cache_dir,
    )
    
    # 加载数据集
    dataset = load_dataset(dataset_name=dataset_name, root_dir=root_dir, split=split)
    
    # 批量推理
    results = batch_inference_from_dataset(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        dataset=dataset,
        prompt=prompt,
        batch_size=batch_size,
        max_samples=max_samples,
        **kwargs,
    )
    
    return results


if __name__ == "__main__":
    # 示例用法
    import torch
    
    # 设置环境变量（如果需要）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_CACHE", "cache/models/pretrained")
    
    # 方式1: 分步加载和推理
    # model, tokenizer, processor = load_llava_med(device="cuda:3")
    # image = ...  # 你的图像
    # output = llava_med_chat(
    #     model, tokenizer, processor,
    #     new_text="请根据这张视网膜图片进行医学诊断。",
    #     new_image=image
    # )
    # print("模型诊断输出：", output)
    
    # 方式2: 一键推理（包含模型加载）
    # output = llava_med_inference(
    #     prompt="请根据这张视网膜图片进行医学诊断。",
    #     image="path/to/image.jpg"
    # )
    # print("模型诊断输出：", output)
    
    # 方式3: 从数据集批量推理
    results = inference_dataset(
        dataset_name="DDR",
        root_dir="/media/yons/Datasets/OIA-DDR/DDR-dataset/",
        split="train",
        device="cuda:3",
        prompt="请根据这张视网膜图片进行医学诊断。",
        max_samples=10,  # 只处理前10个样本
    )
    for result in results:
        print(f"模型输出: {result['model_output']}")
        print(f"真实标注: {result['ground_truth_caption']}")
        print("-" * 50)

