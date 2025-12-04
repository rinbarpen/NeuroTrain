from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence, Union, cast
from collections.abc import Sequence as SeqABC

import torch
import cv2
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor

from src.config import get_config_value
from src.constants import PRETRAINED_MODEL_DIR
from src.utils.llm.chat_history import ChatHistory
from src.utils.llm.llm_utils import image_to_PIL

logger = logging.getLogger(__name__)

TextInput = Union[str, ChatHistory, Mapping[str, Any], Sequence[Any]]
ImageInput = Union[str, Path, Image.Image, cv2.Mat]


@dataclass
class VLMGenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMEncodingResult:
    image_embeds: torch.Tensor | None = None
    text_embeds: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMConfig:
    model_id: str
    vlm_type: Literal['chat', 'encoder'] = 'chat'
    cache_dir: str = PRETRAINED_MODEL_DIR
    device: str | torch.device | None = None
    dtype: str | torch.dtype | None = None
    max_length: int | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _default_device_name() -> str:
    configured = get_config_value("device", default=None)
    if isinstance(configured, str) and configured != "auto":
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        device = _default_device_name()
    return torch.device(device)


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        value = content.get("text")
        return "" if value is None else str(value)
    if isinstance(content, SeqABC) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text_value = item.get("text")
                if text_value is not None:
                    parts.append(str(text_value))
        return "\n".join(parts)
    return str(content)


class VLMBase:
    """
    通用 VLM 包装器，负责加载模型/处理器、输入归一化以及设备管理。
    子类只需实现具体任务逻辑。
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self.cache_dir = str(config.cache_dir or PRETRAINED_MODEL_DIR)
        self.device = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.dtype)

        self.model_kwargs = dict(config.model_kwargs)
        self.model_kwargs.setdefault("torch_dtype", self.dtype)
        self.model_kwargs.setdefault("cache_dir", self.cache_dir)

        self.processor_kwargs = dict(config.processor_kwargs)
        self.processor_kwargs.setdefault("cache_dir", self.cache_dir)

        self.model: torch.nn.Module | None = None
        self.processor: Any | None = None

        self._load_components()
        logger.info("VLM %s 初始化完成（type=%s）", config.model_id, config.vlm_type)

    def _load_components(self):
        model, processor = self._build_model_and_processor()
        if model is None or processor is None:
            raise RuntimeError("模型或处理器加载失败")

        model_on_device = model.to(self.device)
        model_on_device.eval()
        self.model = model_on_device
        self.processor = processor

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("模型尚未初始化")
        return self.model

    def _require_processor(self):
        if self.processor is None:
            raise RuntimeError("处理器尚未初始化")
        return self.processor

    def _normalize_texts(self, prompts: TextInput | Sequence[TextInput] | None) -> list[str] | None:
        if prompts is None:
            return None
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, ChatHistory):
            return [self._history_to_text(prompts.get_history())]
        if isinstance(prompts, Mapping):
            return [self._messages_to_text([prompts])]
        if isinstance(prompts, SeqABC) and not isinstance(prompts, (str, bytes, bytearray)):
            if len(prompts) == 0:
                return []
            if isinstance(prompts[0], Mapping):
                return [self._messages_to_text(prompts)]  # type: ignore[arg-type]
            normalized: list[str] = []
            for item in prompts:
                normalized.append(self._normalize_single_prompt(item))
            return normalized
        return [str(prompts)]

    def _normalize_single_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, ChatHistory):
            return self._history_to_text(prompt.get_history())
        if isinstance(prompt, Mapping):
            return self._messages_to_text([prompt])
        if isinstance(prompt, SeqABC) and not isinstance(prompt, (str, bytes, bytearray)):
            if prompt and isinstance(prompt[0], Mapping):
                return self._messages_to_text(prompt)  # type: ignore[arg-type]
            return " ".join(str(item) for item in prompt)
        return str(prompt)

    def _history_to_text(self, history: Sequence[Mapping[str, Any]]) -> str:
        processor = self._require_processor()
        if hasattr(processor, "apply_chat_template"):
            try:
                return processor.apply_chat_template(
                    list(history),
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as exc:
                logger.debug("apply_chat_template 失败，使用回退模板: %s", exc)
        return self._messages_to_text(history)

    def _messages_to_text(self, messages: Sequence[Mapping[str, Any]]) -> str:
        buffer: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = _content_to_text(message.get("content", ""))
            buffer.append(f"{role.upper()}: {content}".strip())
        buffer.append("ASSISTANT:")
        return "\n".join(buffer)

    def _normalize_images(self, images: ImageInput | Sequence[ImageInput] | None) -> list[Image.Image] | None:
        if images is None:
            return None
        if isinstance(images, (str, Path, Image.Image)):
            return [image_to_PIL(images)]
        if isinstance(images, SeqABC) and not isinstance(images, (str, bytes, bytearray)):
            normalized: list[Image.Image] = []
            for item in images:
                normalized.append(image_to_PIL(item))
            return normalized
        return [image_to_PIL(images)]

    def _align_texts_images(
        self,
        texts: list[str] | None,
        images: list[Image.Image] | None,
    ) -> tuple[list[str] | None, list[Image.Image] | None]:
        if texts is None:
            return None, images
        if not images or len(images) == 0:
            return texts, None
        if len(images) == len(texts):
            return texts, images
        if len(images) == 1 and len(texts) > 1:
            return texts, images * len(texts)
        if len(texts) == 1 and len(images) > 1:
            return texts * len(images), images
        raise ValueError("文本与图片数量不匹配")

    def prepare_inputs(
        self,
        *,
        texts: TextInput | Sequence[TextInput] | None = None,
        images: ImageInput | Sequence[ImageInput] | None = None,
        return_tensors: str = "pt",
        max_length: int | None = None,
        **processor_kwargs: Any,
    ):
        processor = self._require_processor()
        normalized_texts = self._normalize_texts(texts)
        normalized_images = self._normalize_images(images)
        aligned_texts, aligned_images = self._align_texts_images(normalized_texts, normalized_images)

        payload: dict[str, Any] = {"return_tensors": return_tensors}
        if aligned_texts is not None:
            payload["text"] = aligned_texts
        if aligned_images is not None:
            payload["images"] = aligned_images

        if not payload.get("text") and not payload.get("images"):
            raise ValueError("至少需要提供文本或图像")

        effective_max_length = max_length or self.config.max_length
        if effective_max_length:
            processor_kwargs.setdefault("max_length", int(effective_max_length))

        payload.update(processor_kwargs)
        inputs = processor(**payload)
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def to(self, device: str | torch.device):
        self.device = _resolve_device(device)
        model = self._require_model()
        model.to(self.device)
        return self

    def __call__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _build_model_and_processor(self):
        raise NotImplementedError


class VLMChatModel(VLMBase):
    """面向 AutoModelForCausalLM 的多模态生成模型"""

    DEFAULT_GENERATION = {
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
    }

    def _build_model_and_processor(self):
        processor = AutoProcessor.from_pretrained(self.config.model_id, **self.processor_kwargs)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **self.model_kwargs)
        return model, processor

    def _merge_generation_kwargs(self, runtime_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.DEFAULT_GENERATION)
        merged.update(self.config.generation_params)
        if runtime_kwargs:
            merged.update(runtime_kwargs)
        return merged

    @torch.no_grad()
    def generate(
        self,
        prompt: TextInput | Sequence[TextInput],
        *,
        images: ImageInput | Sequence[ImageInput] | None = None,
        **gen_kwargs: Any,
    ) -> VLMGenerationResult | list[VLMGenerationResult]:
        normalized_texts = self._normalize_texts(prompt) or [""]
        is_batch = len(normalized_texts) > 1
        inputs = self.prepare_inputs(texts=normalized_texts, images=images)

        model = cast(Any, self._require_model())
        processor = self._require_processor()
        generation_kwargs = self._merge_generation_kwargs(gen_kwargs)
        outputs = model.generate(**inputs, **generation_kwargs)
        decoded = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        attention_mask = inputs.get("attention_mask")
        prompt_tokens = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [inputs["input_ids"].shape[-1]] * len(decoded)
        )

        results: list[VLMGenerationResult] = []
        for text, output_ids, prompt_len in zip(decoded, outputs, prompt_tokens):
            completion_tokens = max(int(output_ids.shape[-1]) - int(prompt_len), 0)
            results.append(
                VLMGenerationResult(
                    text=text.strip(),
                    prompt_tokens=int(prompt_len),
                    completion_tokens=completion_tokens,
                    total_tokens=int(output_ids.shape[-1]),
                )
            )
        return results if is_batch else results[0]

    __call__ = generate


class VLMEncoderModel(VLMBase):
    """面向 AutoModel 的多模态编码器（如 CLIP/OpenCLIP 等）"""

    def _build_model_and_processor(self):
        processor = AutoProcessor.from_pretrained(self.config.model_id, **self.processor_kwargs)
        model = AutoModel.from_pretrained(self.config.model_id, **self.model_kwargs)
        return model, processor

    @torch.no_grad()
    def encode(
        self,
        *,
        texts: TextInput | Sequence[TextInput] | None = None,
        images: ImageInput | Sequence[ImageInput] | None = None,
        normalize: bool = True,
        **processor_kwargs: Any,
    ) -> VLMEncodingResult:
        if texts is None and images is None:
            raise ValueError("编码模式至少需要文本或图像")
        inputs = self.prepare_inputs(texts=texts, images=images, **processor_kwargs)
        model = cast(Any, self._require_model())
        outputs = model(**inputs, return_dict=True)

        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        logits = getattr(outputs, "logits_per_image", None) or getattr(outputs, "logits", None)

        if normalize:
            if isinstance(image_embeds, torch.Tensor):
                image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
            if isinstance(text_embeds, torch.Tensor):
                text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        return VLMEncodingResult(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logits=logits,
        )

    __call__ = encode


def build_vlm(config: VLMConfig | Dict[str, Any]) -> VLMBase:
    """
    根据配置创建通用 VLM 模型。
    vlm_type = 'chat' -> VLMChatModel
    vlm_type = 'encoder' -> VLMEncoderModel
    """
    if isinstance(config, dict):
        config = VLMConfig(**config)
    vlm_type = config.vlm_type.lower()
    if vlm_type == "chat":
        config.vlm_type = "chat"
        return VLMChatModel(config)
    if vlm_type == "encoder":
        config.vlm_type = "encoder"
        return VLMEncoderModel(config)
    raise ValueError(f"Unsupported vlm_type: {config.vlm_type}")


__all__ = [
    "VLMConfig",
    "VLMGenerationResult",
    "VLMEncodingResult",
    "VLMBase",
    "VLMChatModel",
    "VLMEncoderModel",
    "build_vlm",
]

