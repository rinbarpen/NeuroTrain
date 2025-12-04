from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence, Union, cast
from collections.abc import Sequence as SeqABC

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.config import get_config_value
from src.constants import PRETRAINED_MODEL_DIR
from src.utils.llm.chat_history import ChatHistory

logger = logging.getLogger(__name__)

PromptInput = Union[str, ChatHistory, Sequence[Dict[str, Any]], Sequence[Any]]


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray | torch.Tensor
    texts: list[str]
    pooling: Literal['cls', 'mean']
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    model_id: str
    llm_type: Literal['gpt', 'bert'] = 'gpt'
    cache_dir: str = PRETRAINED_MODEL_DIR
    device: str | torch.device | None = None
    dtype: str | torch.dtype | None = None
    max_length: int | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
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
        result = "" if value is None else str(value)
        return cast(str, result)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text_value = part.get("text")
                if text_value is not None:
                    pieces.append(str(text_value))
        return "\n".join(pieces)
    return str(content)


class LLMBase:
    """
    LLM通用基类，封装了模型与分词器的构建、设备管理以及输入规范化。
    子类只需关注具体的前向推理逻辑。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache_dir = str(config.cache_dir or PRETRAINED_MODEL_DIR)
        self.device = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.dtype)

        self.model_kwargs = dict(config.model_kwargs)
        self.model_kwargs.setdefault("torch_dtype", self.dtype)
        self.model_kwargs.setdefault("cache_dir", self.cache_dir)

        self.tokenizer_kwargs = dict(config.tokenizer_kwargs)
        self.tokenizer_kwargs.setdefault("cache_dir", self.cache_dir)

        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None

        self._load_components()
        logger.info("LLM %s 初始化完成（type=%s）", config.model_id, config.llm_type)

    def _load_components(self):
        model, tokenizer = self._build_model_and_tokenizer()
        if model is None or tokenizer is None:
            raise RuntimeError("模型或分词器加载失败")

        model_on_device = model.to(self.device)
        model_on_device.eval()
        self.model = model_on_device
        self.tokenizer = tokenizer

        tok = self._require_tokenizer()
        if getattr(tok, "pad_token", None) is None:
            pad_token = getattr(tok, "eos_token", None) or getattr(tok, "bos_token", None)
            if pad_token is None:
                raise ValueError("当前分词器缺少 pad_token / eos_token，无法正常推理")
            tok.pad_token = pad_token

        if self.config.max_length:
            tok.model_max_length = self.config.max_length

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("模型尚未初始化")
        return self.model

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("分词器尚未初始化")
        return self.tokenizer

    def _normalize_prompts(self, prompt: PromptInput) -> tuple[list[str], bool]:
        if isinstance(prompt, ChatHistory):
            return [self._chat_history_to_text(prompt.get_history())], False
        if isinstance(prompt, str):
            return [prompt], False
        if isinstance(prompt, Mapping):
            return [self._messages_to_text([prompt])], False
        if isinstance(prompt, SeqABC):
            normalized: list[str] = []
            for item in prompt:
                normalized.append(self._normalize_single_prompt(item))
            return normalized, True
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    def _normalize_single_prompt(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, ChatHistory):
            return self._chat_history_to_text(item.get_history())
        if isinstance(item, Mapping):
            return self._messages_to_text([item])
        if isinstance(item, Sequence):
            if item and isinstance(item[0], Mapping):
                return self._messages_to_text(item)  # type: ignore[arg-type]
            return " ".join(str(part) for part in item)
        return str(item)

    def _chat_history_to_text(self, history: Sequence[Mapping[str, Any]]) -> str:
        tokenizer = self._require_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    list(history),
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as exc:
                logger.debug("apply_chat_template 失败，使用降级格式: %s", exc)
        return self._messages_to_text(history)

    def _messages_to_text(self, messages: Sequence[Mapping[str, Any]]) -> str:
        buffer: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = _content_to_text(message.get("content", ""))
            buffer.append(f"{role.upper()}: {content}".strip())
        buffer.append("ASSISTANT:")
        return "\n".join(buffer)

    def to(self, device: str | torch.device):
        self.device = _resolve_device(device)
        model = self._require_model()
        model.to(self.device)
        return self

    def tokenize(
        self,
        texts: list[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self._require_tokenizer()
        tokenizer_kwargs: dict[str, Any] = {
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
        }
        effective_max_length = max_length or self.config.max_length
        if effective_max_length:
            tokenizer_kwargs["max_length"] = int(effective_max_length)
        tokenizer_kwargs.update(kwargs)
        encoded = tokenizer(texts, **tokenizer_kwargs)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def __call__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("请在子类中实现具体调用逻辑")

    def _build_model_and_tokenizer(self):
        raise NotImplementedError


class GPTLikeLLM(LLMBase):
    """面向 AutoModelForCausalLM 的 GPT 类模型包装"""

    DEFAULT_GENERATION = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.05,
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        tokenizer = self._require_tokenizer()
        tokenizer.padding_side = "left"

    def _build_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, **self.tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **self.model_kwargs)
        return model, tokenizer

    def _merge_generation_kwargs(self, runtime_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.DEFAULT_GENERATION)
        merged.update(self.config.generation_params)
        if runtime_kwargs:
            merged.update(runtime_kwargs)
        return merged

    @torch.no_grad()
    def generate(self, prompt: PromptInput, **gen_kwargs: Any) -> GenerationResult | list[GenerationResult]:
        prompts, is_batch = self._normalize_prompts(prompt)
        tokenized = self.tokenize(prompts)
        generation_kwargs = self._merge_generation_kwargs(gen_kwargs)

        model = cast(Any, self._require_model())
        tokenizer = self._require_tokenizer()

        outputs = model.generate(**tokenized, **generation_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        attention_mask = tokenized.get("attention_mask")
        prompt_lengths = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [tokenized["input_ids"].shape[-1]] * len(decoded)
        )

        results: list[GenerationResult] = []
        for text, output_ids, prompt_len in zip(decoded, outputs, prompt_lengths):
            completion_tokens = max(int(output_ids.shape[-1]) - int(prompt_len), 0)
            results.append(
                GenerationResult(
                    text=text.strip(),
                    raw_text=text,
                    prompt_tokens=int(prompt_len),
                    completion_tokens=completion_tokens,
                    total_tokens=int(output_ids.shape[-1]),
                )
            )
        return results if is_batch else results[0]

    __call__ = generate


class BERTLikeLLM(LLMBase):
    """面向 AutoModel（编码器）的 BERT 类模型包装"""

    def __init__(self, config: LLMConfig, *, pooling: Literal['cls', 'mean'] = 'cls'):
        self.pooling: Literal['cls', 'mean'] = pooling
        super().__init__(config)

    def _build_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, **self.tokenizer_kwargs)
        model = AutoModel.from_pretrained(self.config.model_id, **self.model_kwargs)
        return model, tokenizer

    def forward(self, texts: PromptInput, **kwargs: Any):
        if isinstance(texts, str):
            text_list = [texts]
        elif isinstance(texts, ChatHistory):
            text_list = [self._chat_history_to_text(texts.get_history())]
        elif isinstance(texts, SeqABC):
            text_list = [self._normalize_single_prompt(item) for item in texts]
        else:
            raise TypeError(f"Unsupported input type: {type(texts)}")
        tokenized = self.tokenize(text_list, **kwargs)
        model = cast(Any, self._require_model())
        outputs = model(**tokenized, return_dict=True)
        return outputs, tokenized

    @torch.no_grad()
    def encode(
        self,
        texts: PromptInput,
        *,
        pooling: Literal['cls', 'mean'] | None = None,
        normalize: bool = True,
        return_tensors: bool = False,
        **tokenizer_kwargs: Any,
    ) -> EmbeddingResult:
        outputs, tokenized = self.forward(texts, **tokenizer_kwargs)
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise RuntimeError("BERT 模型未返回 last_hidden_state，无法计算向量")

        pooling_choice = pooling or self.pooling
        pooling_choice = cast(Literal['cls', 'mean'], pooling_choice)
        if pooling_choice == 'cls':
            embeddings = hidden_states[:, 0]
        elif pooling_choice == 'mean':
            mask = tokenized["attention_mask"].unsqueeze(-1)
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling_choice}")

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        output_embeddings: torch.Tensor | np.ndarray
        if return_tensors:
            output_embeddings = embeddings
        else:
            output_embeddings = embeddings.detach().cpu().numpy()

        if isinstance(texts, str):
            text_list = [texts]
        elif isinstance(texts, ChatHistory):
            text_list = [self._chat_history_to_text(texts.get_history())]
        elif isinstance(texts, SeqABC):
            text_list = [self._normalize_single_prompt(item) for item in texts]
        else:
            text_list = [str(texts)]

        return EmbeddingResult(
            embeddings=output_embeddings,
            texts=text_list,
            pooling=pooling_choice,
        )

    __call__ = encode


def build_llm(config: LLMConfig | dict[str, Any]) -> LLMBase:
    """根据配置创建 GPT-like / BERT-like 模型"""
    if isinstance(config, dict):
        config = LLMConfig(**config)
    llm_type = config.llm_type.lower()
    if llm_type in ("gpt", "decoder", "causal"):
        config.llm_type = "gpt"
        return GPTLikeLLM(config)
    if llm_type in ("bert", "encoder", "masked"):
        config.llm_type = "bert"
        return BERTLikeLLM(config)
    raise ValueError(f"Unsupported llm_type: {config.llm_type}")


__all__ = [
    "LLMConfig",
    "GenerationResult",
    "EmbeddingResult",
    "LLMBase",
    "GPTLikeLLM",
    "BERTLikeLLM",
    "build_llm",
]

