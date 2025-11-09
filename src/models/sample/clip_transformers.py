from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from src.constants import PRETRAINED_MODEL_DIR
from src.utils.tensor import move_to_device

class TransformersCLIP(nn.Module):
    """
    基于 HuggingFace transformers 的 CLIP 包装。

    - 支持文本 / 图像编码
    - 支持返回字典以方便训练器消费
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | None = None,
        torch_dtype: torch.dtype | None = torch.float16,
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ) -> None:
        super().__init__()
        cache_dir = cache_dir or PRETRAINED_MODEL_DIR

        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        clip = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )
        self.clip: CLIPModel = cast(CLIPModel, clip)
        nn.Module.to(self.clip, device)

        if freeze_vision:
            for param in self.clip.vision_model.parameters():
                param.requires_grad = False
        if freeze_text:
            for param in self.clip.text_model.parameters():
                param.requires_grad = False

        self.device = torch.device(device)

    def encode_image(self, images: Sequence | torch.Tensor) -> torch.Tensor:
        processor = cast(Any, self.processor)
        inputs = processor(
            images=images,
            return_tensors="pt",
        )
        tensor_inputs = cast(dict[str, torch.Tensor], move_to_device(inputs, self.device))
        pixel_values = cast(torch.Tensor, tensor_inputs["pixel_values"])
        outputs = self.clip.get_image_features(pixel_values=pixel_values)  # type: ignore[arg-type]
        return F.normalize(outputs, dim=-1)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        processor = cast(Any, self.processor)
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tensor_inputs = cast(dict[str, torch.Tensor], move_to_device(inputs, self.device))
        outputs = self.clip.get_text_features(
            input_ids=tensor_inputs["input_ids"],
            attention_mask=tensor_inputs.get("attention_mask"),
        )
        return F.normalize(outputs, dim=-1)

    def forward(
        self,
        *,
        images: Sequence | torch.Tensor | None = None,
        texts: Sequence[str] | None = None,
        return_loss: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        if images is None and texts is None:
            raise ValueError("必须至少提供 images 或 texts 其中之一")

        processor = cast(Any, self.processor)
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tensor_inputs = cast(dict[str, torch.Tensor], move_to_device(inputs, self.device))

        outputs = self.clip(
            input_ids=tensor_inputs.get("input_ids"),
            attention_mask=tensor_inputs.get("attention_mask"),
            pixel_values=tensor_inputs.get("pixel_values"),
            return_dict=True,
        )
        result: dict[str, torch.Tensor] = {}

        logits_per_image = getattr(outputs, "logits_per_image", None)
        logits_per_text = getattr(outputs, "logits_per_text", None)

        if logits_per_image is not None and logits_per_text is not None and texts is not None and images is not None:
            targets = torch.arange(
                logits_per_image.shape[0],
                device=logits_per_image.device,
            )
            loss = (
                F.cross_entropy(logits_per_image, targets)
                + F.cross_entropy(logits_per_text, targets)
            ) / 2.0 if return_loss else torch.tensor(0.0, device=logits_per_image.device)

            result["loss"] = loss
            result["preds"] = logits_per_image
            result["targets"] = targets
        elif return_loss:
            result["loss"] = torch.tensor(0.0, device=self.device)

        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        if image_embeds is not None:
            result["image_embeds"] = image_embeds
        if text_embeds is not None:
            result["text_embeds"] = text_embeds
        return result

