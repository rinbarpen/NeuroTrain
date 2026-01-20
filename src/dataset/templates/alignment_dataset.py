from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import torch
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision import transforms as mtf
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..custom_dataset import CustomDataset
from src.constants import PRETRAINED_MODEL_DIR
from src.utils.annotation import unimplemented

logger = logging.getLogger(__name__)

_TV_RESIZE_BICUBIC = InterpolationMode.BICUBIC
_PIL_RESIZE_BILINEAR = Resampling.BILINEAR

@dataclass
class ImageTextExample:
    """基础的图文对齐样本."""

    image_path: Path
    texts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionDescription:
    """区域级别的描述信息."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str
    category_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionAlignmentExample(ImageTextExample):
    """包含区域列表的对齐样本."""

    regions: List[RegionDescription] = field(default_factory=list)


def default_clip_transform(image_size: int = 224, center_crop: bool = True) -> mtf.Compose:
    """返回与CLIP兼容的默认图像预处理."""

    transform_steps: List[Any] = [
        mtf.Resize(image_size, interpolation=_TV_RESIZE_BICUBIC),
    ]
    if center_crop:
        transform_steps.append(mtf.CenterCrop(image_size))
    transform_steps.extend(
        [
            mtf.ToTensor(),
            mtf.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )
    return mtf.Compose(transform_steps)


class AlignmentDatasetBase(CustomDataset):
    """图文/区域对齐数据集模板基类."""

    example_cls = ImageTextExample
    default_tokenizer_name = "openai/clip-vit-base-patch14"
    default_image_size = 224

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str,
        *,
        manifest_path: Union[str, Path, None] = None,
        image_root: Union[str, Path, None] = None,
        image_transform: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        max_texts_per_image: int = 1,
        manifest_image_key: str = "image",
        manifest_text_key: Union[str, Sequence[str], None] = "text",
        text_delimiter: Optional[str] = "||",
        allow_empty_texts: bool = False,
        **kwargs,
    ):
        super().__init__(root_dir, split, **kwargs)

        self.root_dir = Path(root_dir)
        self.image_root = Path(image_root) if image_root else self.root_dir
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.manifest_image_key = manifest_image_key
        self.manifest_text_key = manifest_text_key
        self.text_delimiter = text_delimiter
        self.allow_empty_texts = allow_empty_texts
        self.max_texts_per_image = max(1, int(max_texts_per_image))
        self.image_transform = image_transform or default_clip_transform(self.default_image_size)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            model_name = tokenizer_name or self.default_tokenizer_name
            tk_kwargs = {"cache_dir": PRETRAINED_MODEL_DIR, "use_fast": True}
            if tokenizer_kwargs:
                tk_kwargs.update(tokenizer_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tk_kwargs)

        if self.tokenizer.pad_token is None:
            pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token or "<pad>"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        pad_token_value = self.tokenizer.pad_token_id
        if not isinstance(pad_token_value, int):
            pad_token_value = 0
        self.pad_token_id: int = pad_token_value

        if not self._cache_loaded:
            self.samples = self._load_samples_entry()
            self.n = len(self.samples)
            self._save_to_cache_if_needed()
        else:
            self.samples = [self._coerce_example(sample) for sample in self.samples]
            self.n = len(self.samples)

    # ------------------------------------------------------------------
    # 数据加载逻辑
    # ------------------------------------------------------------------
    def _load_samples_entry(self) -> List[ImageTextExample]:
        if self.manifest_path is not None:
            logger.info("Loading alignment manifest: %s", self.manifest_path)
            return self._load_samples_from_manifest(self.manifest_path)
        return self.load_samples()

    @unimplemented
    def load_samples(self) -> List[ImageTextExample]:  # pragma: no cover - 由子类实现
        raise NotImplementedError("子类需要实现 load_samples 或提供 manifest_path")

    def _load_samples_from_manifest(self, manifest_path: Path) -> List[ImageTextExample]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"未找到 manifest 文件: {manifest_path}")

        records = self._read_manifest_records(manifest_path)
        examples: List[ImageTextExample] = []
        for record in records:
            example = self._manifest_record_to_example(record)
            if example is not None:
                examples.append(example)

        if not examples:
            logger.warning("Manifest %s 未解析出有效样本", manifest_path)
        return examples

    def _manifest_record_to_example(self, record: Dict[str, Any]) -> Optional[ImageTextExample]:
        image_path = self._extract_image_path(record)
        if image_path is None:
            return None
        texts = self._extract_texts(record)
        if not texts and not self.allow_empty_texts:
            return None
        metadata = self._extract_metadata(record)
        return self.example_cls(image_path=image_path, texts=texts or [""], metadata=metadata)

    # ------------------------------------------------------------------
    # Manifest 工具
    # ------------------------------------------------------------------
    def _read_manifest_records(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            if isinstance(data, dict):
                # 兼容 {image: text} 形式
                return [
                    {self.manifest_image_key: key, self._text_key_fallback(): value}
                    for key, value in data.items()
                ]
            raise ValueError(f"无法解析的 JSON manifest 格式: {path}")

        if suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)

        raise ValueError(f"不支持的 manifest 扩展名: {path}")

    def _text_key_fallback(self) -> str:
        keys = self._manifest_text_keys
        if keys:
            return keys[0]
        return "text"

    @property
    def _manifest_text_keys(self) -> List[str]:
        if self.manifest_text_key is None:
            return []
        if isinstance(self.manifest_text_key, str):
            return [self.manifest_text_key]
        return [k for k in self.manifest_text_key if k]

    def _extract_image_path(self, record: Dict[str, Any]) -> Optional[Path]:
        value = record.get(self.manifest_image_key)
        if value is None:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = (self.manifest_path.parent if self.manifest_path else self.image_root) / path
        return path

    def _extract_texts(self, record: Dict[str, Any]) -> List[str]:
        keys = self._manifest_text_keys
        texts: List[str] = []
        if not keys:
            raw = record.get("text")
            texts.extend(self._normalize_text_field(raw))
        else:
            for key in keys:
                raw = record.get(key)
                texts.extend(self._normalize_text_field(raw))
        filtered = [t for t in texts if t or self.allow_empty_texts]
        if not filtered and self.allow_empty_texts:
            return [""]
        return filtered

    def _extract_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {self.manifest_image_key, *self._manifest_text_keys}
        return {k: v for k, v in record.items() if k not in reserved}

    def _normalize_text_field(self, raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            if self.text_delimiter and self.text_delimiter in raw:
                parts = [seg.strip() for seg in raw.split(self.text_delimiter)]
                return [p for p in parts if p or self.allow_empty_texts]
            return [raw.strip()]
        if isinstance(raw, (list, tuple, set)):
            texts: List[str] = []
            for val in raw:
                texts.extend(self._normalize_text_field(val))
            return texts
        return [str(raw)]

    # ------------------------------------------------------------------
    # 样本/图像处理
    # ------------------------------------------------------------------
    def _coerce_example(self, sample: Any) -> ImageTextExample:
        if isinstance(sample, self.example_cls):
            return sample
        if isinstance(sample, dict):
            image_value = sample.get("image_path")
            if image_value is None:
                raise KeyError("缓存样本缺少 image_path")
            data: Dict[str, Any] = {
                "image_path": Path(str(image_value)),
                "texts": list(sample.get("texts", [])),
                "metadata": sample.get("metadata", {}),
            }
            if isinstance(self.example_cls, type) and issubclass(self.example_cls, RegionAlignmentExample):
                regions_raw = sample.get("regions", [])
                if isinstance(regions_raw, list) and regions_raw and isinstance(regions_raw[0], RegionDescription):
                    data["regions"] = regions_raw
                elif hasattr(self, "_normalize_region_list"):
                    data["regions"] = getattr(self, "_normalize_region_list")(regions_raw)
                else:
                    data["regions"] = []
            return self.example_cls(**data)
        raise TypeError(f"无法还原缓存样本: {type(sample)}")

    def _load_image(self, image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _apply_image_transform(self, image: Image.Image) -> Tensor:
        tensor = self.image_transform(image) if self.image_transform else mtf.ToTensor()(image)
        if isinstance(tensor, torch.Tensor):
            return tensor
        return torch.as_tensor(tensor)

    def _tokenize_texts(self, texts: List[str]):
        if not texts:
            texts = [""]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


class ClipAlignmentDatasetTemplate(AlignmentDatasetBase):
    """CLIP风格的图文对齐数据集模板."""

    example_cls = ImageTextExample

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if not isinstance(sample, ImageTextExample):
            sample = self._coerce_example(sample)

        image = self._load_image(sample.image_path)
        image_tensor = self._apply_image_transform(image)

        texts = sample.texts[: self.max_texts_per_image] or [""]
        tokenized = self._tokenize_texts(texts)

        item = {
            "image": image_tensor,
            "texts": texts,
            "text_ids": tokenized["input_ids"],
            "metadata": sample.metadata,
            "image_path": str(sample.image_path),
        }
        if "attention_mask" in tokenized:
            item["text_attn_mask"] = tokenized["attention_mask"]
        if "token_type_ids" in tokenized:
            item["text_token_type_ids"] = tokenized["token_type_ids"]
        return item

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    def get_collate_fn(self):
        pad_value = self.pad_token_id

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            images = torch.stack([item["image"] for item in batch], dim=0)
            text_tensors = [item["text_ids"] for item in batch]

            max_texts = max(t.shape[0] for t in text_tensors)
            max_seq = max(t.shape[1] for t in text_tensors)
            text_ids = torch.full(
                (len(batch), max_texts, max_seq),
                fill_value=pad_value,
                dtype=text_tensors[0].dtype,
            )
            for i, tensor in enumerate(text_tensors):
                n_texts, seq_len = tensor.shape
                text_ids[i, :n_texts, :seq_len] = tensor

            collated: Dict[str, Any] = {
                "images": images,
                "text_ids": text_ids,
                "texts": [item["texts"] for item in batch],
                "metadata": [item["metadata"] for item in batch],
                "image_paths": [item["image_path"] for item in batch],
            }

            if "text_attn_mask" in batch[0]:
                attn_tensors = [item["text_attn_mask"] for item in batch if "text_attn_mask" in item]
                if attn_tensors:
                    attn = torch.zeros((len(batch), max_texts, max_seq), dtype=attn_tensors[0].dtype)
                    for i, tensor in enumerate(attn_tensors):
                        n_texts, seq_len = tensor.shape
                        attn[i, :n_texts, :seq_len] = tensor
                    collated["text_attn_mask"] = attn

            if "text_token_type_ids" in batch[0]:
                type_tensors = [item["text_token_type_ids"] for item in batch if "text_token_type_ids" in item]
                if type_tensors:
                    token_type = torch.zeros((len(batch), max_texts, max_seq), dtype=type_tensors[0].dtype)
                    for i, tensor in enumerate(type_tensors):
                        n_texts, seq_len = tensor.shape
                        token_type[i, :n_texts, :seq_len] = tensor
                    collated["text_token_type_ids"] = token_type

            return collated

        return _collate


class RegionAlignmentDatasetTemplate(AlignmentDatasetBase):
    """区域级图文对齐数据集模板."""

    example_cls = RegionAlignmentExample

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str,
        *,
        manifest_region_key: str = "regions",
        region_crop_size: int = 224,
        region_padding: float = 0.1,
        region_transform: Optional[Any] = None,
        max_regions_per_image: Optional[int] = None,
        region_sampling_strategy: str = "truncate",
        return_full_image: bool = True,
        **kwargs,
    ):
        self.manifest_region_key = manifest_region_key
        self.region_crop_size = region_crop_size
        self.region_padding = max(0.0, float(region_padding))
        self.region_transform = region_transform or default_clip_transform(region_crop_size)
        self.max_regions_per_image = max_regions_per_image if (max_regions_per_image or 0) > 0 else None
        self.region_sampling_strategy = region_sampling_strategy
        self.return_full_image = return_full_image
        super().__init__(root_dir, split, **kwargs)

    # Manifest 解析 -----------------------------------------------------------------
    def _extract_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = super()._extract_metadata(record)
        metadata.pop(self.manifest_region_key, None)
        return metadata

    def _manifest_record_to_example(self, record: Dict[str, Any]) -> Optional[RegionAlignmentExample]:
        base_example = super()._manifest_record_to_example(record)
        if base_example is None:
            return None

        region_field = record.get(self.manifest_region_key)
        regions = self._normalize_region_list(region_field)
        return RegionAlignmentExample(
            image_path=base_example.image_path,
            texts=base_example.texts,
            metadata=base_example.metadata,
            regions=regions,
        )

    def _normalize_region_list(self, raw: Any) -> List[RegionDescription]:
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, Iterable):
            return []

        regions: List[RegionDescription] = []
        for region in raw:
            if isinstance(region, dict):
                bbox = region.get("bbox") or region.get("box") or region.get("region")
                text = region.get("text") or region.get("caption") or region.get("description") or ""
                category_id = region.get("category_id")
                meta = {k: v for k, v in region.items() if k not in {"bbox", "box", "region", "text", "caption", "description"}}
            elif isinstance(region, (list, tuple)) and len(region) >= 4:
                bbox = list(region[:4])
                text = ""
                category_id = None
                meta = {}
            else:
                continue

            bbox_tuple = self._to_xyxy_tuple(bbox)
            if bbox_tuple is None:
                continue

            regions.append(
                RegionDescription(
                    bbox=bbox_tuple,
                    text=text,
                    category_id=category_id if category_id is not None else None,
                    metadata=meta,
                )
            )
        return regions

    def _to_xyxy_tuple(self, bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None:
            return None
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except json.JSONDecodeError:
                return None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2_or_w, y2_or_h = map(float, bbox[:4])
            if x2_or_w <= x1 or y2_or_h <= y1:
                # 视为 [x, y, w, h]
                x2 = x1 + max(0.0, x2_or_w)
                y2 = y1 + max(0.0, y2_or_h)
            else:
                x2, y2 = x2_or_w, y2_or_h
            return (x1, y1, x2, y2)
        return None

    # 采样 & 裁剪 ------------------------------------------------------------------
    def _select_regions(self, regions: List[RegionDescription]) -> List[RegionDescription]:
        if not regions:
            return []
        if self.max_regions_per_image is None or len(regions) <= self.max_regions_per_image:
            return regions

        if self.region_sampling_strategy == "random":
            indices = sorted(random.sample(range(len(regions)), self.max_regions_per_image))
            return [regions[i] for i in indices]

        return regions[: self.max_regions_per_image]

    def _crop_region(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = width * self.region_padding
        pad_y = height * self.region_padding

        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(image.width, x2 + pad_x)
        y2 = min(image.height, y2 + pad_y)

        region = image.crop((x1, y1, x2, y2))
        region = region.resize((self.region_crop_size, self.region_crop_size), _PIL_RESIZE_BILINEAR)
        return region

    def _apply_region_transform(self, image: Image.Image) -> Tensor:
        tensor = self.region_transform(image) if self.region_transform else mtf.ToTensor()(image)
        if isinstance(tensor, torch.Tensor):
            return tensor
        return torch.as_tensor(tensor)

    # Dataset API ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if not isinstance(sample, RegionAlignmentExample):
            sample = self._coerce_example(sample)  # type: ignore[arg-type]
        sample = cast(RegionAlignmentExample, sample)

        image = self._load_image(sample.image_path)
        image_tensor = self._apply_image_transform(image) if self.return_full_image else torch.empty(0)

        regions = self._select_regions(sample.regions)
        if not regions:
            # 回退到整图
            fallback_bbox = (0.0, 0.0, float(image.width), float(image.height))
            regions = [
                RegionDescription(
                    bbox=fallback_bbox,
                    text=sample.texts[0] if sample.texts else "",
                    metadata={"auto_region": True},
                )
            ]

        region_images: List[Tensor] = []
        region_bboxes: List[List[float]] = []
        region_categories: List[int] = []
        region_texts: List[str] = []

        for region in regions:
            cropped = self._crop_region(image, region.bbox)
            tensor = self._apply_region_transform(cropped)
            region_images.append(tensor)
            region_bboxes.append(list(region.bbox))
            if region.category_id is not None:
                region_categories.append(int(region.category_id))
            region_texts.append(region.text or "")

        region_stack = torch.stack(region_images, dim=0)
        bboxes_tensor = torch.tensor(region_bboxes, dtype=torch.float32)
        tokenized = self._tokenize_texts(region_texts)

        item: Dict[str, Any] = {
            "image": image_tensor,
            "regions": region_stack,
            "region_bboxes": bboxes_tensor,
            "region_texts": region_texts,
            "region_text_ids": tokenized["input_ids"],
            "metadata": sample.metadata,
            "image_path": str(sample.image_path),
        }
        if "attention_mask" in tokenized:
            item["region_text_attn_mask"] = tokenized["attention_mask"]
        if region_categories:
            item["region_category_ids"] = torch.as_tensor(region_categories, dtype=torch.int64)

        return item

    def get_collate_fn(self):
        pad_value = self.pad_token_id
        max_regions_cap = self.max_regions_per_image

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            images = torch.stack([item["image"] for item in batch], dim=0) if batch[0]["image"].numel() else None

            region_tensors = [item["regions"] for item in batch]
            max_regions = max(t.shape[0] for t in region_tensors)
            if max_regions_cap is not None:
                max_regions = min(max_regions, max_regions_cap)
            _, C, H, W = region_tensors[0].shape
            regions = torch.zeros((len(batch), max_regions, C, H, W), dtype=region_tensors[0].dtype)
            for i, tensor in enumerate(region_tensors):
                n = min(tensor.shape[0], max_regions)
                regions[i, :n] = tensor[:n]

            bbox_tensors = [item["region_bboxes"] for item in batch]
            bboxes = torch.full((len(batch), max_regions, 4), fill_value=-1.0, dtype=bbox_tensors[0].dtype)
            for i, tensor in enumerate(bbox_tensors):
                n = min(tensor.shape[0], max_regions)
                bboxes[i, :n] = tensor[:n]

            text_tensors = [item["region_text_ids"] for item in batch]
            max_seq = max(t.shape[1] for t in text_tensors)
            text_ids = torch.full((len(batch), max_regions, max_seq), pad_value, dtype=text_tensors[0].dtype)
            for i, tensor in enumerate(text_tensors):
                n = min(tensor.shape[0], max_regions)
                seq_len = tensor.shape[1]
                text_ids[i, :n, :seq_len] = tensor[:n]

            collated: Dict[str, Any] = {
                "regions": regions,
                "region_bboxes": bboxes,
                "region_text_ids": text_ids,
                "region_texts": [item["region_texts"] for item in batch],
                "metadata": [item["metadata"] for item in batch],
                "image_paths": [item["image_path"] for item in batch],
            }

            if images is not None:
                collated["images"] = images

            if "region_text_attn_mask" in batch[0]:
                attn_tensors = [item["region_text_attn_mask"] for item in batch if "region_text_attn_mask" in item]
                if attn_tensors:
                    attn = torch.zeros((len(batch), max_regions, max_seq), dtype=attn_tensors[0].dtype)
                    for i, tensor in enumerate(attn_tensors):
                        n = min(tensor.shape[0], max_regions)
                        seq_len = tensor.shape[1]
                        attn[i, :n, :seq_len] = tensor[:n]
                    collated["region_text_attn_mask"] = attn

            if "region_category_ids" in batch[0]:
                cat_tensors = [item.get("region_category_ids") for item in batch]
                cat = torch.full((len(batch), max_regions), fill_value=-1, dtype=torch.int64)
                for i, tensor in enumerate(cat_tensors):
                    if tensor is None:
                        continue
                    n = min(tensor.shape[0], max_regions)
                    cat[i, :n] = tensor[:n]
                collated["region_category_ids"] = cat

            return collated

        return _collate


__all__ = [
    "ImageTextExample",
    "RegionDescription",
    "RegionAlignmentExample",
    "AlignmentDatasetBase",
    "ClipAlignmentDatasetTemplate",
    "RegionAlignmentDatasetTemplate",
]

