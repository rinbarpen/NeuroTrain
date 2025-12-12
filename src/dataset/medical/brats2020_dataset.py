from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from monai.transforms.compose import Compose
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst, ToTensor

from ..custom_dataset import CustomDataset

logger = logging.getLogger(__name__)


class BraTS2020Dataset(CustomDataset):
    """BraTS2020 数据集读取器，支持 MRI 多模态与可选文本描述。"""

    DATA_DIRS = {
        "train": "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        "valid": "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
        "test": "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
    }
    TEXT_DIR = "TextBraTS/TextBraTSData"
    MODALITY_SUFFIXES = {
        "t1": "_t1.nii",
        "t1ce": "_t1ce.nii",
        "t2": "_t2.nii",
        "flair": "_flair.nii",
    }
    SEG_SUFFIX = "_seg.nii"

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        modalities: Optional[Sequence[str]] = None,
        transform: Optional[Compose] = None,
        load_seg: bool = True,
        load_txt: bool = True,
        cache_data: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # 1. 优先初始化影响数据加载/缓存的属性
        self.modalities: List[str] = list(modalities or self.MODALITY_SUFFIXES.keys())
        self.load_seg = load_seg
        self.load_txt = load_txt

        # 2. 调用父类初始化
        super().__init__(Path(root_dir), split, **kwargs)

        # 3. 初始化其他属性
        self.transform = transform or Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Resize(spatial_size=(224, 224, 112)),
                NormalizeIntensity(nonzero=True, channel_wise=True),
                ToTensor(),
            ]
        )
        self.cache_data = cache_data

        self._tokenizer = None
        if self.load_txt and tokenizer:
            from transformers import AutoTokenizer

            tokenizer_params = {"trust_remote_code": True, "cache_dir": "cache"}
            if tokenizer_kwargs:
                tokenizer_params.update(tokenizer_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_params)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

        # 4. 如果没有从文件缓存加载数据，则手动加载并尝试保存缓存
        if not self._cache_loaded:
            self.samples = self._load_samples()
            self.n = len(self.samples)
            self._save_to_cache_if_needed()
            
        self._sample_cache: Dict[int, Dict[str, Any]] = {} if cache_data else {}

    def _get_cache_config(self) -> Dict[str, Any]:
        """生成缓存配置，用于计算缓存键"""
        config = super()._get_cache_config()
        config.update({
            "modalities": self.modalities,
            "load_seg": self.load_seg,
            "load_txt": self.load_txt,
            "split": self.split
        })
        return config

    def _split_root(self) -> Path:
        split_root = self.root_dir / self.DATA_DIRS[self.split]
        if not split_root.exists():
            raise FileNotFoundError(f"数据目录不存在: {split_root}")
        return split_root

    def _load_samples(self) -> List[Dict[str, Path | str]]:
        samples: List[Dict[str, Path | str]] = []
        data_root = self._split_root()
        text_root = (self.root_dir / self.TEXT_DIR) if self.load_txt else None

        subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            sample_paths: Dict[str, Path | str] = {"id": subject_id}

            for modality in self.modalities:
                suffix = self.MODALITY_SUFFIXES.get(modality, f"_{modality}.nii")
                modality_path = subject_dir / f"{subject_id}{suffix}"
                
                sample_paths[modality] = modality_path

            if self.load_seg:
                seg_path = subject_dir / f"{subject_id}{self.SEG_SUFFIX}"
                sample_paths["seg"] = seg_path

            if self.load_txt and text_root is not None:
                txt_path = text_root / subject_id / f"{subject_id}_flair_text.txt"
                
                sample_paths["flair_txt"] = txt_path

            samples.append(sample_paths)

        return samples

    def __len__(self) -> int:
        return self.n

    def _load_text(self, text_path: Path) -> Any:
        with text_path.open("r", encoding="utf-8") as fp:
            text = fp.read()
        if self._tokenizer is None:
            return text
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} 超出范围 0-{len(self) - 1}")

        if self.cache_data and idx in self._sample_cache:
            return self._sample_cache[idx]

        sample_paths = self.samples[idx]
        result: Dict[str, Any] = {"id": sample_paths["id"]}

        for modality in self.modalities:
            result[modality] = self.transform(sample_paths[modality])

        if self.load_seg and "seg" in sample_paths:
            result["seg"] = self.transform(sample_paths["seg"])

        if self.load_txt and "flair_txt" in sample_paths:
            text_path = cast(Path, sample_paths["flair_txt"])
            result["flair_txt"] = self._load_text(text_path)

        if self.cache_data:
            self._sample_cache[idx] = result

        return result

    @staticmethod
    def name() -> str:
        return "brats2020"
    
    def metadata(self, **kwargs) -> dict:
        return {
            'task_type': 'segmentation',
            'modalities': self.modalities,
            'metrics': ['dice'],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = BraTS2020Dataset(
        root_dir=Path("/media/yons/Datasets/BraTS2020"),
        split="train",
        load_txt=True,
        cache_data=False,
        enable_cache=True,
        tokenizer="emilyalsentzer/Bio_ClinicalBERT",
    )
    print(ds[0])

    dl = ds.dataloader(batch_size=1, shuffle=True)
    for batch in dl:
        print(batch)
        break
