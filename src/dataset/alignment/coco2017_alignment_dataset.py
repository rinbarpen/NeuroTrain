"""
COCO Region Alignment Dataset

将COCO图像及其区域标注转换为 region-level 样本。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as mtf
from transformers import AutoTokenizer

from ..custom_dataset import CustomDataset

from src.constants import PRETRAINED_MODEL_DIR

logger = logging.getLogger(__name__)


class COCO2017RegionAlignment(CustomDataset):
    """COCO 区域对齐数据集。

    每个样本由整张图像及其所有目标区域构成。
    返回的数据结构包含整图、区域图像、bbox、类别以及生成的文本描述。
    """
    
    mapping = {
        'train': 'train2017',
        'val': 'val2017',
        'valid': 'val2017',
        'test': 'val2017'
    }
    
    # COCO类别ID到类别名称的映射
    COCO_CATEGORIES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }

    OBJECT_TEXT = "{label} is in the image."
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Literal['train', 'val', 'test', 'valid'],
        transform=None,
        tokenizer='openai/clip-vit-base-patch32',
        **kwargs
    ):
        super(COCO2017RegionAlignment, self).__init__(root_dir, split, **kwargs)
        
        self.transform = transform or mtf.Compose([
            mtf.Resize((224, 224)),
            mtf.PILToTensor(),
            mtf.ConvertImageDtype(torch.float32),
            mtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, cache_dir=PRETRAINED_MODEL_DIR)
        self.root_dir = Path(root_dir)
        
        if split not in self.mapping:
            raise ValueError(f"Unsupported split: {split}. Must be one of {list(self.mapping.keys())}")
        coco_split = self.mapping[split]
        
        self.img_dir = self.root_dir / f'{coco_split}'
        self.samples = self._load_samples()
        
        self.n = len(self.samples)
        
        logger.info(f"Loaded {self.n} region samples from COCO {split} set")
    
    def _load_samples(self):
        """构建region级别的样本列表"""
        # parquet 文件命名约定：train -> train，val/valid -> valid，test -> test
        file_split_map = {
            'train': 'train',
            'val': 'valid',
            'valid': 'valid',
            'test': 'test',
        }
        file_split = file_split_map.get(self.split, self.split)

        parquet_path = self.root_dir / f'bbox_list_{file_split}.parquet'
        csv_path = self.root_dir / f'bbox_list_{file_split}.csv'

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            for col in ['bboxes', 'labels', 'category_ids']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        else:
            raise FileNotFoundError(
                f"未找到 bbox_list 文件: {parquet_path} 或 {csv_path}"
            )

        return df
    
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Dict:  # type: ignore[override]
        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]
        else:
            sample = self.samples[index]

        img_path = self.img_dir / sample['file_name']
        image = Image.open(img_path).convert('RGB')

        bboxes = sample['bboxes']
        category_ids = torch.tensor(np.array(sample['category_ids']), dtype=torch.int64)
        labels = sample['labels']
        region_texts = [self.OBJECT_TEXT.format(label=label) for label in labels]
        tokenized = self.tokenizer(region_texts, padding=True, truncation=True, return_tensors="pt")
        region_text_ids = tokenized['input_ids']
        region_text_attn_mask = tokenized.get('attention_mask')

        image_tensor = self.transform(image)
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        region_images: List[torch.Tensor] = [] # [(C, H, W)]
        for box in bboxes:
            x, y, w, h = box.tolist()
            region_image = image.crop((x, y, x + w, y + h))
            region_tensor = self.transform(region_image)
            if not isinstance(region_tensor, torch.Tensor):
                region_tensor = torch.from_numpy(np.array(region_image)).permute(2, 0, 1).float() / 255.0
            region_images.append(region_tensor)

        return {
            'inputs': torch.stack([image_tensor, *region_images], dim=0), # (N_BG+N_OBJ, C, H, W)
            'text_ids': region_text_ids,
            'text_attn_mask': region_text_attn_mask,
            'category_ids': category_ids,
        }
    
    @staticmethod
    def name() -> str:
        return "COCO2017RegionAlignment"
    
    @staticmethod
    def metadata(**kwargs) -> Dict:
        return {
            'task_type': 'region_caption',
            'num_classes': 80,  # COCO有80个目标类别
            'metrics': ['BLEU', 'METEOR', 'ROUGE', 'CIDEr', 'mAP'],
            'dataset_name': 'COCO2017RegionAlignment',
        }
    
    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):  # type: ignore[override]
        return COCO2017RegionAlignment(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):  # type: ignore[override]
        return COCO2017RegionAlignment(root_dir, 'val', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):  # type: ignore[override]
        logger.warning("COCO2017 test set annotations are not publicly available, using validation set instead")
        return COCO2017RegionAlignment(root_dir, 'val', **kwargs)
    
    def get_category_name(self, category_id: int|None=None) -> str|dict[int, str]:
        """根据类别ID获取类别名称"""
        if category_id is None:
            return self.COCO_CATEGORIES
        else:
            return self.COCO_CATEGORIES.get(category_id, 'unknown')

    def get_collate_fn(self):
        return self.collate_fn

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        batch_size = len(batch)

        # Inputs (包含整图 + 区域)
        inputs_list = [item['inputs'] for item in batch]
        max_len = max(t.shape[0] for t in inputs_list)
        _, C, H, W = inputs_list[0].shape

        inputs = torch.zeros((batch_size, max_len, C, H, W), dtype=inputs_list[0].dtype)
        for i, tensor in enumerate(inputs_list):
            n = tensor.shape[0]
            inputs[i, :n] = tensor

        # 文本 token ids
        text_list = [item['text_ids'] for item in batch]
        has_attn = bool(batch) and batch[0].get('text_attn_mask') is not None
        attn_list = [item['text_attn_mask'] for item in batch] if has_attn else None
        max_regions = max(t.shape[0] for t in text_list) if text_list else 0
        max_seq_len = max(t.shape[1] for t in text_list) if text_list else 0

        text_ids = torch.zeros((batch_size, max_regions, max_seq_len), dtype=text_list[0].dtype if max_regions > 0 else torch.long)
        for i, tensor in enumerate(text_list):
            n, seq_len = tensor.shape
            text_ids[i, :n, :seq_len] = tensor

        text_attn_mask = None
        if attn_list:
            text_attn_mask = torch.zeros((batch_size, max_regions, max_seq_len), dtype=attn_list[0].dtype)
            for i, tensor in enumerate(attn_list):
                n, seq_len = tensor.shape
                text_attn_mask[i, :n, :seq_len] = tensor

        # 类别 id
        category_list = [item['category_ids'] for item in batch]
        max_category = max(cat.shape[0] if isinstance(cat, torch.Tensor) else len(cat) for cat in category_list) if category_list else 0
        category_ids = torch.full((batch_size, max_category), fill_value=-1, dtype=torch.int64)
        for i, cat in enumerate(category_list):
            cat_tensor = cat if isinstance(cat, torch.Tensor) else torch.as_tensor(cat, dtype=torch.int64)
            cat_tensor = cat_tensor.view(-1)
            n = cat_tensor.shape[0]
            category_ids[i, :n] = cat_tensor

        output = {
            'inputs': inputs,
            'text_ids': text_ids,
            'category_ids': category_ids,
        }
        if text_attn_mask is not None:
            output['text_attn_mask'] = text_attn_mask
        return output

if __name__ == '__main__':
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    logging.basicConfig(level=logging.INFO)

    root_dir = Path('data') / 'coco2017'

    print("Testing COCO2017RegionAlignment...")
    ds = COCO2017RegionAlignment(root_dir=root_dir, split='train')
    print(f"Dataset size: {len(ds)}")

    if len(ds) > 0:
        sample = ds[0]
        print(type(sample['inputs']), type(sample['text_ids']), type(sample['category_ids']))
        print(f"Image shape: {sample['inputs'].shape}")
        print(f"Region images shape: {sample['inputs'].shape}")
        print(f"First region text: {sample['text_ids'].shape}")

    dataloader = ds.dataloader(
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=COCO2017RegionAlignment.collate_fn,
    )
    for batch in dataloader:
        print(batch)
        break
