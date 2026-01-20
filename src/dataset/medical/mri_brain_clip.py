import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms as mtf
import os.path as osp
from pathlib import Path

from typing import Union

from ..custom_dataset import CustomDataset

TEMPALTES = [
    "{label} exists in the brain mri image.", 
    "the brain mri image has {label}.",
    "there is {label} in the brain mri image.",
]

# "/media/rczx/Data/data/MRI_BRAIN_CLIP"
class MriBrainClipDataset(CustomDataset):
    def __init__(self, base_dir: Union[str, Path], split: str, transform: mtf.Compose = mtf.Compose([
        mtf.RandomHorizontalFlip(),
        mtf.RandomVerticalFlip(),
        mtf.RandomRotation(degrees=15),
        mtf.PILToTensor(),
        mtf.ConvertImageDtype(torch.float32),
    ])):
        base_path = Path(base_dir)
        super().__init__(base_path, split, transform=transform)
        self.base_dir = base_path

        # {base_dir}/{cancer}/{image_filename}.png | {image_filename}_mask.png
        self.cancer_types = [d.name for d in self.base_dir.iterdir()]
        self.image_paths = {ct: sorted([d for d in (self.base_dir / ct).glob("*.png") if not d.name.endswith("_mask.png")]) for ct in self.cancer_types}

        self.cancer_n = np.cumsum([len(v) for v in self.image_paths.values()])
        self.n = self.cancer_n[-1]

    def __getitem__(self, index):
        # 计算当前索引对应的癌症类型和图片索引
        for i, n in enumerate(self.cancer_n):
            if index < n:
                ct = self.cancer_types[i]
                img_idx = index - (self.cancer_n[i - 1] if i > 0 else 0)
                break

        # 获取图片路径并加载
        image_path = self.image_paths[ct][img_idx]
        image = Image.open(image_path).convert('RGB')

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'cancer': ct}

class MRI_Brain_from_classification(CustomDataset):
    mapping = {
        'train': 'Training',
        'test': 'Testing',
        'val': 'Testing',
    }
    TEMPLATES = [
        "{cancer} is exposed in the {image_source}", 
    ]
    def __init__(self, base_dir: Union[str, Path], split: str='train', *, processor=None):
        super().__init__()
        base_path = Path(base_dir)
        self.data_dir = base_path / self.mapping[split]
        self.processor = processor
        self.samples = []
        self.lesions = [x.name for x in self.data_dir.iterdir()]
        for lesion in self.lesions:
            lesion_dir = self.data_dir / lesion
            for img_path in lesion_dir.glob('*.jpg'):
                self.samples.append((img_path, lesion))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, lesion = self.samples[index]
        if 'no' in lesion:
            text = "No abnormal findings exposed in the mri brain slice"
        else:
            text = self.TEMPLATES[0].format(cancer=lesion, image_source="mri brain slice")
        image = Image.open(img_path).convert("RGB")
        if self.processor:
            outputs = self.processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to('cuda')

        return {**outputs, "cancer_type": lesion, }
