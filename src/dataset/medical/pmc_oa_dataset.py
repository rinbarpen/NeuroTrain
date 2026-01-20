import pandas as pd
from pathlib import Path
from typing import Literal, Union
from PIL import Image
import json
from torchvision import transforms as mtf

from ..custom_dataset import CustomDataset

class PMCOADataset(CustomDataset):
    ROOT_DIR = "/media/yons/Datasets/PMC-OA"
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'test', 'valid']):
        super(PMCOADataset, self).__init__(root_dir, split)

        self.transform = mtf.Compose([
            mtf.Resize((224, 224)),
            mtf.PILToTensor(),
        ])

        self.samples = self._load_samples()
        self.n = len(self.samples)

    def _load_samples(self):
        self.image_dir = self.root_dir / "images" / "caption_T060_filtered_top4_sep_v0_subfigures"
        self.split_file = self.root_dir / f"{self.split}.jsonl"
        
        samples = []
        with open(self.split_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append({
                    "image_id": data["image"],
                    "caption": data["caption"]
                })

        return samples

    def __getitem__(self, index: int):
        sample = self.samples[index]
        
        image_id = sample["image_id"]
        image_path = self.image_dir / image_id
        caption = sample["caption"]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return {"image": image, "caption": caption}

    def __len__(self):
        return self.n
    
    def name(self):
        return "PMC-OA"