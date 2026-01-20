# OIA-DDR
# https://github.com/nkicsl/DDR-dataset


from ..custom_dataset import CustomDataset
from pathlib import Path
from PIL import Image
from typing import Union, Literal
import torch
from torchvision import transforms as mtf

class DDRDataset(CustomDataset):
    ROOT_DIR = "/media/yons/Datasets/OIA-DDR/DDR-dataset/"
    DR_severity = ["none", "mild", "moderate", "severe", "proliferative DR", "proliferative DR"]
    max_DR_severity = 5
    
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'valid', 'test'], **kwargs):
        super(DDRDataset, self).__init__(root_dir, split, **kwargs)

        self.transforms = mtf.Compose([
            mtf.Resize((512, 512)),
            mtf.PILToTensor(),
            mtf.ConvertImageDtype(torch.float32),
        ])

        self.samples = self._load_samples()
        self.image_dir = self.root_dir / "DR_grading" / self.split

        self.n = len(self.samples)
    
    def _load_samples(self):
        label_file = self.root_dir / "DR_grading" / f"{self.split}.txt"
        
        # load label info
        with open(label_file, 'r') as f:
            labels = f.readlines()
        labels = sorted(labels)
        labels = [label.strip().split(' ') for label in labels]
        labels = [(label[0], self.DR_severity[int(label[1])], int(label[1]) / self.max_DR_severity) for label in labels]
        return labels
    
    def __getitem__(self, index: int):
        image_id, diagnosis, diagnosis_severity = self.samples[index]
        image = Image.open(self.image_dir / image_id).convert('RGB')
        image = self.transforms(image)

        return {
            'image': image,
            'severity': diagnosis,
            'severity_point': diagnosis_severity,
        }

    def __len__(self):
        return self.n
    
    @staticmethod
    def name():
        return "OIA-DDR"

if __name__ == "__main__":
    ds = DDRDataset(root_dir="/media/yons/Datasets/OIA-DDR/DDR-dataset/", split="train")
    dl = ds.dataloader(batch_size=2, shuffle=True)
    for batch in dl:
        print(batch)
        break
