from pathlib import Path
from typing import Literal
from PIL import Image

from .custom_dataset import CustomDataset

class ISIC2016Dataset(CustomDataset):
    mapping = {
        'train': ('{task_type}/ISBI2016_ISIC_Part1_Training_Data', '{task_type}/ISBI2016_ISIC_Part1_Training_GroundTruth'),
        'valid': ('{task_type}/ISBI2016_ISIC_Part1_Test_Data', '{task_type}/ISBI2016_ISIC_Part1_Test_GroundTruth'),
        'test': ('{task_type}/ISBI2016_ISIC_Part1_Test_Data', '{task_type}/ISBI2016_ISIC_Part1_Test_GroundTruth'),
    }
    def __init__(self, base_dir: Path, split: Literal['train', 'valid', 'test'], task_type: Literal['Task1', 'Task2', 'Task3'], **kwargs):
        super(ISIC2016Dataset, self).__init__(base_dir, split)
        self.task_type = task_type

        self.base_dir = base_dir
        self.image_dir = self.base_dir / self.mapping[split][0].format(task_type=task_type)
        self.mask_dir = self.base_dir / self.mapping[split][1].format(task_type=task_type)
        self.image_paths = [p for p in self.image_dir.glob('*.jpg')]
        self.mask_paths = [p for p in self.mask_dir.glob('*.png')]

        self.config = kwargs

        self.n = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if 'transform' in self.config:
            image = self.config['transform'](image)
            mask = self.config['transform'](mask)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
            mask = transform(mask)
        return image, mask # (3, H, W), (1, H, W)

def get_isic2016_dataloader(base_dir: str|Path, split: Literal['train', 'valid', 'test'], task_type: Literal['Task1', 'Task2', 'Task3'], **kwargs):
    base_dir = Path(base_dir)
    dataloader = ISIC2016Dataset(base_dir, split, task_type, **kwargs).dataloader(
        batch_size=kwargs.get('batch_size', 1),
        shuffle=kwargs.get('shuffle', True),
        num_workers=kwargs.get('num_workers', 0),
        drop_last=kwargs.get('drop_last', False),
        pin_memory=kwargs.get('pin_memory', False),
    )
    return dataloader