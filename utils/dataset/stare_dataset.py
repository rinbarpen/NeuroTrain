import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import yaml
import numpy as np

from utils.dataset.custom_dataset import CustomDataset
from utils.util import save_numpy_data

class StareDataset(CustomDataset):
    def __init__(self, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False):
        super(StareDataset, self).__init__(base_dir, between, use_numpy=use_numpy)

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb}

        image, label = "images", "1st_labels_ah"
        images = [p for p in (base_dir / image).glob('*.png')] if not use_numpy else [p for p in (base_dir / image).glob('*.npy')]
        masks = [p for p in (base_dir / label).glob('*.png')] if not use_numpy else [p for p in (base_dir / label).glob('*.npy')]

        self.n = len(images)
        bw = (int(self.between[0] * self.n), int(self.between[1] * self.n))
        self.images = images[bw[0]:bw[1]]
        self.masks = masks[bw[0]:bw[1]]

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        if self.use_numpy:
            return torch.from_numpy(np.load(image)), torch.from_numpy(np.load(mask))
        
        if self.config['is_rgb']:
            image, mask = Image.open(image).convert('RGB'), Image.open(mask).convert('RGB')
        else:
            image, mask = Image.open(image).convert('L'), Image.open(mask).convert('L')
        
        image, mask = self.transforms(image), self.transforms(mask)
        return image, mask

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: dict[str, tuple[float, float]], **kwargs):
        save_dir = save_dir / StareDataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        image_dir = save_dir / "images"
        mask_dir = save_dir / "1st_labels_ah"

        train_dataset = StareDataset.get_train_dataset(base_dir / "training", between=betweens['train'])
        test_dataset = StareDataset.get_test_dataset(base_dir / "test", between=betweens['test'])

        for i, (image, mask) in enumerate(train_dataset):
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)
        for i, (image, mask) in enumerate(test_dataset):
            save_numpy_data(image_dir / f'{i}.npy', image)
            save_numpy_data(mask_dir / f'{i}.npy', mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "STARE"
