import torch
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms

from utils.dataset.custom_dataset import CustomDataset
from utils.util import save_numpy_data

# Instance Segmentation Dataset
class BOWL2018Dataset(CustomDataset):
    def __init__(self, base_dir: Path, between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False):
        super(BOWL2018Dataset, self).__init__(base_dir, between, use_numpy=use_numpy)

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb}

        paths = [p for p in base_dir.iterdir() if p.is_dir()]

        self.n = len(paths)
        bw = (int(self.between[0] * self.n), int(self.between[1] * self.n))
        self.paths = paths[bw[0]:bw[1]]

    def __getitem__(self, index):
        path = self.paths[index]
        image_dir, mask_dir = path / "images", path / "masks"
        if self.use_numpy:
            images = [torch.from_numpy(np.load(p)) for p in image_dir.iter_dir()][0]
            masks = [torch.from_numpy(np.load(p)) for p in mask_dir.iter_dir()]
            images, masks = torch.hstack(images), torch.hstack(masks)
            return images, masks

        if self.config['is_rgb']:
            images = [Image.open(p).convert('RGB') for p in image_dir.iter_dir()][0]
            masks = [Image.open(p).convert('RGB') for p in mask_dir.iter_dir()]
        else:
            images = [Image.open(p).convert('L') for p in image_dir.iter_dir()][0]
            masks = [Image.open(p).convert('L') for p in mask_dir.iter_dir()]

        images = self.transforms(images)
        masks = [self.transforms(mask) for mask in masks]
        images, masks = torch.hstack(images), torch.hstack(masks)
        return images, masks

    @staticmethod
    def to_numpy(save_dir: Path, base_dir: Path, betweens: dict[str, tuple[float, float]], **kwargs):
        save_dir = save_dir / BOWL2018Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = BOWL2018Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = BOWL2018Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = BOWL2018Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        for i, (image, masks) in enumerate(train_dataset):
            image_dir, mask_dir = save_dir / i / "images", save_dir / i / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)
        for i, (image, masks) in enumerate(valid_dataset):
            image_dir, mask_dir = save_dir / i / "images", save_dir / i / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)
        for i, (image, masks) in enumerate(test_dataset):
            image_dir, mask_dir = save_dir / i / "images", save_dir / i / "masks"
            save_numpy_data(image_dir / f'{i}.npy', image)
            for j, mask in enumerate(masks):
                save_numpy_data(mask_dir / f'{i}_{j}.npy', mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "BOWL2018"
