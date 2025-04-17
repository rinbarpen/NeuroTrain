import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import yaml
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset
from utils.util import save_numpy_data

class ISIC2017Dataset(CustomDataset):
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import yaml
from typing import Literal

from utils.dataset.custom_dataset import CustomDataset, Betweens
from utils.util import save_numpy_data

class ISIC2017Dataset(CustomDataset):
    mapping={"train": ("2017_Training_Data/*.jpg", "2017_Training_Part1_GroundTruth/*.jpg"), 
            "valid": ("ISIC-2017_Validation_Data/*.jpg", "ISIC-2017_Validation_Part1_GroundTruth/*.jpg"),
            "test":  ("ISIC-2017_Test_v2_Data/*.jpg", "ISIC-2017_Test_v2_Part1_GroundTruth/*.jpg")}
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(ISIC2017Dataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        if use_numpy:
            image_glob = image_glob.replace('*.jpg', '*.npy')
            label_glob = label_glob.replace('*.jpg', '*.npy')

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

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
    def to_numpy(save_dir: Path, base_dir: Path, betweens: Betweens, **kwargs):
        save_dir = save_dir / ISIC2017Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2017Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = ISIC2017Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = ISIC2017Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        for dataset, dataset_type in zip((train_dataset, valid_dataset, test_dataset), ('train', 'valid', 'test')):
            for i, (image, mask) in enumerate(dataset):
                image_dir = base_dir / ISIC2017Dataset.mapping[dataset_type][0].replace('*.jpg', f'{i}.npy')
                mask_dir = base_dir / ISIC2017Dataset.mapping[dataset_type][1].replace('*.jpg', f'{i}.npy')
                save_numpy_data(image_dir, image)
                save_numpy_data(mask_dir, mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "ISIC2017"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'valid', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)
from torchvision import transforms
    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'test', 'valid'], between: tuple[float, float]=(0.0, 1.0), transforms: transforms.Compose|None=None, use_numpy=False, is_rgb=False, **kwargs):
        super(ISIC2017Dataset, self).__init__(base_dir, dataset_type, between, use_numpy=use_numpy)

        if 'source' in kwargs.keys() and 'target' in kwargs.keys():
            image_glob, label_glob = kwargs['source'], kwargs['target']
        else:
            image_glob, label_glob = self.mapping[dataset_type]

        self.transforms = transforms
        self.config = {"is_rgb": is_rgb, "source": image_glob, "target": label_glob}

        if use_numpy:
            image_glob = image_glob.replace('*.jpg', '*.npy')
            label_glob = label_glob.replace('*.jpg', '*.npy')

        images = [p for p in base_dir.glob(image_glob)]
        masks = [p for p in base_dir.glob(label_glob)]

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
        save_dir = save_dir / ISIC2017Dataset.name()
        save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ISIC2017Dataset.get_train_dataset(base_dir, between=betweens['train'], **kwargs)
        valid_dataset = ISIC2017Dataset.get_valid_dataset(base_dir, between=betweens['valid'], **kwargs)
        test_dataset  = ISIC2017Dataset.get_test_dataset(base_dir, between=betweens['test'], **kwargs)

        for dataset, dataset_type in zip((train_dataset, valid_dataset, test_dataset), ('train', 'valid', 'test')):
            for i, (image, mask) in enumerate(dataset):
                image_dir = base_dir / ISIC2017Dataset.mapping[dataset_type][0].replace('*.jpg', f'{i}.npy')
                mask_dir = base_dir / ISIC2017Dataset.mapping[dataset_type][1].replace('*.jpg', f'{i}.npy')
                save_numpy_data(image_dir, image)
                save_numpy_data(mask_dir, mask)

        config_file = save_dir / "config.yaml"
        with config_file.open('w', encoding='utf-8') as f:
            yaml.dump({"betweens": betweens, **kwargs}, f)

    @staticmethod
    def name():
        return "ISIC2017"
    @staticmethod
    def get_train_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'train', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_valid_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'valid', between, use_numpy=use_numpy, **kwargs)
    @staticmethod
    def get_test_dataset(base_dir: Path, between: tuple[float, float]=(0.0, 1.0), use_numpy=False, **kwargs):
        return ISIC2017Dataset(base_dir, 'test', between, use_numpy=use_numpy, **kwargs)
