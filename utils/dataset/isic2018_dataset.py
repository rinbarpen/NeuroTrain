from torch.utils.data import Dataset
from pathlib import Path

from utils.transform import to_gray, to_rgb, image_transform

def get_isic2018_train_dataset(base_dir: Path, to_rgb=False):
    train_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Training_Input",
                                    base_dir / "ISIC2018_Task1_Training_GroundTruth",
                                    to_rgb)
    return train_dataset

def get_isic2018_valid_dataset(base_dir: Path, to_rgb=False):
    valid_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Validation_Input",
                                    base_dir / "ISIC2018_Task1_Validation_GroundTruth",
                                    to_rgb)
    return valid_dataset

def get_isic2018_test_dataset(base_dir: Path, *, to_rgb=False):
    test_dataset = ISIC2018Dataset(base_dir / "ISIC2018_Task1-2_Test_Input",
                                   base_dir / "ISIC2018_Task1_Test_GroundTruth",
                                   to_rgb)
    return test_dataset

class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, to_rgb=False):
        super(ISIC2018Dataset, self).__init__()

        self.to_rgb = to_rgb
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = [p for p in image_dir.glob('*.jpg')]
        self.mask_paths = [p for p in mask_dir.glob('*.jpg')]

    def __getitem__(self, index):
        image_file, mask_file = self.image_paths[index], self.mask_paths[index]

        if self.to_rgb:
            image, mask = to_rgb(image_file), to_rgb(mask_file)
        else:
            image, mask = to_gray(image_file), to_gray(mask_file)

        image, mask = image_transform(image, (512, 512), self.to_rgb), image_transform(mask, (512, 512), self.to_rgb)

        return image, mask


    def __len__(self):
        return len(self.image_paths)
