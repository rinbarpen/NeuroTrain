from pathlib import Path
from typing import Literal, Union
from PIL import Image

from ..custom_dataset import CustomDataset

class ISIC2016Dataset(CustomDataset):
    mapping = {
        'train': ('{task_type}/ISBI2016_ISIC_Part1_Training_Data', '{task_type}/ISBI2016_ISIC_Part1_Training_GroundTruth'),
        'valid': ('{task_type}/ISBI2016_ISIC_Part1_Test_Data', '{task_type}/ISBI2016_ISIC_Part1_Test_GroundTruth'),
        'test': ('{task_type}/ISBI2016_ISIC_Part1_Test_Data', '{task_type}/ISBI2016_ISIC_Part1_Test_GroundTruth'),
    }
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'valid', 'test'], task_type: Literal['Task1', 'Task2', 'Task3'], **kwargs):
        super(ISIC2016Dataset, self).__init__(root_dir, split)
        self.task_type = task_type

        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / self.mapping[split][0].format(task_type=task_type)
        self.mask_dir = self.root_dir / self.mapping[split][1].format(task_type=task_type)
        # 获取所有图像文件并排序
        self.image_paths = sorted([p for p in self.image_dir.glob('*.jpg')])
        
        # 为每个图像文件找到对应的掩码文件
        self.mask_paths = []
        for img_path in self.image_paths:
            # 从图像文件名提取ID (例如: ISIC_0000000.jpg -> ISIC_0000000)
            img_id = img_path.stem
            # 构建对应的掩码文件名 (例如: ISIC_0000000_Segmentation.png)
            mask_filename = f"{img_id}_Segmentation.png"
            mask_path = self.mask_dir / mask_filename
            
            # 检查掩码文件是否存在
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                # 如果掩码文件不存在，从图像列表中移除对应的图像
                print(f"Warning: Mask file not found for {img_path}, skipping this sample")
        
        # 确保图像和掩码数量一致
        if len(self.image_paths) != len(self.mask_paths):
            # 重新构建配对的列表
            paired_paths = []
            for img_path in self.image_paths:
                img_id = img_path.stem
                mask_filename = f"{img_id}_Segmentation.png"
                mask_path = self.mask_dir / mask_filename
                if mask_path.exists():
                    paired_paths.append((img_path, mask_path))
            
            self.image_paths = [pair[0] for pair in paired_paths]
            self.mask_paths = [pair[1] for pair in paired_paths]

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
            # 添加统一的图像尺寸变换
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 统一图像尺寸
                transforms.ToTensor(),
            ])
            image = transform(image)
            mask = transform(mask)
        # 返回字典格式，包含图像、掩码和元数据
        return {
            'image': image,  # (3, H, W)
            'mask': mask,    # (1, H, W)
            'metadata': {
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'task_type': self.task_type,
                'split': self.dataset_type
            }
        }

    @staticmethod
    def name():
        return "ISIC2016"

    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], task_type: Literal['Task1', 'Task2', 'Task3'] = 'Task1', **kwargs):
        """获取训练集实例"""
        return ISIC2016Dataset(root_dir, 'train', task_type, **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], task_type: Literal['Task1', 'Task2', 'Task3'] = 'Task1', **kwargs):
        """获取验证集实例"""
        return ISIC2016Dataset(root_dir, 'valid', task_type, **kwargs)

    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], task_type: Literal['Task1', 'Task2', 'Task3'] = 'Task1', **kwargs):
        """获取测试集实例"""
        return ISIC2016Dataset(root_dir, 'test', task_type, **kwargs)

def get_isic2016_dataloader(root_dir: str|Path, split: Literal['train', 'valid', 'test'], task_type: Literal['Task1', 'Task2', 'Task3'], **kwargs):
    root_dir = Path(root_dir)
    dataloader = ISIC2016Dataset(root_dir, split, task_type, **kwargs).dataloader(
        batch_size=kwargs.get('batch_size', 1),
        shuffle=kwargs.get('shuffle', True),
        num_workers=kwargs.get('num_workers', 0),
        drop_last=kwargs.get('drop_last', False),
        pin_memory=kwargs.get('pin_memory', False),
    )
    return dataloader