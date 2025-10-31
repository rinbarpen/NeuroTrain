"""
Diffusion模型数据集支持

这个模块提供了专门为diffusion模型设计的数据集类，支持：
1. 无条件图像生成（unconditional generation）
2. 条件图像生成（conditional generation）- 支持类别标签、文本等条件
3. 图像-文本配对（用于text-to-image diffusion）
"""

from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image

from .custom_dataset import CustomDataset


class DiffusionDataset(CustomDataset):
    """Diffusion模型数据集基类
    
    特点：
    - 支持无条件和条件生成两种模式
    - 返回格式统一为 (image, condition) 元组
    - 对于无条件生成，condition 为 None
    - 对于条件生成，condition 可以是类别标签、文本描述等
    """
    
    def __init__(
        self, 
        root_dir: Path, 
        split: str, 
        desired_n: int = 0,
        conditional: bool = False,
        condition_type: str = 'label',  # 'label', 'text', 'custom'
        **kwargs
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集类型 ('train', 'test', 'valid')
            desired_n: 期望的样本数量
            conditional: 是否为条件生成模式
            condition_type: 条件类型 ('label': 类别标签, 'text': 文本描述, 'custom': 自定义)
            **kwargs: 其他扩展参数
        """
        super().__init__(root_dir, split, desired_n, **kwargs)
        self.conditional = conditional
        self.condition_type = condition_type
        
        # 对于条件生成，子类需要提供相关信息
        self.num_classes = kwargs.get('num_classes', None)
        self.class_names = kwargs.get('class_names', None)
        
    @abstractmethod
    def __getitem__(self, index) -> Tuple[torch.Tensor, Optional[Any]]:
        """获取指定索引的数据样本
        
        Returns:
            (image, condition) 元组
            - image: 图像张量，形状为 (C, H, W)
            - condition: 条件信息
              - 无条件模式：None
              - 类别标签模式：整数标签 (int) 或 one-hot 向量
              - 文本模式：文本字符串 (str) 或 token ids
              - 自定义模式：任意类型
        """
        ...
    
    @staticmethod
    def metadata(**kwargs) -> dict:
        """获取数据集元数据信息"""
        return {
            'task_type': 'generation',
            'subtask': 'diffusion',
            'metrics': ['fid', 'is', 'lpips'],  # Frechet Inception Distance, Inception Score, LPIPS
            **kwargs
        }
    
    def get_condition_info(self) -> Dict[str, Any]:
        """获取条件信息（用于模型配置）
        
        Returns:
            包含条件相关信息的字典
        """
        if not self.conditional:
            return {'conditional': False}
        
        info = {
            'conditional': True,
            'condition_type': self.condition_type,
        }
        
        if self.condition_type == 'label' and self.num_classes is not None:
            info['num_classes'] = self.num_classes
            if self.class_names is not None:
                info['class_names'] = self.class_names
        
        return info


class UnconditionalDiffusionDataset(DiffusionDataset):
    """无条件Diffusion数据集包装器
    
    将任意图像数据集转换为无条件diffusion数据集
    """
    
    def __init__(
        self, 
        base_dataset: CustomDataset,
        **kwargs
    ):
        """
        Args:
            base_dataset: 基础数据集（提供图像数据）
            **kwargs: 其他参数
        """
        self.base_dataset = base_dataset
        super().__init__(
            root_dir=base_dataset.root_dir,
            split=base_dataset.split,
            desired_n=len(base_dataset),
            conditional=False,
            **kwargs
        )
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, None]:
        """返回 (image, None)"""
        # 从基础数据集获取数据
        item = self.base_dataset[index]
        
        # 统一处理返回格式
        if isinstance(item, (tuple, list)):
            image = item[0]  # 假设第一个元素是图像
        else:
            image = item
        
        return image, None
    
    def __len__(self):
        return len(self.base_dataset)
    
    @staticmethod
    def name() -> str:
        return "unconditional_diffusion"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取训练数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return UnconditionalDiffusionDataset(base_dataset, **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取验证数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return UnconditionalDiffusionDataset(base_dataset, **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取测试数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return UnconditionalDiffusionDataset(base_dataset, **kwargs)


class ConditionalDiffusionDataset(DiffusionDataset):
    """条件Diffusion数据集包装器
    
    将带标签的分类数据集转换为条件diffusion数据集
    """
    
    def __init__(
        self, 
        base_dataset: CustomDataset,
        condition_type: str = 'label',
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            base_dataset: 基础数据集（提供图像和标签）
            condition_type: 条件类型
            num_classes: 类别数量
            class_names: 类别名称列表
            **kwargs: 其他参数
        """
        self.base_dataset = base_dataset
        
        # 尝试从基础数据集获取类别信息
        if num_classes is None:
            metadata = getattr(base_dataset, 'metadata', lambda: {})()
            num_classes = metadata.get('num_classes', None)
        
        if class_names is None and hasattr(base_dataset, 'classes'):
            class_names = base_dataset.classes
        
        super().__init__(
            root_dir=base_dataset.root_dir,
            split=base_dataset.split,
            desired_n=len(base_dataset),
            conditional=True,
            condition_type=condition_type,
            num_classes=num_classes,
            class_names=class_names,
            **kwargs
        )
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, Any]:
        """返回 (image, condition)"""
        item = self.base_dataset[index]
        
        # 统一处理返回格式
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            image, label = item[0], item[1]
        else:
            raise ValueError("基础数据集必须返回 (image, label) 元组")
        
        # 根据条件类型处理标签
        if self.condition_type == 'label':
            condition = label  # 直接使用标签
        elif self.condition_type == 'text':
            # 如果有类别名称，转换为文本
            if self.class_names is not None and isinstance(label, int):
                condition = self.class_names[label]
            else:
                condition = str(label)
        else:
            condition = label
        
        return image, condition
    
    def __len__(self):
        return len(self.base_dataset)
    
    @staticmethod
    def name() -> str:
        return "conditional_diffusion"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取训练数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return ConditionalDiffusionDataset(base_dataset, **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取验证数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return ConditionalDiffusionDataset(base_dataset, **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, base_dataset: CustomDataset = None, **kwargs):
        """获取测试数据集"""
        if base_dataset is None:
            raise ValueError("必须提供 base_dataset 参数")
        return ConditionalDiffusionDataset(base_dataset, **kwargs)


class TextToImageDiffusionDataset(DiffusionDataset):
    """文本到图像的Diffusion数据集
    
    用于训练text-to-image diffusion模型（如Stable Diffusion）
    """
    
    def __init__(
        self, 
        root_dir: Path, 
        split: str,
        image_paths: List[Path],
        captions: List[str],
        tokenizer = None,
        max_length: int = 77,
        **kwargs
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集类型
            image_paths: 图像路径列表
            captions: 文本描述列表
            tokenizer: 文本tokenizer（可选）
            max_length: 文本最大长度
            **kwargs: 其他参数
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            desired_n=len(image_paths),
            conditional=True,
            condition_type='text',
            **kwargs
        )
        
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(image_paths) == len(captions), "图像和文本数量必须相同"
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, Union[str, torch.Tensor]]:
        """返回 (image, caption)"""
        # 加载图像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        # 应用图像变换
        if hasattr(self, 'transform') and self.transform is not None:
            image = self.transform(image)
        else:
            # 默认转换为tensor
            import torchvision.transforms as T
            image = T.ToTensor()(image)
        
        # 处理文本
        caption = self.captions[index]
        if self.tokenizer is not None:
            # 如果提供了tokenizer，进行tokenize
            caption = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption = caption['input_ids'].squeeze(0)
        
        return image, caption
    
    def __len__(self):
        return len(self.image_paths)
    
    @staticmethod
    def name() -> str:
        return "text_to_image_diffusion"
    
    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练数据集"""
        return TextToImageDiffusionDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证数据集"""
        return TextToImageDiffusionDataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试数据集"""
        return TextToImageDiffusionDataset(root_dir, 'test', **kwargs)


def create_diffusion_dataset(
    base_dataset: CustomDataset,
    conditional: bool = False,
    condition_type: str = 'label',
    **kwargs
) -> DiffusionDataset:
    """创建Diffusion数据集的便捷函数
    
    Args:
        base_dataset: 基础数据集
        conditional: 是否为条件生成
        condition_type: 条件类型
        **kwargs: 其他参数
    
    Returns:
        DiffusionDataset实例
    
    Examples:
        >>> from src.dataset.mnist_dataset import MNISTDataset
        >>> 
        >>> # 创建无条件MNIST diffusion数据集
        >>> mnist_train = MNISTDataset.get_train_dataset(Path('data/mnist'))
        >>> diffusion_ds = create_diffusion_dataset(mnist_train, conditional=False)
        >>> 
        >>> # 创建条件MNIST diffusion数据集（带类别标签）
        >>> diffusion_ds = create_diffusion_dataset(mnist_train, conditional=True, condition_type='label')
        >>> 
        >>> # 使用数据
        >>> image, condition = diffusion_ds[0]
        >>> # 无条件时 condition 为 None
        >>> # 条件时 condition 为标签值
    """
    if conditional:
        return ConditionalDiffusionDataset(
            base_dataset=base_dataset,
            condition_type=condition_type,
            **kwargs
        )
    else:
        return UnconditionalDiffusionDataset(
            base_dataset=base_dataset,
            **kwargs
        )


# 常用数据集的diffusion包装器
def get_mnist_diffusion_dataset(
    root_dir: Path,
    split: str = 'train',
    conditional: bool = False,
    **kwargs
) -> DiffusionDataset:
    """获取MNIST Diffusion数据集
    
    Examples:
        >>> # 无条件MNIST生成
        >>> ds = get_mnist_diffusion_dataset(Path('data/mnist'), 'train', conditional=False)
        >>> 
        >>> # 条件MNIST生成（基于数字类别）
        >>> ds = get_mnist_diffusion_dataset(Path('data/mnist'), 'train', conditional=True)
    """
    from .mnist_dataset import MNISTDataset
    
    if split == 'train':
        base_ds = MNISTDataset.get_train_dataset(root_dir, **kwargs)
    elif split in ['valid', 'val']:
        base_ds = MNISTDataset.get_valid_dataset(root_dir, **kwargs)
    else:
        base_ds = MNISTDataset.get_test_dataset(root_dir, **kwargs)
    
    return create_diffusion_dataset(base_ds, conditional=conditional)


def get_cifar10_diffusion_dataset(
    root_dir: Path,
    split: str = 'train',
    conditional: bool = False,
    **kwargs
) -> DiffusionDataset:
    """获取CIFAR-10 Diffusion数据集
    
    Examples:
        >>> # 无条件CIFAR-10生成
        >>> ds = get_cifar10_diffusion_dataset(Path('data/cifar10'), 'train', conditional=False)
        >>> 
        >>> # 条件CIFAR-10生成（基于10个类别）
        >>> ds = get_cifar10_diffusion_dataset(Path('data/cifar10'), 'train', conditional=True)
    """
    from .cifar_dataset import CIFAR10Dataset
    
    if split == 'train':
        base_ds = CIFAR10Dataset.get_train_dataset(root_dir, **kwargs)
    elif split in ['valid', 'val']:
        base_ds = CIFAR10Dataset.get_valid_dataset(root_dir, **kwargs)
    else:
        base_ds = CIFAR10Dataset.get_test_dataset(root_dir, **kwargs)
    
    return create_diffusion_dataset(base_ds, conditional=conditional)


def get_imagenet_diffusion_dataset(
    root_dir: Path,
    split: str = 'train',
    conditional: bool = True,  # ImageNet通常用于条件生成
    **kwargs
) -> DiffusionDataset:
    """获取ImageNet Diffusion数据集
    
    Examples:
        >>> # 条件ImageNet生成（基于1000个类别）
        >>> ds = get_imagenet_diffusion_dataset(Path('data/imagenet'), 'train', conditional=True)
    """
    from .imagenet_dataset import ImageNet1KDataset
    
    if split == 'train':
        base_ds = ImageNet1KDataset.get_train_dataset(root_dir, **kwargs)
    elif split in ['valid', 'val']:
        base_ds = ImageNet1KDataset.get_valid_dataset(root_dir, **kwargs)
    else:
        base_ds = ImageNet1KDataset.get_test_dataset(root_dir, **kwargs)
    
    return create_diffusion_dataset(base_ds, conditional=conditional, condition_type='label')

