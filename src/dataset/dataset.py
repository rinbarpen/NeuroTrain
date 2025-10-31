import logging
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Union, Dict
from torch.utils.data import DataLoader, Subset


from src.config import get_config_value, get_config
# from src.utils.transform import get_transforms
from .custom_dataset import CustomDataset
from .hybrid_dataset import HybridDataset, create_hybrid_dataset_from_config
from .diffusion_dataset import (
    DiffusionDataset,
    UnconditionalDiffusionDataset,
    ConditionalDiffusionDataset,
    TextToImageDiffusionDataset,
    create_diffusion_dataset,
    get_mnist_diffusion_dataset,
    get_cifar10_diffusion_dataset,
    get_imagenet_diffusion_dataset,
)


# 数据集注册表：记录所有可用数据集及其信息
DATASET_REGISTRY = {
    'medical/mri_brain_clip': {
        'class': 'MriBrainClipDataset',
        'module': 'medical.mri_brain_clip_dataset',
        'task_type': ['multimodal', 'medical_imaging'],
        'description': 'MRI Brain CLIP dataset for multimodal learning',
        'aliases': ['medical/mri_brain_clip']
    },
    'medical/btcv': {
        'class': 'BTCVDataset',
        'module': 'medical.btcv_dataset',
        'task_type': ['segmentation', 'medical_imaging', '3d_segmentation'],
        'description': 'BTCV multi-organ segmentation dataset',
        'aliases': ['medical/btcv']
    },
    'drive': {
        'class': 'DriveDataset',
        'module': 'drive_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'retinal_vessel_segmentation'],
        'description': 'DRIVE retinal vessel segmentation dataset',
        'aliases': ['drive']
    },
    'medical/stare': {
        'class': 'StareDataset',
        'module': 'medical.stare_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'retinal_vessel_segmentation'],
        'description': 'STARE retinal vessel segmentation dataset',
        'aliases': ['medical/stare']
    },
    'medical/isic2016': {
        'class': 'ISIC2016Dataset',
        'module': 'medical.isic2016_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'skin_lesion_segmentation'],
        'description': 'ISIC 2016 skin lesion segmentation dataset',
        'aliases': ['medical/isic2016']
    },
    'medical/isic2017': {
        'class': 'ISIC2017Dataset',
        'module': 'medical.isic2017_dataset',
        'task_type': ['classification', 'medical_imaging', 'skin_lesion_classification'],
        'description': 'ISIC 2017 skin lesion classification dataset',
        'aliases': ['medical/isic2017']
    },
    'medical/isic2018': {
        'class': 'ISIC2018Dataset',
        'module': 'medical.isic2018_dataset',
        'task_type': ['segmentation', 'classification', 'medical_imaging'],
        'description': 'ISIC 2018 skin lesion analysis dataset',
        'aliases': ['medical/isic2018']
    },
    'medical/bowl2018': {
        'class': 'BOWL2018Dataset',
        'module': 'medical.bowl2018_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'cell_segmentation'],
        'description': 'Data Science Bowl 2018 nuclei segmentation dataset',
        'aliases': ['medical/bowl2018']
    },
    'medical/chasedb1': {
        'class': 'ChaseDB1Dataset',
        'module': 'medical.chasedb1_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'retinal_vessel_segmentation'],
        'description': 'CHASE-DB1 retinal vessel segmentation dataset',
        'aliases': ['medical/chasedb1']
    },
    'mnist': {
        'class': 'MNISTDataset',
        'module': 'mnist_dataset',
        'task_type': ['classification', 'digit_recognition'],
        'description': 'MNIST handwritten digit classification dataset',
        'aliases': ['mnist']
    },
    'medical/vqarad': {
        'class': 'VQARADDataset',
        'module': 'medical.vqa_rad_dataset',
        'task_type': ['vqa', 'medical_imaging', 'multimodal'],
        'description': 'VQA-RAD medical visual question answering dataset',
        'aliases': ['medical/vqarad']
    },
    'medical/pathvqa': {
        'class': 'PathVQADataset',
        'module': 'medical.pathvqa_dataset',
        'task_type': ['vqa', 'medical_imaging', 'multimodal'],
        'description': 'PathVQA pathology visual question answering dataset',
        'aliases': ['medical/pathvqa']
    },
    'medical/isic2016_reasoning_seg': {
        'class': 'ISIC2016ReasoningSegDataset',
        'module': 'medical.isic2016_reasoning_seg_dataset',
        'task_type': ['segmentation', 'medical_imaging', 'reasoning'],
        'description': 'ISIC 2016 with reasoning for segmentation',
        'aliases': ['medical/isic2016_reasoning_seg']
    },
    'brainmri_clip': {
        'class': 'BrainMRIClipDataset',
        'module': 'medical.brain_mri_clip_dataset',
        'task_type': ['multimodal', 'medical_imaging'],
        'description': 'Brain MRI CLIP dataset for multimodal learning',
        'aliases': ['brainmri_clip']
    },
    'coco': {
        'class': 'COCODataset',
        'module': 'coco_dataset',
        'task_type': ['detection', 'instance_segmentation', 'keypoint_detection', 'captioning'],
        'description': 'COCO dataset for detection, segmentation, keypoints, and captions',
        'aliases': ['coco']
    },
    'coco_segmentation': {
        'class': 'COCOSegmentationDataset',
        'module': 'coco_dataset',
        'task_type': ['semantic_segmentation', 'segmentation'],
        'description': 'COCO semantic segmentation dataset',
        'aliases': ['coco_segmentation', 'coco_seg']
    },
    'cifar10': {
        'class': 'CIFAR10Dataset',
        'module': 'cifar_dataset',
        'task_type': ['classification', 'image_classification'],
        'description': 'CIFAR-10 image classification dataset',
        'aliases': ['cifar', 'cifar10']
    },
    'cifar100': {
        'class': 'CIFAR100Dataset',
        'module': 'cifar_dataset',
        'task_type': ['classification', 'image_classification'],
        'description': 'CIFAR-100 image classification dataset',
        'aliases': ['cifar100']
    },
    'imagenet': {
        'class': 'ImageNet1KDataset',
        'module': 'imagenet_dataset',
        'task_type': ['classification', 'image_classification'],
        'description': 'ImageNet-1K large-scale image classification dataset',
        'aliases': ['imagenet', 'imagenet1k', 'imagenet-1k']
    },
    'diffusion/mnist': {
        'class': 'DiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'unconditional_generation', 'conditional_generation'],
        'description': 'MNIST dataset for diffusion models (unconditional/conditional)',
        'aliases': ['diffusion/mnist', 'mnist_diffusion']
    },
    'diffusion/cifar10': {
        'class': 'DiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'unconditional_generation', 'conditional_generation'],
        'description': 'CIFAR-10 dataset for diffusion models (unconditional/conditional)',
        'aliases': ['diffusion/cifar10', 'cifar10_diffusion']
    },
    'diffusion/imagenet': {
        'class': 'DiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'conditional_generation'],
        'description': 'ImageNet dataset for conditional diffusion models',
        'aliases': ['diffusion/imagenet', 'imagenet_diffusion']
    },
    'diffusion/unconditional': {
        'class': 'UnconditionalDiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'unconditional_generation'],
        'description': 'Generic unconditional diffusion dataset wrapper',
        'aliases': ['diffusion/unconditional']
    },
    'diffusion/conditional': {
        'class': 'ConditionalDiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'conditional_generation'],
        'description': 'Generic conditional diffusion dataset wrapper',
        'aliases': ['diffusion/conditional']
    },
    'diffusion/text_to_image': {
        'class': 'TextToImageDiffusionDataset',
        'module': 'diffusion_dataset',
        'task_type': ['generation', 'diffusion', 'text_to_image', 'multimodal'],
        'description': 'Text-to-image diffusion dataset',
        'aliases': ['diffusion/text_to_image', 'text2image', 't2i']
    },
}


def _get_dataset_by_case(dataset_name: str):
    """根据数据集名称返回对应的数据集类"""
    name = dataset_name.lower()
    if name == 'medical/mri_brain_clip':
        from .medical.brain_mri_clip_dataset import BrainMRIClipDataset
        return BrainMRIClipDataset
    elif name == 'medical/btcv':
        from .medical.btcv_dataset import BTCVDataset
        return BTCVDataset
    elif name == 'drive':
        from .drive_dataset import DriveDataset
        return DriveDataset
    elif name == 'medical/stare':
        from .medical.stare_dataset import StareDataset
        return StareDataset
    elif name == 'medical/isic2016':
        from .medical.isic2016_dataset import ISIC2016Dataset
        return ISIC2016Dataset
    elif name == 'medical/isic2017':
        from .medical.isic2017_dataset import ISIC2017Dataset
        return ISIC2017Dataset
    elif name == 'medical/isic2018':
        from .medical.isic2018_dataset import ISIC2018Dataset
        return ISIC2018Dataset
    elif name == 'medical/bowl2018':
        from .medical.bowl2018_dataset import BOWL2018Dataset
        return BOWL2018Dataset
    elif name == 'medical/chasedb1':
        from .medical.chasedb1_dataset import ChaseDB1Dataset
        return ChaseDB1Dataset
    elif name == 'mnist':
        from .mnist_dataset import MNISTDataset
        return MNISTDataset
    elif name == 'medical/vqarad':
        from .medical.vqa_rad_dataset import VQARadDataset
        return VQARadDataset
    elif name == 'medical/pathvqa':
        from .medical.path_vqa_dataset import PathVQADataset
        return PathVQADataset
    elif name == 'medical/isic2016_reasoning_seg':
        from .medical.isic2016_reasoning_seg_dataset import ISIC2016ReasoningSegDataset
        return ISIC2016ReasoningSegDataset
    elif name == 'brainmri_clip':
        from .medical.brain_mri_clip_dataset import BrainMRIClipDataset
        return BrainMRIClipDataset
    elif name == 'coco':
        from .coco_dataset import COCODataset
        return COCODataset
    elif name == 'coco_segmentation' or name == 'coco_seg':
        from .coco_dataset import COCOSegmentationDataset
        return COCOSegmentationDataset
    elif name == 'cifar' or name == 'cifar10':
        from .cifar_dataset import CIFAR10Dataset
        return CIFAR10Dataset
    elif name == 'cifar100':
        from .cifar_dataset import CIFAR100Dataset
        return CIFAR100Dataset
    elif name == 'imagenet' or name == 'imagenet1k' or name == 'imagenet-1k':
        from .imagenet_dataset import ImageNet1KDataset
        return ImageNet1KDataset
    elif name == 'diffusion/mnist' or name == 'mnist_diffusion':
        # 返回包装函数而不是类
        return lambda root_dir, conditional=False, **kwargs: get_mnist_diffusion_dataset(
            root_dir, split='train', conditional=conditional, **kwargs
        )
    elif name == 'diffusion/cifar10' or name == 'cifar10_diffusion':
        return lambda root_dir, conditional=False, **kwargs: get_cifar10_diffusion_dataset(
            root_dir, split='train', conditional=conditional, **kwargs
        )
    elif name == 'diffusion/imagenet' or name == 'imagenet_diffusion':
        return lambda root_dir, conditional=True, **kwargs: get_imagenet_diffusion_dataset(
            root_dir, split='train', conditional=conditional, **kwargs
        )
    elif name == 'diffusion/unconditional':
        from .diffusion_dataset import UnconditionalDiffusionDataset
        return UnconditionalDiffusionDataset
    elif name == 'diffusion/conditional':
        from .diffusion_dataset import ConditionalDiffusionDataset
        return ConditionalDiffusionDataset
    elif name == 'diffusion/text_to_image' or name == 'text2image' or name == 't2i':
        from .diffusion_dataset import TextToImageDiffusionDataset
        return TextToImageDiffusionDataset
    else:
        logging.warning(f'No target dataset: {dataset_name}')
        return None


def get_train_dataset(dataset_name: str, root_dir: Path, **kwargs):
    """获取训练数据集"""
    if dataset_name.lower() == 'cholect50':
        return
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    return DatasetClass.get_train_dataset(root_dir, **kwargs)


def get_valid_dataset(dataset_name: str, root_dir: Path, **kwargs):
    """获取验证数据集"""
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    return DatasetClass.get_valid_dataset(root_dir, **kwargs)


def get_test_dataset(dataset_name: str, root_dir: Path, **kwargs):
    """获取测试数据集"""
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    return DatasetClass.get_test_dataset(root_dir, **kwargs)

def get_dataset(mode: str):
    c_dataset = get_config_value('dataset')
    assert c_dataset is not None, "Dataset configuration is not defined in the config file."

    # 检查是否为增强版混合数据集配置
    if c_dataset.get('type') == 'hybrid':
        logging.info(f"检测到混合数据集配置，创建HybridDataset for mode: {mode}")
        # 传递完整的配置给 create_hybrid_dataset_from_config
        full_config = get_config()
        return create_hybrid_dataset_from_config(c_dataset, mode, full_config)

    # 传统单数据集模式
    config = c_dataset.get('config', {})
    match mode:
        case 'train':
            dataset0 = get_train_dataset(c_dataset['name'], Path(c_dataset['root_dir']), **config)
        case 'test':
            dataset0 = get_test_dataset(c_dataset['name'], Path(c_dataset['root_dir']), **config)
        case 'valid' | 'val':
            dataset0 = get_valid_dataset(c_dataset['name'], Path(c_dataset['root_dir']), **config)
        case _:
            DatasetClass = _get_dataset_by_case(c_dataset['name'])
            if DatasetClass is None:
                return None
            dataset0 = DatasetClass.get_dataset(Path(c_dataset['root_dir']), **config)

    if not dataset0:
        logging.error(f"{c_dataset['name']} is empty!")

    return dataset0


def random_sample(dataset: CustomDataset, sample_ratio: float = 0.1, num_samples: int=0, *, generator=None, output_dataloader: bool = True, batch_size: int = 1):
    """随机采样数据集的一部分样本索引"""
    from torch.utils.data import RandomSampler
    if num_samples <= 0:
        num_samples = int(sample_ratio * len(dataset))
    elif num_samples > len(dataset):
        num_samples = len(dataset)
    sampler = RandomSampler(dataset, num_samples=num_samples, generator=generator)
    if output_dataloader:
        c = get_config()
        num_workers = c["dataloader"]["num_workers"]
        if c["dataloader"]["shuffle"]:
            print('Random sampler used, set shuffle to False')
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, shuffle=False)
    else:
        return sampler


def get_train_valid_test_dataloader(use_valid=False):
    """根据全局配置构建训练/验证/测试的DataLoader
    
    增强功能：
    - 支持 train_valid_test_split 配置驱动的数据集分割
    - 支持 kfold 配置驱动的交叉验证（返回首折的train/valid/test）
    
    配置格式:
        [dataset.split]
        type = "train_valid_test_split" # 或 "kfold"
        train = 0.8
        valid = 0.1  
        test = 0.1
        shuffle = true
        random_state = 42
        
        # 仅当 type = "kfold" 时生效
        n_splits = 5
        include_test = false
        test_ratio = 0.0
    """
    # 检查是否启用了配置驱动的数据集分割
    split_config = get_config_value('dataset.split', default=None)
    
    if split_config is not None:
        # 配置驱动模式：使用split配置进行数据集分割
        return _get_dataloader_with_split_config(split_config, use_valid)
    else:
        # 传统模式：直接获取预定义的train/valid/test数据集
        return _get_dataloader_traditional_mode(use_valid)


def _get_dataloader_traditional_mode(use_valid: bool):
    """传统模式：直接获取预定义的train/valid/test数据集并构建DataLoader"""
    c = get_config()

    train_dataset = get_dataset('train')
    test_dataset = get_dataset('test')
    num_workers = c["dataloader"]["num_workers"]
    shuffle = c["dataloader"]["shuffle"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=c["train"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=c["test"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    if use_valid and "valid" in c:
        valid_dataset = get_dataset('valid')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )

        return train_loader, valid_loader, test_loader

    return train_loader, None, test_loader


def _get_dataloader_with_split_config(split_config: dict, use_valid: bool):
    """配置驱动模式：根据split配置进行数据集分割并构建DataLoader"""
    c = get_config()
    
    split_type = split_config.get('type', 'train_valid_test_split')
    
    if split_type == 'train_valid_test_split':
        return _get_dataloader_with_train_valid_test_split(split_config, use_valid)
    elif split_type == 'kfold':
        return _get_dataloader_with_kfold(split_config, use_valid)
    else:
        raise ValueError(f"不支持的split类型: {split_type}，仅支持 'train_valid_test_split' 或 'kfold'")


def _get_dataloader_with_train_valid_test_split(split_config: dict, use_valid: bool):
    """基于train_valid_test_split配置进行数据集分割"""
    c = get_config()
    
    # 获取完整数据集（通常使用train模式获取最大的数据集）
    full_dataset = get_dataset('train')
    if not full_dataset:
        logging.error("无法获取完整数据集进行分割")
        return None, None, None
    
    # 解析分割参数
    train_ratio = split_config.get('train', 0.8)
    valid_ratio = split_config.get('valid', 0.1) 
    test_ratio = split_config.get('test', 0.1)
    shuffle = split_config.get('shuffle', True)
    random_state = split_config.get('random_state', None)
    
    # 执行数据集分割
    train_subset, valid_subset, test_subset = full_dataset.train_valid_test_split(
        train=train_ratio,
        valid=valid_ratio, 
        test=test_ratio,
        shuffle=shuffle,
        random_state=random_state
    )
    
    # 构建DataLoader
    num_workers = c["dataloader"]["num_workers"]
    dataloader_shuffle = c["dataloader"]["shuffle"]
    
    train_loader = DataLoader(
        train_subset,
        batch_size=c["train"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=dataloader_shuffle,
    ) if train_subset else None
    
    test_loader = DataLoader(
        test_subset,
        batch_size=c["test"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=dataloader_shuffle,
    ) if test_subset else None
    
    if use_valid and valid_subset:
        valid_loader = DataLoader(
            valid_subset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
        )
        return train_loader, valid_loader, test_loader
    
    return train_loader, None, test_loader


def _get_dataloader_with_kfold(split_config: dict, use_valid: bool):
    """基于kfold配置进行交叉验证分割（返回第一折的结果）
    
    注意：该函数仅返回第一折的DataLoader，完整的kfold训练循环需要在上层实现
    """
    c = get_config()
    
    # 获取完整数据集
    full_dataset = get_dataset('train')
    if not full_dataset:
        logging.error("无法获取完整数据集进行kfold分割")
        return None, None, None
    
    # 解析kfold参数
    n_splits = split_config.get('n_splits', 5)
    shuffle = split_config.get('shuffle', True)
    random_state = split_config.get('random_state', None)
    include_test = split_config.get('include_test', False)
    test_ratio = split_config.get('test_ratio', 0.0)
    
    # 执行kfold分割
    fold_results = CustomDataset.kfold(
        dataset=full_dataset,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        include_test=include_test,
        test_ratio=test_ratio
    )
    
    if not fold_results:
        logging.error("kfold分割失败")
        return None, None, None
    
    # 使用第一折的结果
    train_subset, valid_subset, test_subset = fold_results[0]
    
    # 构建DataLoader
    num_workers = c["dataloader"]["num_workers"]
    dataloader_shuffle = c["dataloader"]["shuffle"]
    
    train_loader = DataLoader(
        train_subset,
        batch_size=c["train"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=dataloader_shuffle,
    ) if train_subset else None
    
    test_loader = DataLoader(
        test_subset,
        batch_size=c["test"]["batch_size"],
        pin_memory=True,
        num_workers=num_workers,
        shuffle=dataloader_shuffle,
    ) if test_subset else None
    
    if use_valid and valid_subset:
        valid_loader = DataLoader(
            valid_subset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
        )
        return train_loader, valid_loader, test_loader
    
    return train_loader, None, test_loader


def get_kfold_dataloaders(
    n_splits: Optional[int] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None,
    include_test: Optional[bool] = None,
    test_ratio: Optional[float] = None
) -> List[Tuple[DataLoader, DataLoader, Optional[DataLoader]]]:
    """获取所有折的DataLoader列表，用于完整的kfold交叉验证训练
    
    参数优先级：函数参数 > 配置文件 > 默认值
    
    返回:
        List[Tuple[train_loader, valid_loader, test_loader]]，长度为n_splits
    """
    c = get_config()
    split_config = get_config_value('dataset.split', default={})
    
    # 参数合并：函数参数 > 配置文件 > 默认值
    final_n_splits = n_splits if n_splits is not None else split_config.get('n_splits', 5)
    final_shuffle = shuffle if shuffle is not None else split_config.get('shuffle', True)
    final_random_state = random_state if random_state is not None else split_config.get('random_state', None)
    final_include_test = include_test if include_test is not None else split_config.get('include_test', False)
    final_test_ratio = test_ratio if test_ratio is not None else split_config.get('test_ratio', 0.0)
    
    # 获取完整数据集
    full_dataset = get_dataset('train')
    if not full_dataset:
        logging.error("无法获取完整数据集进行kfold分割")
        return []
    
    # 执行kfold分割
    fold_results = CustomDataset.kfold(
        dataset=full_dataset,
        n_splits=final_n_splits,
        shuffle=final_shuffle,
        random_state=final_random_state,
        include_test=final_include_test,
        test_ratio=final_test_ratio
    )
    
    # 为每一折构建DataLoader
    dataloaders = []
    num_workers = c["dataloader"]["num_workers"]
    dataloader_shuffle = c["dataloader"]["shuffle"]
    
    for train_subset, valid_subset, test_subset in fold_results:
        train_loader = DataLoader(
            train_subset,
            batch_size=c["train"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
        ) if train_subset else None
        
        valid_loader = DataLoader(
            valid_subset,
            batch_size=c["valid"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
        ) if valid_subset else None
        
        test_loader = DataLoader(
            test_subset,
            batch_size=c["test"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            shuffle=dataloader_shuffle,
        ) if test_subset else None
        
        dataloaders.append((train_loader, valid_loader, test_loader))
    
    return dataloaders


def support_datasets(task_type: str) -> List[str]:
    """查询支持指定任务类型的数据集列表
    
    Args:
        task_type: 任务类型，如 'classification', 'segmentation', 'detection' 等
        
    Returns:
        支持该任务类型的数据集名称列表
        
    Examples:
        >>> datasets = support_datasets('classification')
        >>> print(datasets)
        ['mnist', 'cifar10', 'cifar100', 'imagenet', 'medical/isic2017', 'medical/isic2018']
    """
    task_type_lower = task_type.lower()
    matching_datasets = []
    
    for dataset_name, info in DATASET_REGISTRY.items():
        if task_type_lower in [t.lower() for t in info['task_type']]:
            matching_datasets.append(dataset_name)
    
    return sorted(matching_datasets)


def supported_tasks(dataset_name: str) -> List[str]:
    """查询指定数据集支持的任务类型列表
    
    Args:
        dataset_name: 数据集名称或别名
        
    Returns:
        该数据集支持的任务类型列表，如果数据集不存在则返回空列表
        
    Examples:
        >>> tasks = supported_tasks('coco')
        >>> print(tasks)
        ['detection', 'instance_segmentation', 'keypoint_detection', 'captioning']
    """
    info = get_dataset_info(dataset_name)
    if info:
        return info['task_types']
    return []


def get_datasets_by_task(task_type: str) -> List[Dict[str, any]]:
    """根据任务类型获取数据集详细信息列表（保留用于向后兼容）
    
    Args:
        task_type: 任务类型
        
    Returns:
        匹配的数据集详细信息列表
    """
    task_type_lower = task_type.lower()
    matching_datasets = []
    
    for dataset_name, info in DATASET_REGISTRY.items():
        if task_type_lower in [t.lower() for t in info['task_type']]:
            matching_datasets.append({
                'name': dataset_name,
                'class': info['class'],
                'task_types': info['task_type'],
                'description': info['description'],
                'aliases': info['aliases']
            })
    
    return matching_datasets


def get_all_task_types() -> List[str]:
    """获取所有可用的任务类型
    
    Returns:
        所有任务类型的列表（去重且排序）
    """
    all_tasks = set()
    for info in DATASET_REGISTRY.values():
        all_tasks.update(info['task_type'])
    
    return sorted(list(all_tasks))


def get_dataset_info(dataset_name: str) -> Optional[Dict[str, any]]:
    """获取指定数据集的详细信息
    
    Args:
        dataset_name: 数据集名称或别名
        
    Returns:
        数据集信息字典，如果未找到则返回 None
    """
    dataset_name_lower = dataset_name.lower()
    
    # 直接查找
    if dataset_name_lower in DATASET_REGISTRY:
        info = DATASET_REGISTRY[dataset_name_lower]
        return {
            'name': dataset_name_lower,
            'class': info['class'],
            'module': info['module'],
            'task_types': info['task_type'],
            'description': info['description'],
            'aliases': info['aliases']
        }
    
    # 通过别名查找
    for main_name, info in DATASET_REGISTRY.items():
        if dataset_name_lower in [alias.lower() for alias in info['aliases']]:
            return {
                'name': main_name,
                'class': info['class'],
                'module': info['module'],
                'task_types': info['task_type'],
                'description': info['description'],
                'aliases': info['aliases']
            }
    
    return None


def list_all_datasets(task_type: Optional[str] = None, 
                     sort_by: Literal['name', 'task'] = 'name') -> Dict[str, List[Dict]]:
    """列出所有可用的数据集
    
    Args:
        task_type: 可选的任务类型过滤器
        sort_by: 排序方式，'name' 按名称排序，'task' 按任务类型分组
        
    Returns:
        数据集信息字典
    """
    if task_type:
        datasets = get_datasets_by_task(task_type)
        return {task_type: datasets}
    
    if sort_by == 'name':
        all_datasets = []
        for dataset_name, info in sorted(DATASET_REGISTRY.items()):
            all_datasets.append({
                'name': dataset_name,
                'class': info['class'],
                'task_types': info['task_type'],
                'description': info['description'],
                'aliases': info['aliases']
            })
        return {'all': all_datasets}
    
    elif sort_by == 'task':
        # 按任务类型分组
        task_groups = {}
        for dataset_name, info in DATASET_REGISTRY.items():
            for task in info['task_type']:
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append({
                    'name': dataset_name,
                    'class': info['class'],
                    'task_types': info['task_type'],
                    'description': info['description'],
                    'aliases': info['aliases']
                })
        
        # 排序
        return {k: sorted(v, key=lambda x: x['name']) 
                for k, v in sorted(task_groups.items())}
    
    return {}


def print_dataset_task_matrix():
    """打印数据集-任务类型矩阵，显示各数据集支持的任务"""
    print("\n" + "="*100)
    print("数据集任务支持矩阵".center(100))
    print("="*100)
    
    # 获取所有任务类型
    all_tasks = get_all_task_types()
    
    # 打印表头
    print(f"\n{'数据集名称':<35}", end="")
    for task in all_tasks[:10]:  # 只显示前10个任务以适应宽度
        print(f"{task[:12]:<13}", end="")
    print()
    print("-"*100)
    
    # 打印每个数据集
    for dataset_name in sorted(DATASET_REGISTRY.keys()):
        info = DATASET_REGISTRY[dataset_name]
        print(f"{dataset_name:<35}", end="")
        for task in all_tasks[:10]:
            mark = "✓" if task in info['task_type'] else " "
            print(f"{mark:^13}", end="")
        print(f"  {', '.join(info['task_type'][:2])}")
    
    print("-"*100)
    print(f"\n总计: {len(DATASET_REGISTRY)} 个数据集, {len(all_tasks)} 种任务类型\n")


def create_hybrid_dataset(
    dataset_names: List[str],
    root_dirs: Optional[Union[List[Path], List[str]]] = None,
    split: str = 'train',
    ratios: Optional[List[float]] = None,
    priorities: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    shuffle_order: bool = False,
    random_seed: Optional[int] = None,
    **dataset_kwargs
) -> HybridDataset:
    """创建混合数据集（使用 HybridDataset）
    
    注意：默认情况下（不设置 ratios/priorities/weights），行为与 HybridDataset 一致。
    
    Args:
        dataset_names: 数据集名称列表
        root_dirs: 数据集根目录列表（支持 Path 或 str），如果为 None 则需要在 dataset_kwargs 中指定
        split: 数据集划分 ('train', 'val', 'test')
        ratios: 每个数据集的采样比例（默认 None，按原始长度比例）
        priorities: 数据集优先级（默认 None，按输入顺序）
        weights: 数据集权重（默认 None，均等权重）
        shuffle_order: 是否随机打乱数据集顺序（默认 False）
        random_seed: 随机种子（默认 None）
        **dataset_kwargs: 每个数据集的额外参数，格式为 {dataset_name: {...}}
        
    Returns:
        HybridDataset 实例
        
    Examples:
        >>> # 基本用法（等同于 HybridDataset）
        >>> hybrid_ds = create_hybrid_dataset(
        ...     dataset_names=['mnist', 'cifar10'],
        ...     root_dirs=['data/mnist', 'data/cifar10'],  # 支持 str
        ...     split='train'
        ... )
        
        >>> # 高级用法（使用 HybridDataset 特性）
        >>> hybrid_ds = create_hybrid_dataset(
        ...     dataset_names=['mnist', 'cifar10'],
        ...     root_dirs=[Path('data/mnist'), Path('data/cifar10')],  # 也支持 Path
        ...     split='train',
        ...     ratios=[0.3, 0.7],
        ...     priorities=[1, 2],
        ...     weights=[1.0, 1.0]
        ... )
        
        >>> # 医学影像混合数据集
        >>> hybrid_ds = create_hybrid_dataset(
        ...     dataset_names=['drive', 'medical/stare', 'medical/chasedb1'],
        ...     root_dirs=['data/drive', 'data/stare', 'data/chasedb1'],
        ...     split='train',
        ...     priorities=[1, 2, 3],
        ...     weights=[1.0, 1.0, 0.5]
        ... )
    """
    from pathlib import Path
    
    if not dataset_names:
        raise ValueError("数据集名称列表不能为空")
    
    # 如果未提供 root_dirs，尝试从 dataset_kwargs 中获取
    if root_dirs is None:
        root_dirs = []
        for ds_name in dataset_names:
            if ds_name in dataset_kwargs and 'root_dir' in dataset_kwargs[ds_name]:
                root_dir = dataset_kwargs[ds_name]['root_dir']
                # 转换为 Path 对象
                root_dirs.append(Path(root_dir) if isinstance(root_dir, str) else root_dir)
            else:
                raise ValueError(f"未提供数据集 {ds_name} 的 root_dir")
    else:
        # 确保所有 root_dirs 都是 Path 对象
        root_dirs = [Path(rd) if isinstance(rd, str) else rd for rd in root_dirs]
    
    if len(dataset_names) != len(root_dirs):
        raise ValueError("数据集名称和根目录的数量必须相同")
    
    # 创建数据集列表
    datasets = []
    for i, (ds_name, root_dir) in enumerate(zip(dataset_names, root_dirs)):
        # 获取该数据集的特定参数
        ds_specific_kwargs = dataset_kwargs.get(ds_name, {})
        
        # 根据 split 获取对应的数据集
        if split == 'train':
            dataset = get_train_dataset(ds_name, root_dir, **ds_specific_kwargs)
        elif split in ['val', 'valid']:
            dataset = get_valid_dataset(ds_name, root_dir, **ds_specific_kwargs)
        elif split == 'test':
            dataset = get_test_dataset(ds_name, root_dir, **ds_specific_kwargs)
        else:
            raise ValueError(f"不支持的 split: {split}")
        
        if dataset is None:
            raise ValueError(f"无法创建数据集: {ds_name}")
        
        datasets.append(dataset)
    
    # 创建混合数据集
    return HybridDataset(
        datasets=datasets,
        ratios=ratios,
        priorities=priorities,
        weights=weights,
        shuffle_order=shuffle_order,
        random_seed=random_seed
    )


def create_hybrid_dataset_by_task(
    task_type: str,
    root_dir_mapping: Dict[str, Union[Path, str]],
    split: str = 'train',
    max_datasets: Optional[int] = None,
    ratios: Optional[List[float]] = None,
    **kwargs
) -> HybridDataset:
    """根据任务类型创建混合数据集（使用 HybridDataset）
    
    自动查找支持指定任务类型的所有数据集，并创建混合数据集。
    
    Args:
        task_type: 任务类型，如 'classification', 'segmentation'
        root_dir_mapping: 数据集名称到根目录的映射字典（支持 Path 或 str）
        split: 数据集划分
        max_datasets: 最多使用的数据集数量
        ratios: 数据集采样比例
        **kwargs: 传递给 create_hybrid_dataset 的其他参数
        
    Returns:
        HybridDataset 实例
        
    Examples:
        >>> # 创建所有分类数据集的混合数据集（支持 str）
        >>> hybrid_ds = create_hybrid_dataset_by_task(
        ...     task_type='classification',
        ...     root_dir_mapping={
        ...         'mnist': 'data/mnist',
        ...         'cifar10': 'data/cifar10',
        ...         'imagenet': 'data/imagenet'
        ...     },
        ...     split='train',
        ...     max_datasets=2
        ... )
        
        >>> # 创建所有医学分割数据集的混合数据集（也支持 Path）
        >>> hybrid_ds = create_hybrid_dataset_by_task(
        ...     task_type='segmentation',
        ...     root_dir_mapping={
        ...         'drive': Path('data/drive'),
        ...         'medical/stare': Path('data/stare')
        ...     },
        ...     split='train'
        ... )
    """
    # 获取支持该任务类型的所有数据集
    available_datasets = support_datasets(task_type)
    
    # 过滤出在 root_dir_mapping 中的数据集
    dataset_names = [ds for ds in available_datasets if ds in root_dir_mapping]
    
    if not dataset_names:
        raise ValueError(f"没有找到支持 '{task_type}' 任务且在 root_dir_mapping 中的数据集")
    
    # 限制数据集数量
    if max_datasets and len(dataset_names) > max_datasets:
        dataset_names = dataset_names[:max_datasets]
    
    # 创建 root_dirs 列表（支持 str 和 Path）
    root_dirs = [root_dir_mapping[ds] for ds in dataset_names]
    
    logging.info(f"为任务 '{task_type}' 创建混合数据集，包含 {len(dataset_names)} 个数据集: {dataset_names}")
    
    # 创建混合数据集
    return create_hybrid_dataset(
        dataset_names=dataset_names,
        root_dirs=root_dirs,
        split=split,
        ratios=ratios,
        **kwargs
    )


def get_hybrid_dataset_info(dataset_names: List[str]) -> Dict:
    """获取混合数据集的信息
    
    Args:
        dataset_names: 数据集名称列表
        
    Returns:
        混合数据集信息字典
    """
    all_tasks = set()
    all_descriptions = []
    
    for ds_name in dataset_names:
        info = get_dataset_info(ds_name)
        if info:
            all_tasks.update(info['task_types'])
            all_descriptions.append(f"{ds_name}: {info['description']}")
    
    return {
        'dataset_names': dataset_names,
        'num_datasets': len(dataset_names),
        'combined_tasks': sorted(list(all_tasks)),
        'descriptions': all_descriptions
    }


def get_diffusion_train_dataset(
    dataset_name: str,
    root_dir: Path,
    conditional: bool = False,
    **kwargs
) -> DiffusionDataset:
    """获取Diffusion训练数据集
    
    Args:
        dataset_name: 数据集名称（如 'diffusion/mnist', 'diffusion/cifar10', 'mnist', 'cifar10'等）
        root_dir: 数据集根目录
        conditional: 是否为条件生成模式
        **kwargs: 其他参数
    
    Returns:
        DiffusionDataset实例
    
    Examples:
        >>> # 无条件MNIST生成
        >>> ds = get_diffusion_train_dataset('mnist', Path('data/mnist'), conditional=False)
        >>> 
        >>> # 条件CIFAR-10生成
        >>> ds = get_diffusion_train_dataset('cifar10', Path('data/cifar10'), conditional=True)
        >>> 
        >>> # 使用diffusion/前缀
        >>> ds = get_diffusion_train_dataset('diffusion/mnist', Path('data/mnist'))
    """
    name = dataset_name.lower()
    
    # 处理diffusion/前缀
    if name.startswith('diffusion/'):
        base_name = name.replace('diffusion/', '')
    else:
        base_name = name
    
    # 根据基础数据集名称选择对应的diffusion包装器
    if base_name in ['mnist', 'mnist_diffusion']:
        return get_mnist_diffusion_dataset(root_dir, 'train', conditional, **kwargs)
    elif base_name in ['cifar10', 'cifar', 'cifar10_diffusion']:
        return get_cifar10_diffusion_dataset(root_dir, 'train', conditional, **kwargs)
    elif base_name in ['imagenet', 'imagenet1k', 'imagenet-1k', 'imagenet_diffusion']:
        return get_imagenet_diffusion_dataset(root_dir, 'train', conditional, **kwargs)
    else:
        # 尝试获取基础数据集并包装为diffusion数据集
        base_dataset = get_train_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)


def get_diffusion_valid_dataset(
    dataset_name: str,
    root_dir: Path,
    conditional: bool = False,
    **kwargs
) -> DiffusionDataset:
    """获取Diffusion验证数据集"""
    name = dataset_name.lower()
    
    if name.startswith('diffusion/'):
        base_name = name.replace('diffusion/', '')
    else:
        base_name = name
    
    if base_name in ['mnist', 'mnist_diffusion']:
        return get_mnist_diffusion_dataset(root_dir, 'valid', conditional, **kwargs)
    elif base_name in ['cifar10', 'cifar', 'cifar10_diffusion']:
        return get_cifar10_diffusion_dataset(root_dir, 'valid', conditional, **kwargs)
    elif base_name in ['imagenet', 'imagenet1k', 'imagenet-1k', 'imagenet_diffusion']:
        return get_imagenet_diffusion_dataset(root_dir, 'valid', conditional, **kwargs)
    else:
        base_dataset = get_valid_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)


def get_diffusion_test_dataset(
    dataset_name: str,
    root_dir: Path,
    conditional: bool = False,
    **kwargs
) -> DiffusionDataset:
    """获取Diffusion测试数据集"""
    name = dataset_name.lower()
    
    if name.startswith('diffusion/'):
        base_name = name.replace('diffusion/', '')
    else:
        base_name = name
    
    if base_name in ['mnist', 'mnist_diffusion']:
        return get_mnist_diffusion_dataset(root_dir, 'test', conditional, **kwargs)
    elif base_name in ['cifar10', 'cifar', 'cifar10_diffusion']:
        return get_cifar10_diffusion_dataset(root_dir, 'test', conditional, **kwargs)
    elif base_name in ['imagenet', 'imagenet1k', 'imagenet-1k', 'imagenet_diffusion']:
        return get_imagenet_diffusion_dataset(root_dir, 'test', conditional, **kwargs)
    else:
        base_dataset = get_test_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)
