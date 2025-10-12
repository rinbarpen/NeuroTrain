import logging
from pathlib import Path
from typing import Literal, Optional, List, Tuple, Union
from torch.utils.data import DataLoader, Subset


from src.config import get_config_value, get_config
# from src.utils.transform import get_transforms
from .custom_dataset import CustomDataset
from .enhanced_hybrid_dataset import EnhancedHybridDataset, create_enhanced_hybrid_dataset_from_config


def _get_dataset_by_case(dataset_name: str):
    """根据数据集名称返回对应的数据集类"""
    name = dataset_name.lower()
    if name == 'medical/mri_brain_clip':
        from .medical.mri_brain_clip_dataset import MriBrainClipDataset
        return MriBrainClipDataset
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
        from .medical.vqa_rad_dataset import VQARADDataset
        return VQARADDataset
    elif name == 'medical/pathvqa':
        from .medical.pathvqa_dataset import PathVQADataset
        return PathVQADataset
    elif name == 'medical/isic2016_reasoning_seg':
        from .medical.isic2016_reasoning_seg_dataset import ISIC2016ReasoningSegDataset
        return ISIC2016ReasoningSegDataset
    elif name == 'brainmri_clip':
        from .medical.brain_mri_clip_dataset import BrainMRIClipDataset
        return BrainMRIClipDataset
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
    if c_dataset.get('type') == 'enhanced_hybrid':
        logging.info(f"检测到增强版混合数据集配置，创建EnhancedHybridDataset for mode: {mode}")
        # 传递完整的配置给 create_enhanced_hybrid_dataset_from_config
        full_config = get_config()
        return create_enhanced_hybrid_dataset_from_config(c_dataset, mode, full_config)
    
    # 检查是否为混合数据集配置
    if 'hybrid' in c_dataset and c_dataset['hybrid'].get('enabled', False):
        logging.info(f"检测到混合数据集配置，创建EnhancedHybridDataset for mode: {mode}")
        # 传递完整的配置给 create_enhanced_hybrid_dataset_from_config
        full_config = get_config()
        return create_enhanced_hybrid_dataset_from_config(c_dataset, mode, full_config)

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
