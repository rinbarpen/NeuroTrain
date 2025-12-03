import logging
from pathlib import Path
from typing import Any, Literal, Optional, List, Tuple, Union, Dict, Sequence
from torch.utils.data import DataLoader


from src.config import get_config_value, get_config

# from src.utils.transform import get_transforms
from .hybrid_dataset import HybridDataset, create_hybrid_dataset_from_config
from .diffusion_dataset import (
    DiffusionDataset,
    create_diffusion_dataset,
    get_mnist_diffusion_dataset,
    get_cifar10_diffusion_dataset,
    get_imagenet_diffusion_dataset,
)


logger = logging.getLogger(__name__)


# 数据集注册表：记录所有可用数据集及其信息
DATASET_REGISTRY = {
    "medical/mri_brain_clip": {
        "class": "MriBrainClipDataset",
        "module": "medical.mri_brain_clip_dataset",
        "task_type": ["multimodal", "medical_imaging"],
        "description": "MRI Brain CLIP dataset for multimodal learning",
        "aliases": ["medical/mri_brain_clip"],
    },
    "medical/btcv": {
        "class": "BTCVDataset",
        "module": "medical.btcv_dataset",
        "task_type": ["segmentation", "medical_imaging", "3d_segmentation"],
        "description": "BTCV multi-organ segmentation dataset",
        "aliases": ["medical/btcv"],
    },
    "drive": {
        "class": "DriveDataset",
        "module": "drive_dataset",
        "task_type": ["segmentation", "medical_imaging", "retinal_vessel_segmentation"],
        "description": "DRIVE retinal vessel segmentation dataset",
        "aliases": ["drive"],
    },
    "medical/stare": {
        "class": "StareDataset",
        "module": "medical.stare_dataset",
        "task_type": ["segmentation", "medical_imaging", "retinal_vessel_segmentation"],
        "description": "STARE retinal vessel segmentation dataset",
        "aliases": ["medical/stare"],
    },
    "medical/isic2016": {
        "class": "ISIC2016Dataset",
        "module": "medical.isic2016_dataset",
        "task_type": ["segmentation", "medical_imaging", "skin_lesion_segmentation"],
        "description": "ISIC 2016 skin lesion segmentation dataset",
        "aliases": ["medical/isic2016"],
    },
    "medical/isic2017": {
        "class": "ISIC2017Dataset",
        "module": "medical.isic2017_dataset",
        "task_type": [
            "classification",
            "medical_imaging",
            "skin_lesion_classification",
        ],
        "description": "ISIC 2017 skin lesion classification dataset",
        "aliases": ["medical/isic2017"],
    },
    "medical/isic2018": {
        "class": "ISIC2018Dataset",
        "module": "medical.isic2018_dataset",
        "task_type": ["segmentation", "classification", "medical_imaging"],
        "description": "ISIC 2018 skin lesion analysis dataset",
        "aliases": ["medical/isic2018"],
    },
    "medical/bowl2018": {
        "class": "BOWL2018Dataset",
        "module": "medical.bowl2018_dataset",
        "task_type": ["segmentation", "medical_imaging", "cell_segmentation"],
        "description": "Data Science Bowl 2018 nuclei segmentation dataset",
        "aliases": ["medical/bowl2018"],
    },
    "medical/chasedb1": {
        "class": "ChaseDB1Dataset",
        "module": "medical.chasedb1_dataset",
        "task_type": ["segmentation", "medical_imaging", "retinal_vessel_segmentation"],
        "description": "CHASE-DB1 retinal vessel segmentation dataset",
        "aliases": ["medical/chasedb1"],
    },
    "mnist": {
        "class": "MNISTDataset",
        "module": "mnist_dataset",
        "task_type": ["classification", "digit_recognition"],
        "description": "MNIST handwritten digit classification dataset",
        "aliases": ["mnist"],
    },
    "medical/vqarad": {
        "class": "VQARADDataset",
        "module": "medical.vqa_rad_dataset",
        "task_type": ["vqa", "medical_imaging", "multimodal"],
        "description": "VQA-RAD medical visual question answering dataset",
        "aliases": ["medical/vqarad"],
    },
    "medical/pathvqa": {
        "class": "PathVQADataset",
        "module": "medical.pathvqa_dataset",
        "task_type": ["vqa", "medical_imaging", "multimodal"],
        "description": "PathVQA pathology visual question answering dataset",
        "aliases": ["medical/pathvqa"],
    },
    "medical/isic2016_reasoning_seg": {
        "class": "ISIC2016ReasoningSegDataset",
        "module": "medical.isic2016_reasoning_seg_dataset",
        "task_type": ["segmentation", "medical_imaging", "reasoning"],
        "description": "ISIC 2016 with reasoning for segmentation",
        "aliases": ["medical/isic2016_reasoning_seg"],
    },
    "brainmri_clip": {
        "class": "BrainMRIClipDataset",
        "module": "medical.brain_mri_clip_dataset",
        "task_type": ["multimodal", "medical_imaging"],
        "description": "Brain MRI CLIP dataset for multimodal learning",
        "aliases": ["brainmri_clip"],
    },
    "coco": {
        "class": "COCODataset",
        "module": "coco_dataset",
        "task_type": [
            "detection",
            "instance_segmentation",
            "keypoint_detection",
            "captioning",
        ],
        "description": "COCO dataset for detection, segmentation, keypoints, and captions",
        "aliases": ["coco"],
    },
    "coco_segmentation": {
        "class": "COCOSegmentationDataset",
        "module": "coco_dataset",
        "task_type": ["semantic_segmentation", "segmentation"],
        "description": "COCO semantic segmentation dataset",
        "aliases": ["coco_segmentation", "coco_seg"],
    },
    "cifar10": {
        "class": "CIFAR10Dataset",
        "module": "cifar_dataset",
        "task_type": ["classification", "image_classification"],
        "description": "CIFAR-10 image classification dataset",
        "aliases": ["cifar", "cifar10"],
    },
    "cifar100": {
        "class": "CIFAR100Dataset",
        "module": "cifar_dataset",
        "task_type": ["classification", "image_classification"],
        "description": "CIFAR-100 image classification dataset",
        "aliases": ["cifar100"],
    },
    "imagenet": {
        "class": "ImageNet1KDataset",
        "module": "imagenet_dataset",
        "task_type": ["classification", "image_classification"],
        "description": "ImageNet-1K large-scale image classification dataset",
        "aliases": ["imagenet", "imagenet1k", "imagenet-1k"],
    },
    "diffusion/mnist": {
        "class": "DiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": [
            "generation",
            "diffusion",
            "unconditional_generation",
            "conditional_generation",
        ],
        "description": "MNIST dataset for diffusion models (unconditional/conditional)",
        "aliases": ["diffusion/mnist", "mnist_diffusion"],
    },
    "diffusion/cifar10": {
        "class": "DiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": [
            "generation",
            "diffusion",
            "unconditional_generation",
            "conditional_generation",
        ],
        "description": "CIFAR-10 dataset for diffusion models (unconditional/conditional)",
        "aliases": ["diffusion/cifar10", "cifar10_diffusion"],
    },
    "diffusion/imagenet": {
        "class": "DiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": ["generation", "diffusion", "conditional_generation"],
        "description": "ImageNet dataset for conditional diffusion models",
        "aliases": ["diffusion/imagenet", "imagenet_diffusion"],
    },
    "diffusion/unconditional": {
        "class": "UnconditionalDiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": ["generation", "diffusion", "unconditional_generation"],
        "description": "Generic unconditional diffusion dataset wrapper",
        "aliases": ["diffusion/unconditional"],
    },
    "diffusion/conditional": {
        "class": "ConditionalDiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": ["generation", "diffusion", "conditional_generation"],
        "description": "Generic conditional diffusion dataset wrapper",
        "aliases": ["diffusion/conditional"],
    },
    "diffusion/text_to_image": {
        "class": "TextToImageDiffusionDataset",
        "module": "diffusion_dataset",
        "task_type": ["generation", "diffusion", "text_to_image", "multimodal"],
        "description": "Text-to-image diffusion dataset",
        "aliases": ["diffusion/text_to_image", "text2image", "t2i"],
    },
    "refcoco": {
        "class": "RefCOCODataset",
        "module": "refcoco_dataset",
        "task_type": [
            "referring_expression_comprehension",
            "multimodal",
            "vision_language",
        ],
        "description": "RefCOCO referring expression comprehension dataset",
        "aliases": ["refcoco"],
    },
    "refcoco+": {
        "class": "RefCOCOPlusDataset",
        "module": "refcoco_dataset",
        "task_type": [
            "referring_expression_comprehension",
            "multimodal",
            "vision_language",
        ],
        "description": "RefCOCO+ referring expression comprehension dataset (no location words)",
        "aliases": ["refcoco+", "refcocoplus"],
    },
    "refcocog": {
        "class": "RefCOCOgDataset",
        "module": "refcoco_dataset",
        "task_type": [
            "referring_expression_comprehension",
            "multimodal",
            "vision_language",
        ],
        "description": "RefCOCOg referring expression comprehension dataset (longer descriptions)",
        "aliases": ["refcocog"],
    },
}


def _resolve_mode_value(value: Any, mode: str) -> Any:
    if not isinstance(value, dict):
        return value

    mode_lower = mode.lower()
    candidates = [mode, mode_lower]
    alias_map = {
        "train": [],
        "valid": ["val", "validation"],
        "val": ["valid", "validation"],
        "test": ["testing", "eval", "evaluation"],
        "predict": ["inference"],
    }
    candidates.extend(alias_map.get(mode_lower, []))

    for key in candidates:
        if key in value:
            return value[key]

    for fallback in ("default", "all", "*", "global"):
        if fallback in value:
            return value[fallback]

    return None


def _safe_float(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r 无法转换为浮点数，将忽略该配置", key, value)
        return None
    if result <= 0:
        logger.warning("%s=%.4f 非法（需 > 0），将忽略该配置", key, result)
        return None
    return result


def _safe_int(value: Any, key: str) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r 无法转换为整数，将忽略该配置", key, value)
        return None
    if result <= 0:
        logger.warning("%s=%d 非法（需 > 0），将忽略该配置", key, result)
        return None
    return result


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _safe_non_negative_int(value: Any, key: str, default: int = 0) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError):
        logger.warning("%s=%r 无法转换为非负整数，将使用默认值 %d", key, value, default)
        return default
    if result < 0:
        logger.warning("%s=%d 小于0，将使用默认值 %d", key, result, default)
        return default
    return result


def _extract_sampling_options(
    dataset_cfg: Dict[str, Any], config: Dict[str, Any], mode: str
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    sampling_options: Dict[str, Any] = {}
    config = config.copy()

    sources: List[Dict[str, Any]] = []
    for source in (
        dataset_cfg,
        dataset_cfg.get("sampling"),
        config,
        config.get("sampling"),
    ):
        if isinstance(source, dict):
            sources.append(source)

    for source in sources:
        if "sample_ratio" in source:
            resolved = _resolve_mode_value(source["sample_ratio"], mode)
            ratio = _safe_float(resolved, "sample_ratio")
            if ratio is not None:
                sampling_options["sample_ratio"] = ratio
        if "max_samples" in source:
            resolved = _resolve_mode_value(source["max_samples"], mode)
            max_samples = _safe_int(resolved, "max_samples")
            if max_samples is not None:
                sampling_options["max_samples"] = max_samples
        if "sample_shuffle" in source:
            resolved = _resolve_mode_value(source["sample_shuffle"], mode)
            shuffle = _safe_bool(resolved)
            if shuffle is not None:
                sampling_options["sample_shuffle"] = shuffle

    # 移除已消费的配置，避免传递给数据集构造函数
    for key in ("sample_ratio", "max_samples", "sample_shuffle"):
        if key in config:
            config.pop(key)
    config.pop("sampling", None)

    return config, sampling_options


def _apply_sampling_to_dataset(
    dataset: Any, sampling_options: Dict[str, Any], mode: str
) -> Any:
    if not dataset or not sampling_options:
        return dataset

    sample_ratio = sampling_options.get("sample_ratio")
    max_samples = sampling_options.get("max_samples")
    if sample_ratio is None and max_samples is None:
        return dataset

    random_sample = sampling_options.get("sample_shuffle")
    if random_sample is None:
        random_sample = mode == "train"

    mininalize_fn = getattr(dataset, "mininalize", None)
    if not callable(mininalize_fn):
        logger.warning(
            "数据集 %s 不支持 mininalize，忽略采样配置",
            dataset.__class__.__name__,
        )
        return dataset

    try:
        if sample_ratio is not None:
            mininalize_fn(sample_ratio, random_sample)
        elif max_samples is not None:
            mininalize_fn(max_samples, random_sample)
    except Exception as exc:
        logger.warning(
            "应用采样配置到数据集 %s 时失败: %s",
            dataset.__class__.__name__,
            exc,
        )

    return dataset


def _resolve_dataloader_option(
    dataloader_cfg: Dict[str, Any], key: str, mode: str, default: Any
) -> Any:
    if not isinstance(dataloader_cfg, dict):
        return default
    value = dataloader_cfg.get(key, default)
    resolved = _resolve_mode_value(value, mode)
    return default if resolved is None else resolved


def _get_dataset_by_case(dataset_name: str):
    """根据数据集名称返回对应的数据集类"""
    name = dataset_name.lower()
    if name == "medical/mri_brain_clip":
        from .medical.brain_mri_clip_dataset import BrainMRIClipDataset

        return BrainMRIClipDataset
    elif name == "medical/btcv":
        from .medical.btcv_dataset import BTCVDataset

        return BTCVDataset
    elif name == "drive":
        from .drive_dataset import DriveDataset

        return DriveDataset
    elif name == "medical/stare":
        from .medical.stare_dataset import StareDataset

        return StareDataset
    elif name == "medical/isic2016":
        from .medical.isic2016_dataset import ISIC2016Dataset

        return ISIC2016Dataset
    elif name == "medical/isic2017":
        from .medical.isic2017_dataset import ISIC2017Dataset

        return ISIC2017Dataset
    elif name == "medical/isic2018":
        from .medical.isic2018_dataset import ISIC2018Dataset

        return ISIC2018Dataset
    elif name == "medical/bowl2018":
        from .medical.bowl2018_dataset import BOWL2018Dataset

        return BOWL2018Dataset
    elif name == "medical/chasedb1":
        from .medical.chasedb1_dataset import ChaseDB1Dataset

        return ChaseDB1Dataset
    elif name == "mnist":
        from .mnist_dataset import MNISTDataset

        return MNISTDataset
    elif name == "medical/vqarad":
        from .medical.vqa_rad_dataset import VQARadDataset

        return VQARadDataset
    elif name == "medical/pathvqa":
        from .medical.path_vqa_dataset import PathVQADataset

        return PathVQADataset
    elif name == "medical/isic2016_reasoning_seg":
        from .medical.isic2016_reasoning_seg_dataset import ISIC2016ReasoningSegDataset

        return ISIC2016ReasoningSegDataset
    elif name == "brainmri_clip":
        from .medical.brain_mri_clip_dataset import BrainMRIClipDataset

        return BrainMRIClipDataset
    elif name == "coco":
        from .coco_dataset import COCODataset

        return COCODataset
    elif name == "coco_segmentation" or name == "coco_seg":
        from .coco_dataset import COCOSegmentationDataset

        return COCOSegmentationDataset
    elif name == "cifar" or name == "cifar10":
        from .cifar_dataset import CIFAR10Dataset

        return CIFAR10Dataset
    elif name == "cifar100":
        from .cifar_dataset import CIFAR100Dataset

        return CIFAR100Dataset
    elif name == "imagenet" or name == "imagenet1k" or name == "imagenet-1k":
        from .imagenet_dataset import ImageNet1KDataset

        return ImageNet1KDataset
    elif name == "diffusion/mnist" or name == "mnist_diffusion":
        # 返回包装函数而不是类
        return (
            lambda root_dir, conditional=False, **kwargs: get_mnist_diffusion_dataset(
                root_dir, split="train", conditional=conditional, **kwargs
            )
        )
    elif name == "diffusion/cifar10" or name == "cifar10_diffusion":
        return (
            lambda root_dir, conditional=False, **kwargs: get_cifar10_diffusion_dataset(
                root_dir, split="train", conditional=conditional, **kwargs
            )
        )
    elif name == "diffusion/imagenet" or name == "imagenet_diffusion":
        return (
            lambda root_dir, conditional=True, **kwargs: get_imagenet_diffusion_dataset(
                root_dir, split="train", conditional=conditional, **kwargs
            )
        )
    elif name == "diffusion/unconditional":
        from .diffusion_dataset import UnconditionalDiffusionDataset

        return UnconditionalDiffusionDataset
    elif name == "diffusion/conditional":
        from .diffusion_dataset import ConditionalDiffusionDataset

        return ConditionalDiffusionDataset
    elif name == "diffusion/text_to_image" or name == "text2image" or name == "t2i":
        from .diffusion_dataset import TextToImageDiffusionDataset

        return TextToImageDiffusionDataset
    elif name == "refcoco":
        from .refcoco_dataset import RefCOCODataset

        return RefCOCODataset
    elif name == "refcoco_alignment" or name == "refcoco-alignment":
        from .alignment.refcoco_dataset import RefCOCOAlignmentDataset

        return RefCOCOAlignmentDataset
    elif name in ("coco2017_alignment", "coco2017-alignment", "coco_alignment"):
        from .alignment.coco2017_alignment_dataset import COCO2017RegionAlignment

        return COCO2017RegionAlignment
    else:
        logging.warning(f"No target dataset: {dataset_name}")
        return None


def get_train_dataset(
    dataset_name: str, root_dir: Path, **kwargs
):
    """获取训练数据集

    Args:
        dataset_name: 用于选择数据集类的名称
        root_dir: 数据集根目录
        dataset_name_inner: 传递给数据集构造函数的 dataset_name 参数（如果有冲突时使用此参数）
        **kwargs: 传递给数据集类构造函数的其他参数
    """
    if dataset_name.lower() == "cholect50":
        return
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    dataset_builder: Any = DatasetClass
    return dataset_builder.get_train_dataset(root_dir, **kwargs)


def get_valid_dataset(
    dataset_name: str, root_dir: Path, **kwargs
):
    """获取验证数据集"""
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    dataset_builder: Any = DatasetClass
    return dataset_builder.get_valid_dataset(root_dir, **kwargs)


def get_test_dataset(
    dataset_name: str, root_dir: Path, **kwargs
):
    """获取测试数据集"""
    DatasetClass = _get_dataset_by_case(dataset_name)
    if DatasetClass is None:
        return None
    dataset_builder: Any = DatasetClass
    return dataset_builder.get_test_dataset(root_dir, **kwargs)


def get_dataset(mode: str):
    c_dataset = get_config_value("dataset")
    assert (
        c_dataset is not None
    ), "Dataset configuration is not defined in the config file."

    if c_dataset.get("type") == "hybrid":
        logging.info(f"检测到混合数据集配置，创建HybridDataset for mode: {mode}")
        # 传递完整的配置给 create_hybrid_dataset_from_config
        full_config = get_config()
        return create_hybrid_dataset_from_config(c_dataset, mode, full_config)

    raw_config = c_dataset.get("config", {})
    config = raw_config.copy() if isinstance(raw_config, dict) else {}
    config, sampling_options = _extract_sampling_options(c_dataset, config, mode)
    dataset_name = c_dataset["name"]
    root_dir = Path(c_dataset["root_dir"])

    match mode:
        case "train":
            dataset0 = get_train_dataset(
                dataset_name,
                root_dir,
                **config,
            )
        case "test":
            dataset0 = get_test_dataset(
                dataset_name,
                root_dir,
                **config,
            )
        case "valid" | "val":
            dataset0 = get_valid_dataset(
                dataset_name, 
                root_dir, 
                **config
            )
        case _:
            DatasetClass = _get_dataset_by_case(dataset_name)
            if DatasetClass is None:
                return None
            dataset_builder: Any = DatasetClass
            dataset0 = dataset_builder.get_dataset(
                root_dir, 
                **config
            )

    if not dataset0:
        logging.error(f"{dataset_name} is empty!")

    _apply_sampling_to_dataset(dataset0, sampling_options, mode)

    return dataset0


def get_dataloader(
    dataset: Any,
    sample_ratio: Optional[float] = None,
    max_samples: int = 0,
    batch_size: int = 1,
    shuffle: bool = False,
    *,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    if dataset is None:
        return None

    if sample_ratio is not None:
        mininalize_fn = getattr(dataset, "mininalize", None)
        if callable(mininalize_fn):
            dataset = mininalize_fn(dataset_size=sample_ratio, random_sample=shuffle)
        else:
            logger.warning(
                "数据集 %s 不支持 mininalize，忽略 sample_ratio",
                dataset.__class__.__name__,
            )
    elif max_samples > 0:
        mininalize_fn = getattr(dataset, "mininalize", None)
        if callable(mininalize_fn):
            dataset = mininalize_fn(dataset_size=max_samples, random_sample=shuffle)
        else:
            logger.warning(
                "数据集 %s 不支持 mininalize，忽略 max_samples",
                dataset.__class__.__name__,
            )
    return dataset.dataloader(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_all_dataloader(use_valid=False):
    c = get_config()

    train_dataset = get_dataset("train")
    test_dataset = get_dataset("test")
    dataloader_cfg = c.get("dataloader", {})

    train_shuffle_raw = _resolve_dataloader_option(dataloader_cfg, "shuffle", "train", True)
    train_shuffle = _safe_bool(train_shuffle_raw)
    if train_shuffle is None:
        train_shuffle = True
    train_workers_raw = _resolve_dataloader_option(dataloader_cfg, "num_workers", "train", 0)
    train_num_workers = _safe_non_negative_int(train_workers_raw, "num_workers[train]", 0)
    train_pin_raw = _resolve_dataloader_option(dataloader_cfg, "pin_memory", "train", True)
    train_pin_memory = _safe_bool(train_pin_raw)
    if train_pin_memory is None:
        train_pin_memory = True
    train_drop_raw = _resolve_dataloader_option(dataloader_cfg, "drop_last", "train", False)
    train_drop_last = _safe_bool(train_drop_raw)
    if train_drop_last is None:
        train_drop_last = False

    test_shuffle_raw = _resolve_dataloader_option(dataloader_cfg, "shuffle", "test", False)
    test_shuffle = _safe_bool(test_shuffle_raw)
    if test_shuffle is None:
        test_shuffle = False
    test_workers_raw = _resolve_dataloader_option(dataloader_cfg, "num_workers", "test", train_num_workers)
    test_num_workers = _safe_non_negative_int(test_workers_raw, "num_workers[test]", train_num_workers)
    test_pin_raw = _resolve_dataloader_option(dataloader_cfg, "pin_memory", "test", train_pin_memory)
    test_pin_memory = _safe_bool(test_pin_raw)
    if test_pin_memory is None:
        test_pin_memory = train_pin_memory
    test_drop_raw = _resolve_dataloader_option(dataloader_cfg, "drop_last", "test", False)
    test_drop_last = _safe_bool(test_drop_raw)
    if test_drop_last is None:
        test_drop_last = False

    train_loader = get_dataloader(
        train_dataset,
        sample_ratio=None,
        max_samples=0,
        batch_size=c["train"]["batch_size"],
        shuffle=train_shuffle,
        num_workers=train_num_workers,
        pin_memory=train_pin_memory,
        drop_last=train_drop_last,
    )
    test_loader = get_dataloader(
        test_dataset,
        sample_ratio=None,
        max_samples=0,
        batch_size=c["test"]["batch_size"],
        shuffle=test_shuffle,
        num_workers=test_num_workers,
        pin_memory=test_pin_memory,
        drop_last=test_drop_last,
    )
    if use_valid and "valid" in c:
        valid_dataset = get_dataset("valid")
        valid_shuffle_raw = _resolve_dataloader_option(dataloader_cfg, "shuffle", "valid", False)
        valid_shuffle = _safe_bool(valid_shuffle_raw)
        if valid_shuffle is None:
            valid_shuffle = False
        valid_workers_raw = _resolve_dataloader_option(dataloader_cfg, "num_workers", "valid", train_num_workers)
        valid_num_workers = _safe_non_negative_int(valid_workers_raw, "num_workers[valid]", train_num_workers)
        valid_pin_raw = _resolve_dataloader_option(dataloader_cfg, "pin_memory", "valid", train_pin_memory)
        valid_pin_memory = _safe_bool(valid_pin_raw)
        if valid_pin_memory is None:
            valid_pin_memory = train_pin_memory
        valid_drop_raw = _resolve_dataloader_option(dataloader_cfg, "drop_last", "valid", False)
        valid_drop_last = _safe_bool(valid_drop_raw)
        if valid_drop_last is None:
            valid_drop_last = False

        valid_loader = get_dataloader(
            valid_dataset,
            sample_ratio=None,
            max_samples=0,
            batch_size=c["valid"]["batch_size"],
            shuffle=valid_shuffle,
            num_workers=valid_num_workers,
            pin_memory=valid_pin_memory,
            drop_last=valid_drop_last,
        )
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader


def get_train_valid_test_dataloader(use_valid=False):
    return get_all_dataloader(use_valid=use_valid)

def get_kfold_dataloaders(
    n_splits: Optional[int] = None,
    shuffle: Optional[bool] = None,
    random_state: Optional[int] = None,
    include_test: Optional[bool] = None,
    test_ratio: Optional[float] = None,
) -> List[Tuple[DataLoader, DataLoader, Optional[DataLoader]]]:
    """暂未实现的kfold数据加载接口。"""
    message = "当前版本暂不支持 k-fold 数据加载，请使用常规数据加载流程。"
    logger.warning(message)
    raise NotImplementedError(message)


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
        if task_type_lower in [t.lower() for t in info["task_type"]]:
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
        return info["task_types"]
    return []


def get_datasets_by_task(task_type: str) -> List[Dict[str, Any]]:
    """根据任务类型获取数据集详细信息列表（保留用于向后兼容）

    Args:
        task_type: 任务类型

    Returns:
        匹配的数据集详细信息列表
    """
    task_type_lower = task_type.lower()
    matching_datasets = []

    for dataset_name, info in DATASET_REGISTRY.items():
        if task_type_lower in [t.lower() for t in info["task_type"]]:
            matching_datasets.append(
                {
                    "name": dataset_name,
                    "class": info["class"],
                    "task_types": info["task_type"],
                    "description": info["description"],
                    "aliases": info["aliases"],
                }
            )

    return matching_datasets


def get_all_task_types() -> List[str]:
    """获取所有可用的任务类型

    Returns:
        所有任务类型的列表（去重且排序）
    """
    all_tasks = set()
    for info in DATASET_REGISTRY.values():
        all_tasks.update(info["task_type"])

    return sorted(list(all_tasks))


def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
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
            "name": dataset_name_lower,
            "class": info["class"],
            "module": info["module"],
            "task_types": info["task_type"],
            "description": info["description"],
            "aliases": info["aliases"],
        }

    # 通过别名查找
    for main_name, info in DATASET_REGISTRY.items():
        if dataset_name_lower in [alias.lower() for alias in info["aliases"]]:
            return {
                "name": main_name,
                "class": info["class"],
                "module": info["module"],
                "task_types": info["task_type"],
                "description": info["description"],
                "aliases": info["aliases"],
            }

    return None


def list_all_datasets(
    task_type: Optional[str] = None, sort_by: Literal["name", "task"] = "name"
) -> Dict[str, List[Dict]]:
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

    if sort_by == "name":
        all_datasets = []
        for dataset_name, info in sorted(DATASET_REGISTRY.items()):
            all_datasets.append(
                {
                    "name": dataset_name,
                    "class": info["class"],
                    "task_types": info["task_type"],
                    "description": info["description"],
                    "aliases": info["aliases"],
                }
            )
        return {"all": all_datasets}

    elif sort_by == "task":
        # 按任务类型分组
        task_groups = {}
        for dataset_name, info in DATASET_REGISTRY.items():
            for task in info["task_type"]:
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append(
                    {
                        "name": dataset_name,
                        "class": info["class"],
                        "task_types": info["task_type"],
                        "description": info["description"],
                        "aliases": info["aliases"],
                    }
                )

        # 排序
        return {
            k: sorted(v, key=lambda x: x["name"])
            for k, v in sorted(task_groups.items())
        }

    return {}


def print_dataset_task_matrix():
    """打印数据集-任务类型矩阵，显示各数据集支持的任务"""
    print("\n" + "=" * 100)
    print("数据集任务支持矩阵".center(100))
    print("=" * 100)

    # 获取所有任务类型
    all_tasks = get_all_task_types()

    # 打印表头
    print(f"\n{'数据集名称':<35}", end="")
    for task in all_tasks[:10]:  # 只显示前10个任务以适应宽度
        print(f"{task[:12]:<13}", end="")
    print()
    print("-" * 100)

    # 打印每个数据集
    for dataset_name in sorted(DATASET_REGISTRY.keys()):
        info = DATASET_REGISTRY[dataset_name]
        print(f"{dataset_name:<35}", end="")
        for task in all_tasks[:10]:
            mark = "✓" if task in info["task_type"] else " "
            print(f"{mark:^13}", end="")
        print(f"  {', '.join(info['task_type'][:2])}")

    print("-" * 100)
    print(f"\n总计: {len(DATASET_REGISTRY)} 个数据集, {len(all_tasks)} 种任务类型\n")


def create_hybrid_dataset(
    dataset_names: List[str],
    root_dirs: Optional[Sequence[Union[Path, str]]] = None,
    split: str = "train",
    ratios: Optional[List[float]] = None,
    priorities: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    shuffle_order: bool = False,
    random_seed: Optional[int] = None,
    **dataset_kwargs,
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
            if ds_name in dataset_kwargs and "root_dir" in dataset_kwargs[ds_name]:
                root_dir = dataset_kwargs[ds_name]["root_dir"]
                # 转换为 Path 对象
                root_dirs.append(
                    Path(root_dir) if isinstance(root_dir, str) else root_dir
                )
            else:
                raise ValueError(f"未提供数据集 {ds_name} 的 root_dir")
    else:
        # 确保所有 root_dirs 都是 Path 对象
        root_dirs = [Path(rd) if isinstance(rd, str) else rd for rd in root_dirs]

    if len(dataset_names) != len(root_dirs):
        raise ValueError("数据集名称和根目录的数量必须相同")

    # 创建数据集列表
    datasets = []
    normalized_root_dirs: List[Path] = [
        Path(rd) if isinstance(rd, str) else rd for rd in root_dirs
    ]

    for i, (ds_name, root_dir) in enumerate(zip(dataset_names, normalized_root_dirs)):
        # 获取该数据集的特定参数
        ds_specific_kwargs = dataset_kwargs.get(ds_name, {})

        # 根据 split 获取对应的数据集
        if split == "train":
            dataset = get_train_dataset(ds_name, root_dir, **ds_specific_kwargs)
        elif split in ["val", "valid"]:
            dataset = get_valid_dataset(ds_name, root_dir, **ds_specific_kwargs)
        elif split == "test":
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
        random_seed=random_seed,
    )


def create_hybrid_dataset_by_task(
    task_type: str,
    root_dir_mapping: Dict[str, Union[Path, str]],
    split: str = "train",
    max_datasets: Optional[int] = None,
    ratios: Optional[List[float]] = None,
    **kwargs,
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
        raise ValueError(
            f"没有找到支持 '{task_type}' 任务且在 root_dir_mapping 中的数据集"
        )

    # 限制数据集数量
    if max_datasets and len(dataset_names) > max_datasets:
        dataset_names = dataset_names[:max_datasets]

    # 创建 root_dirs 列表（支持 str 和 Path）
    root_dirs = [root_dir_mapping[ds] for ds in dataset_names]

    logging.info(
        f"为任务 '{task_type}' 创建混合数据集，包含 {len(dataset_names)} 个数据集: {dataset_names}"
    )

    # 创建混合数据集
    return create_hybrid_dataset(
        dataset_names=dataset_names,
        root_dirs=root_dirs,
        split=split,
        ratios=ratios,
        **kwargs,
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
            all_tasks.update(info["task_types"])
            all_descriptions.append(f"{ds_name}: {info['description']}")

    return {
        "dataset_names": dataset_names,
        "num_datasets": len(dataset_names),
        "combined_tasks": sorted(list(all_tasks)),
        "descriptions": all_descriptions,
    }


def get_diffusion_train_dataset(
    dataset_name: str, root_dir: Path, conditional: bool = False, **kwargs
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
    if name.startswith("diffusion/"):
        base_name = name.replace("diffusion/", "")
    else:
        base_name = name

    # 根据基础数据集名称选择对应的diffusion包装器
    if base_name in ["mnist", "mnist_diffusion"]:
        return get_mnist_diffusion_dataset(root_dir, "train", conditional, **kwargs)
    elif base_name in ["cifar10", "cifar", "cifar10_diffusion"]:
        return get_cifar10_diffusion_dataset(root_dir, "train", conditional, **kwargs)
    elif base_name in ["imagenet", "imagenet1k", "imagenet-1k", "imagenet_diffusion"]:
        return get_imagenet_diffusion_dataset(root_dir, "train", conditional, **kwargs)
    else:
        # 尝试获取基础数据集并包装为diffusion数据集
        base_dataset = get_train_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)


def get_diffusion_valid_dataset(
    dataset_name: str, root_dir: Path, conditional: bool = False, **kwargs
) -> DiffusionDataset:
    """获取Diffusion验证数据集"""
    name = dataset_name.lower()

    if name.startswith("diffusion/"):
        base_name = name.replace("diffusion/", "")
    else:
        base_name = name

    if base_name in ["mnist", "mnist_diffusion"]:
        return get_mnist_diffusion_dataset(root_dir, "valid", conditional, **kwargs)
    elif base_name in ["cifar10", "cifar", "cifar10_diffusion"]:
        return get_cifar10_diffusion_dataset(root_dir, "valid", conditional, **kwargs)
    elif base_name in ["imagenet", "imagenet1k", "imagenet-1k", "imagenet_diffusion"]:
        return get_imagenet_diffusion_dataset(root_dir, "valid", conditional, **kwargs)
    else:
        base_dataset = get_valid_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)


def get_diffusion_test_dataset(
    dataset_name: str, root_dir: Path, conditional: bool = False, **kwargs
) -> DiffusionDataset:
    """获取Diffusion测试数据集"""
    name = dataset_name.lower()

    if name.startswith("diffusion/"):
        base_name = name.replace("diffusion/", "")
    else:
        base_name = name

    if base_name in ["mnist", "mnist_diffusion"]:
        return get_mnist_diffusion_dataset(root_dir, "test", conditional, **kwargs)
    elif base_name in ["cifar10", "cifar", "cifar10_diffusion"]:
        return get_cifar10_diffusion_dataset(root_dir, "test", conditional, **kwargs)
    elif base_name in ["imagenet", "imagenet1k", "imagenet-1k", "imagenet_diffusion"]:
        return get_imagenet_diffusion_dataset(root_dir, "test", conditional, **kwargs)
    else:
        base_dataset = get_test_dataset(base_name, root_dir, **kwargs)
        if base_dataset is None:
            raise ValueError(f"不支持的diffusion数据集: {dataset_name}")
        return create_diffusion_dataset(base_dataset, conditional, **kwargs)
