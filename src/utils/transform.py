import os
from pathlib import Path

import torch
from torchvision import transforms as mtf
from torchvision.transforms import InterpolationMode
from typing import Any, List, Sequence

from src.utils.util import str2dtype

# 加载 .env，使 USE_MONAI 等由 .env 管理的变量在下方 MONAI 判断中生效
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass

# ImageNet-1K 官方预处理均值/方差（torchvision 常用）
IMAGE_1K_MEAN = [0.485, 0.456, 0.406]
IMAGE_1K_STD = [0.229, 0.224, 0.225]

# CLIP 官方预处理均值/方差（与 alignment_dataset.default_clip_transform 一致）
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# VisionTransformersBuilder may be flawed
class VisionTransformersBuilder:
    def __init__(self):
        self._transforms = []

    def resize(self, size: tuple[int, int], interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        self._transforms.append(mtf.Resize(size, interpolation=interpolation))
        return self
    def crop(self, size: tuple[int, int], *args):
        self._transforms.append(mtf.RandomCrop(size, *args))
        return self

    def center_crop(self, size: int | tuple[int, int]):
        s = (size, size) if isinstance(size, int) else size
        self._transforms.append(mtf.CenterCrop(s))
        return self

    def random_rotation(self, degrees: float):
        self._transforms.append(mtf.RandomRotation(degrees=degrees))
        return self
    def random_horizontal_flip(self, p: float):
        self._transforms.append(mtf.RandomHorizontalFlip(p=p))
        return self
    def random_vertical_flip(self, p: float):
        self._transforms.append(mtf.RandomVerticalFlip(p=p))
        return self
    def random_invert(self, p: float):
        self._transforms.append(mtf.RandomInvert(p=p))
        return self

    def PIL_to_tensor(self):
        self._transforms.append(mtf.PILToTensor())
        return self
    def to_tensor(self):
        self._transforms.append(mtf.ToTensor())
        return self
    def to_pil_image(self):
        self._transforms.append(mtf.ToPILImage())
        return self
    def convert_image_dtype(self, ttype: torch.dtype=torch.float32):
        self._transforms.append(mtf.ConvertImageDtype(ttype))
        return self

    def gray_scale(self, num_output_channels: int=1):
        self._transforms.append(mtf.Grayscale(num_output_channels=num_output_channels))
        return self

    def color_jitter(self, brightness: float|tuple[float, float] = 0,
        contrast: float|tuple[float, float] = 0,
        saturation: float|tuple[float, float] = 0,
        hue: float|tuple[float, float] = 0):
        self._transforms.append(mtf.ColorJitter(brightness, contrast, saturation, hue))
        return self

    def gaussian_blur(self, kernel_size: int|Sequence[int], sigma: float|tuple[float, float]=(0.1, 2.0)):
        self._transforms.append(mtf.GaussianBlur(kernel_size, sigma))
        return self

    def normalize(self, is_rgb: bool):
        if is_rgb:
            self._transforms.append(mtf.Normalize(mean=IMAGE_1K_MEAN, std=IMAGE_1K_STD))
        else:
            self._transforms.append(
                mtf.Normalize(mean=[0.5], std=[0.5]))
        return self

    def normalize_clip(self):
        self._transforms.append(mtf.Normalize(mean=CLIP_MEAN, std=CLIP_STD))
        return self

    def normalize_image1k(self):
        self._transforms.append(mtf.Normalize(mean=IMAGE_1K_MEAN, std=IMAGE_1K_STD))
        return self

    def aug_mix(self, severity: int=3, mixture_width: int=3, chain_depth: int=-1, alpha: float=1, all_ops: bool=True, interpolation: InterpolationMode=InterpolationMode.BILINEAR, fill: list[float]|None=None):
        self._transforms.append(mtf.AugMix(severity, mixture_width, chain_depth, alpha, all_ops, interpolation, fill))
        return self

    def build(self) -> mtf.Compose:
        return mtf.Compose(self._transforms)

def get_transforms(config: dict | None = None) -> mtf.Compose:
    from src.config import get_config

    cfg = config or get_config()
    transform_conf = cfg.get('transform') or {}

    builder = VisionTransformersBuilder()
    if not transform_conf:
        builder = builder.to_tensor()
        return builder.build()

    for k, v in transform_conf.items():
        match k.upper():
            case 'RESIZE':
                builder = builder.resize(tuple(v))
            case 'HFLIP':
                builder = builder.random_horizontal_flip(*v)
            case 'VFLIP':
                builder = builder.random_vertical_flip(*v)
            case 'ROTATION':
                builder = builder.random_rotation(*v)
            case 'INVERT':
                builder = builder.random_invert(*v)
            case 'CROP':
                builder = builder.crop(tuple(v))
            case 'AUG_MIX':
                builder = builder.aug_mix(*v)
            case 'GRAY_SCALE':
                builder = builder.gray_scale(*v)
            case 'COLOR_JITTER':
                builder = builder.color_jitter(*v)
            case 'GAUSSIAN_BLUR':
                builder = builder.gaussian_blur(*v)
            case 'NORMALIZE':
                builder = builder.normalize(*v)
            case 'TO_TENSOR':
                builder = builder.to_tensor()
            case 'PIL_TO_TENSOR':
                builder = builder.PIL_to_tensor()
            case 'CONVERT_IMAGE_DTYPE':
                builder = builder.convert_image_dtype(str2dtype(v[0]))
    return builder.build()

def build_image_transforms(resize: tuple[int, int]|None=None, 
                           crop: tuple[int, int]|None=None, 
                           rotation: float|None=None, 
                           vflip: float|None=None,
                           hflip: float|None=None,
                           invert: float|None=None,
                           aug_mix: bool=False,
                           gray_scale: bool=False,
                           color_jitter: bool=False,
                           is_pil_image: bool=False,
                           ttype: torch.dtype=torch.float32,
                           norm: bool=False,
                           is_rgb: bool=False, **kwargs) -> mtf.Compose:
    builder = VisionTransformersBuilder()
    if gray_scale:
        builder = builder.gray_scale()
    if color_jitter:
        builder = builder.color_jitter()
    if resize: 
        builder = builder.resize(resize)
    if crop: 
        builder = builder.crop(crop)
    if aug_mix:
        builder = builder.aug_mix()
    if rotation: 
        builder = builder.random_rotation(rotation)
    if vflip: 
        builder = builder.random_vertical_flip(vflip)
    if hflip: 
        builder = builder.random_horizontal_flip(hflip)
    if invert: 
        builder = builder.random_invert(invert)
    if is_pil_image: 
        builder = builder.PIL_to_tensor()
        builder = builder.convert_image_dtype(ttype)
    else:
        builder = builder.to_tensor()
    if norm:
        builder = builder.normalize(is_rgb)

    return builder.build()

def build_default_image_transforms(resize: tuple[int, int]=(224, 224), norm=True, is_rgb=True, ttype=torch.float32, **kwargs) -> mtf.Compose:
    return build_image_transforms(resize=resize, is_pil_image=True, norm=norm, is_rgb=is_rgb, ttype=ttype, **kwargs)


# ---------- 常见场景变换模板 ----------

def template_classification_train(
    size: tuple[int, int] = (224, 224),
    is_rgb: bool = True,
    crop_size: tuple[int, int] | None = None,
    hflip_p: float = 0.5,
    color_jitter: bool = False,
    ttype: torch.dtype = torch.float32,
) -> mtf.Compose:
    """图像分类训练集：Resize + 可选 RandomCrop/HFlip/ColorJitter + ToTensor + Normalize."""
    builder = VisionTransformersBuilder()
    builder = builder.resize(size)
    if crop_size is not None:
        builder = builder.crop(crop_size)
    if hflip_p > 0:
        builder = builder.random_horizontal_flip(hflip_p)
    if color_jitter:
        builder = builder.color_jitter(brightness=0.2, contrast=0.2, saturation=0.2)
    builder = builder.PIL_to_tensor().convert_image_dtype(ttype).normalize(is_rgb)
    return builder.build()


def template_classification_eval(
    size: tuple[int, int] = (224, 224),
    is_rgb: bool = True,
    ttype: torch.dtype = torch.float32,
) -> mtf.Compose:
    """图像分类验证/测试：Resize + ToTensor + Normalize，无随机性."""
    builder = VisionTransformersBuilder()
    builder = builder.resize(size).PIL_to_tensor().convert_image_dtype(ttype).normalize(is_rgb)
    return builder.build()


def template_clip(
    image_size: int = 224,
    center_crop: bool = True,
) -> mtf.Compose:
    """图文/区域对齐（CLIP/VLM）：Resize + CenterCrop + ToTensor + CLIP 官方 mean/std."""
    builder = VisionTransformersBuilder()
    builder = builder.resize(
        (image_size, image_size),
        interpolation=InterpolationMode.BICUBIC,
    )
    if center_crop:
        builder = builder.center_crop(image_size)
    builder = builder.to_tensor().normalize_clip()
    return builder.build()


def template_grayscale_medical(
    size: tuple[int, int] = (224, 224),
    use_grayscale: bool = True,
    ttype: torch.dtype = torch.float32,
) -> mtf.Compose:
    """灰度/医学 2D（如视网膜）：可选 Grayscale + Resize + ToTensor + Normalize(0.5, 0.5)."""
    builder = VisionTransformersBuilder()
    if use_grayscale:
        builder = builder.gray_scale()
    builder = builder.resize(size).PIL_to_tensor().convert_image_dtype(ttype).normalize(is_rgb=False)
    return builder.build()


def template_inference(
    size: tuple[int, int] = (224, 224),
    is_rgb: bool = True,
    ttype: torch.dtype = torch.float32,
) -> mtf.Compose:
    """仅推理/部署：Resize + ToTensor + Normalize，与分类验证一致，显式命名便于语义区分."""
    return template_classification_eval(size=size, is_rgb=is_rgb, ttype=ttype)


def template_segmentation_shared(
    size: tuple[int, int] = (512, 512),
    use_grayscale: bool = True,
    ttype: torch.dtype = torch.float32,
) -> mtf.Compose:
    """分割（image+mask 同几何）：同一 Compose 用于图像与掩码，几何+ToTensor+归一化；与 drive 等用法一致."""
    return template_grayscale_medical(size=size, use_grayscale=use_grayscale, ttype=ttype)


def get_transform_template(
    name: str,
    **kwargs,
) -> mtf.Compose:
    """按模板名返回即用 Compose。name 可选: classification_train, classification_eval, clip, grayscale_medical, inference, segmentation_shared."""
    key = name.strip().lower().replace("-", "_")
    templates = {
        "classification_train": template_classification_train,
        "classification_eval": template_classification_eval,
        "clip": template_clip,
        "grayscale_medical": template_grayscale_medical,
        "inference": template_inference,
        "segmentation_shared": template_segmentation_shared,
    }
    if key not in templates:
        raise ValueError(f"Unknown transform template: {name}. Choose from {list(templates.keys())}.")
    return templates[key](**kwargs)


# ---------- MONAI 支持（3D 医学影像，可选依赖；需 .env 中 USE_MONAI=1 才启用） ----------

_load_env()
_MONAI_ENABLED_VALUES = ("1", "true", "yes")

try:
    from monai.transforms.compose import Compose as MonaiCompose
    _monai_import_ok = True
except ImportError:
    MonaiCompose = None  # type: ignore[misc, assignment]
    _monai_import_ok = False

MONAI_AVAILABLE = (
    _monai_import_ok
    and os.getenv("USE_MONAI", "").strip().lower() in _MONAI_ENABLED_VALUES
)


def template_monai_3d(
    spatial_size: tuple[int, ...] = (224, 224, 112),
    load_image: bool = True,
    normalize_nonzero: bool = True,
    channel_wise: bool = True,
) -> Any:
    """3D 医学影像（如 BraTS）：LoadImage → EnsureChannelFirst → Resize → NormalizeIntensity → ToTensor.
    需安装 monai 且在 .env 中设置 USE_MONAI=1 才会启用，否则抛出 ImportError."""
    if not MONAI_AVAILABLE or MonaiCompose is None:
        raise ImportError(
            "MONAI is required for template_monai_3d. Install monai and set USE_MONAI=1 in .env"
        )
    from monai.transforms.intensity.array import NormalizeIntensity as _Norm
    from monai.transforms.io.array import LoadImage as _Load
    from monai.transforms.spatial.array import Resize as _Resize
    from monai.transforms.utility.array import EnsureChannelFirst as _Ensure, ToTensor as _ToTensor

    steps: list[Any] = []
    if load_image:
        steps.append(_Load(image_only=True))
    steps.extend(
        [
            _Ensure(),
            _Resize(spatial_size=spatial_size),
            _Norm(nonzero=normalize_nonzero, channel_wise=channel_wise),
            _ToTensor(),
        ]
    )
    assert MonaiCompose is not None
    return MonaiCompose(steps)
