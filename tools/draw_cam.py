import os.path
from pathlib import Path
from typing import List, Optional
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import resnet50
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch
from PIL import Image
import cv2
from argparse import ArgumentParser


def _build_classification_transform(is_rgb: bool) -> transforms.Compose:
    """
    构建用于分类模型输入的图像变换。

    - 如果是RGB图像：Resize到(224,224)，转Tensor，并使用ImageNet均值方差归一化。
    - 如果是灰度图：统一转换为3通道灰度图以适配ResNet的3通道输入，再进行归一化。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if is_rgb:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 转为3通道
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def cam_classification(
    model: nn.Module,
    target_layers: List[nn.Module],
    input_tensor: torch.Tensor,
    image_np: np.ndarray,
    output_path: Path,
    *,
    target_category: int = 0,
    is_rgb: bool = True,
    show: bool = False,
) -> None:
    """
    使用GradCAM++对分类模型生成热力图并叠加到原图上。

    Parameters
    ----------
    model : nn.Module
        需要可微分的分类模型。
    target_layers : List[nn.Module]
        用于计算CAM的目标层，通常选择最后一个卷积层。
    input_tensor : torch.Tensor
        经过预处理后的输入张量，形状为[N, C, H, W]。
    image_np : np.ndarray
        原始图像的numpy表示，取值范围需为[0,1]，形状为[H, W, C]。
    output_path : Path
        可视化图片的保存路径。
    target_category : int, optional
        目标类别ID，默认0。
    is_rgb : bool, optional
        是否为RGB图像，默认True。
    show : bool, optional
        是否显示图像窗口，默认False。
    """
    # 构建分类目标
    targets = [ClassifierOutputTarget(target_category)]

    # 使用GradCAM++
    cam = GradCAMPlusPlus(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]

    # 将CAM调整到原图大小
    H, W = image_np.shape[0], image_np.shape[1]
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))

    # 叠加可视化
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    # 保存与可视化
    plt.imshow(visualization)
    plt.title(f"Grad-CAM++ for category {target_category}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def resnet50_check(image_path: Path, output_path: Path, *, is_rgb: bool = True, show: bool = False, target_category: int = 0) -> None:
    """
    使用预训练ResNet50与GradCAM++在原图上绘制热力图示例。

    Parameters
    ----------
    image_path : Path
        输入图像路径。
    output_path : Path
        输出可视化路径。
    is_rgb : bool, optional
        是否以RGB模式读取图片；若为灰度，将自动转为3通道，默认True。
    show : bool, optional
        是否显示可视化窗口，默认False。
    target_category : int, optional
        指定分类目标类别，默认0。
    """
    # 加载预训练ResNet50（兼容不同torch/torchvision版本）
    try:
        model = resnet50(weights="IMAGENET1K_V1")
    except Exception:
        model = resnet50(pretrained=True)
    model.eval()

    # 选择目标层（最后一个卷积block）
    target_layers = [model.layer4[-1]]

    # 读取与预处理图像
    image = Image.open(image_path)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    transform = _build_classification_transform(is_rgb=is_rgb)
    input_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # 设备迁移
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 原图（用于可视化叠加），归一化到[0,1]
    image_np = np.array(image)
    if image_np.ndim == 2:  # 灰度图 -> 3通道
        image_np = np.stack([image_np] * 3, axis=-1)
    image_np = image_np.astype(np.float32) / 255.0

    # 生成并保存热力图
    cam_classification(
        model,
        target_layers,
        input_tensor,
        image_np,
        output_path,
        target_category=target_category,
        is_rgb=is_rgb,
        show=show,
    )


if __name__ == "__main__":
    """
    命令行使用示例：
    python tools/draw_cam.py --image ./assets/cat.jpg --out ./outputs/cam_cat.png --rgb --category 285 --show
    兼容旧参数：-i/--input 与 -o/--output
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", "--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("-o", "--out", "--output", type=str, required=True, help="输出可视化保存路径")
    parser.add_argument("--rgb", action="store_true", help="以RGB模式读取（默认）")
    parser.add_argument("--gray", action="store_true", help="以灰度模式读取（将自动转3通道以适配ResNet）")
    parser.add_argument("--category", type=int, default=0, help="目标类别ID，默认0")
    parser.add_argument("--show", action="store_true", help="是否显示窗口")

    args = parser.parse_args()

    is_rgb = True
    if args.gray:
        is_rgb = False
    elif args.rgb:
        is_rgb = True

    resnet50_check(Path(args.image), Path(args.out), is_rgb=is_rgb, show=args.show, target_category=args.category)
