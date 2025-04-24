from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torch import nn
import numpy as np
import torch
import cv2
from PIL import Image

from utils.transform import build_image_transforms
from models.sample.unet import UNet
from models.transformer.deit_vit import vit_base_patch16_224

def unet_check(image: Path, mask: np.ndarray, is_rgb: bool, *, target_category: int=0) -> None:
    # Load a pre-trained ResNet50 model
    model = UNet(1, 1, True)
    model.load_state_dict(torch.load(r'..\results\train\unet\weights\best_model.pth')['model'])
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    transform = build_image_transforms(resize=(224, 224), is_pil_image=True, is_rgb=is_rgb)
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    
    targets = [SemanticSegmentationTarget(0, mask)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

def resnet50_check(image: Path, is_rgb: bool, *, target_category: int=0) -> None:
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    transform = build_image_transforms(resize=(224, 224), is_pil_image=True, is_rgb=is_rgb)
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    targets = [ClassifierOutputTarget(target_category)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    return Image.fromarray(cam_image, mode='RGB')


class ImageHeatMapGenerator():
    def __init__(self, model: nn.Module, target_layers: list[nn.Module], targets: list, model_params_file: str|Path|None=None):
        self.model = model
        if model_params_file:
            self.model.load_state_dict(torch.load(model_params_file))
        self.model.eval()
        self.target_layers = target_layers
        self.targets = targets
    
    def check(self, image: Image, transform_fn):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        H, W, C = image.shape
        input_tensor = transform_fn(image).unsqueeze(0)
        image_np = np.array(image) / 255.0

        self.model = self.model.to(device)
        input_tensor = input_tensor.to(device)

        is_rgb = (C == 3)
        with GradCAM(self.model, target_layers=self.target_layers) as cam: 
            grayscale_cam = cam(input_tensor, targets=self.targets, eigen_smooth=False)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (W, H))

            cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)
            heatmap_image = Image.fromarray(cam_image, mode='RGB' if is_rgb else 'L')
        return heatmap_image

    def check_batch(self, images: list[Image], transform_fn):
        return [self.check(image, transform_fn) for image in images]
