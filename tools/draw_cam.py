import os.path
from pathlib import Path
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget
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
from models.sample.unet import UNet


def cam_segment(model: nn.Module, target_layers: list, input_tensor: torch.Tensor, image_np: np.ndarray, target_category: int, mask_path: Path, *, is_rgb=True, show=False):
    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    targets = [ClassifierOutputTarget(target_category)]
    targets = [SemanticSegmentationTarget(0, mask)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    plt.imshow(visualization)
    plt.title(f"Grad-CAM for category {target_category} (all pixels)")
    plt.savefig(mask_path)
    if show:
        plt.show()


def resnet50_check(image_path: Path, mask_path: Path, *, is_rgb: bool, show=False):
    # Load a pre-trained ResNet50 model
    # model = resnet50(pretrained=True)
    model = UNet(1, 1, True)
    model.load_state_dict(torch.load(r'..\results\train\unet\weights\best_model.pth')['model'])
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image_path)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    # transform = image_transforms(resize=(224, 224), is_rgb=is_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if is_rgb else transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Convert the image to a NumPy array for visualization
    image_np = np.array(image) / 255.0

    # Choose a target category (e.g., 0 for the first class)
    target_category = 0

    # Generate and display the Grad-CAM visualization
    cam_segment(model, target_layers, input_tensor, image_np, target_category, mask_path, is_rgb=is_rgb, show=show)

if __name__ == '__main__':
    # image_path = r'data\DRIVE\training\images\21.png'
    parser = ArgumentParser('draw cam')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input Image to Draw Cam')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output Cammed Image to be saved')
    parser.add_argument('--show', action='store_true', help='Show image using plt.show()')
    
    args = parser.parse_args()
    input_image = args.input
    output_image = args.output

    resnet50_check(input_image, output_image, show=args.show, is_rgb=True)
