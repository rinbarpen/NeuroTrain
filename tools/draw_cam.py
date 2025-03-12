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


def cam_segment(model: nn.Module, target_layers: list, input_tensor: torch.Tensor, image_np: np.ndarray, target_category: int, mask_path: Path|None=None):
    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    use_rgb = image_np.ndim == 3
    targets = [ClassifierOutputTarget(target_category)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=use_rgb)

    plt.imshow(visualization)
    plt.title(f"Grad-CAM for category {target_category} (all pixels)")
    if mask_path:
        plt.savefig(mask_path)
    else:
        plt.show()

def resnet50_check(image_path: Path, mask_path: Path|None, *, is_rgb: bool):
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    cam_segment(model, target_layers, input_tensor, image_np, target_category, mask_path)

if __name__ == '__main__':
    # image_path = r'data\DRIVE\training\images\21.png'
    parser = ArgumentParser('draw cam')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input Image to Draw Cam')
    parser.add_argument('-o', '--output', type=str, default='', help='Output Cammed Image to be saved')
    
    args = parser.parse_args()
    input_image = args.input
    output_image = args.output
    if not os.path.isfile(args.output):
        output_image = None

    resnet50_check(input_image, output_image, is_rgb=True)
