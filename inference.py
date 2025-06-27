from pathlib import Path
import torch
from torch import nn 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, SemanticSegmentationTarget
from pytorch_grad_cam import GradCAM
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import logging
from typing import Sequence

from model_operation import Predictor
from config import get_config
from utils import get_transforms, to_path, FilePath, build_image_transforms, VisionTransformersBuilder, Timer

class SegmentPredictor:
    def __init__(self, output_dir: Path, 
                 model: nn.Module, 
                 target_layers: list[nn.Module], 
                 targets: list[ClassifierOutputTarget|ClassifierOutputSoftmaxTarget|SemanticSegmentationTarget]):
        self.output_dir = output_dir
        self.model = model
        self.timer = Timer()
        self.logger = logging.getLogger('predictor')

        self.target_layers = target_layers
        self.targets = targets
    
    @torch.inference_mode()
    def predict(self, inputs: Sequence[FilePath], **kwargs):
        c = get_config()
        device = torch.device(c['device'])
        transforms = kwargs['transforms'] if 'transforms' in kwargs else None

        self.model = self.model.to(device)
        self.model.eval()

        for input in tqdm(inputs, desc="Predicting..."):
            input = to_path(input)
            input_filename = input.name

            with self.timer.timeit(input_filename + '.preprocess'):
                input, ext = self.preprocess(input, transforms=transforms)

            with self.timer.timeit(input_filename + '.inference'):
                input = input.to(device)
                pred = self.model(input)

            with self.timer.timeit(input_filename + '.postprocess'):
                image = self.postprocess(pred, **ext)

            output_filename = self.output_dir / input_filename
            image.save(output_filename)
            
        cost = self.timer.total_elapsed_time()
        self.logger.info(f'Predicting had cost {cost}s')

    def preprocess(self, input: Path, transforms: transforms.Compose|None=None):
        image = Image.open(input).convert('RGB')
        if transforms:
            image_tensor = transforms(image).unsqueeze(0)
        else:
            image_tensor = torch.as_tensor(np.array(image)).unsqueeze(0)
        return image_tensor, {
            'size': image.size
        }

    def postprocess(self, pred: torch.Tensor, **kwargs) -> Image.Image:
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(0)
        pred = pred.astype(torch.uint8)

        image = Image.fromarray(pred, mode='L')
        if 'size' in kwargs:
            image = image.resize(kwargs['size'])
        return image
