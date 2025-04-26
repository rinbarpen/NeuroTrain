from pathlib import Path
import torch
from torch import nn 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, SemanticSegmentationTarget
from pytorch_grad_cam import GradCAM
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from model_operation import Predictor
from config import get_config
from utils.transform import get_transforms

class SegmentPredictor(Predictor):
    def __init__(self, output_dir: Path, 
                 model: nn.Module, 
                 target_layers: list[nn.Module], 
                 targets: list[ClassifierOutputTarget|ClassifierOutputSoftmaxTarget|SemanticSegmentationTarget]):
        super(SegmentPredictor, self).__init__(output_dir, model)

        self.target_layers = target_layers
        self.targets = targets
    
    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])

        self.model = self.model.to(device)
        self.model.eval()

        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name

            with self.timer.timeit(input_filename + '.preprocess'):
                input = self.preprocess(input)

            with self.timer.timeit(input_filename + '.inference'):
                input = input.to(device)
                pred = self.model(input)

            with self.timer.timeit(input_filename + '.postprocess'):
                image = self.postprocess(pred)

            output_filename = self.output_dir / input_filename
            image.save(output_filename)
            
        cost = self.timer.total_elapsed_time()
        self.logger.info(f'Predicting had cost {cost}s')

    @classmethod
    def preprocess(self, input: Path) -> torch.Tensor:
        input = Image.open(input).convert('L')
        transforms = get_transforms()
        input = transforms(input).unsqueeze(0)
        return input

    @classmethod
    def postprocess(self, pred: torch.Tensor) -> Image.Image:
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze(0).squeeze(0)
        pred = pred.astype(np.uint8)
        image = Image.fromarray(pred, mode='L')
        return image
