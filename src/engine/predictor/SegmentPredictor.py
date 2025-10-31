import torch
from PIL import Image
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as mtf
import numpy as np

from engine.predictor.BasePredictor import BasePredictor

class SegmentPredictor(BasePredictor):
    def __init__(self, output_dir: str|Path, model: nn.Module, device: str, **kwargs):
        super().__init__(output_dir, model, device, **kwargs)
    
    @torch.no_grad()
    def predict(self, batch_inputs: list[str|Image.Image|Path], **kwargs):
        self.model.eval()
        batch_inputs = [self.image_preprocess(batch_input, resize=kwargs.get('resize', (224, 224))).to(self.device) for batch_input in batch_inputs]

        masks = []
        for inputs in batch_inputs:
            outputs = self.model(inputs.unsqueeze(0), **kwargs)
            mask = outputs['masks'].squeeze(0) # (N, H, W) or (H, W)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.max() > 1:
                mask = mask.sigmoid()
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
            masks.append(mask)
        return masks # [(N, H, W)]
    
    def image_preprocess(self, image: str|Image.Image|Path, mode='RGB', resize: tuple[int, int]=(224, 224)):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Path):
            image = Image.open(str(image))
        image = image.convert(mode)

        image = mtf.Compose([
            mtf.Resize(resize),
            mtf.PILToTensor(),
            mtf.ConvertImageDtype(torch.float32),
        ])(image)
        return image
