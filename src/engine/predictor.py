from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image
import numpy as np
import logging
from src.config import get_config
from src.utils import Timer, get_transforms, select_postprocess_fn

class Predictor:
    def __init__(self, output_dir: Path, model: nn.Module):
        self.output_dir = output_dir
        self.model = model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('predict')
        self.timer = Timer()
        c = get_config()
        postprocess_name = c.get('postprocess', "")
        self.postprocess = select_postprocess_fn(postprocess_name)
        assert postprocess_name is not None, f"Not supported postprocess function {postprocess_name}, please set 'postprocess' in config file"

    @torch.inference_mode()
    def predict(self, inputs: list[Path], **kwargs):
        c = get_config()
        device = torch.device(c['device'])
        self.model = self.model.to(device)
        self.model.eval()
        for input in tqdm(inputs, desc="Predicting..."):
            input_filename = input.name
            with self.timer.timeit(task=input_filename + '.preprocess'):
                image = Image.open(input).convert('L')
                size = image.size # (H, W)
                transforms = get_transforms()
                image_tensor = transforms(image).unsqueeze(0)
            with self.timer.timeit(task=input_filename + '.inference'):
                image_tensor = image_tensor.to(device)
                pred_tensor = self.model(image_tensor)
            with self.timer.timeit(task=input_filename + '.postprocess'):
                import torch.nn.functional as F
                pred = F.sigmoid(pred_tensor)
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                pred = pred.detach().cpu().numpy()
                pred = pred.squeeze(0).squeeze(0)
                pred = pred.astype(np.uint8)
            output_filename = self.output_dir / input_filename
            pred_image = Image.fromarray(pred, mode='L')
            pred_image = pred_image.resize(size)
            pred_image.save(output_filename)
        cost = self.timer.total_elapsed_time()
        self.logger.info(f'Predicting had cost {cost}s')
