from pathlib import Path
from torchvision import transforms
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

import json
# nnU-Net Style
"""
|-imageTr
|-imageTs
|-labelTr
|-dataset.json
"""
class NNUNetDataset(CustomDataset):

    def __init__(self, base_dir: Path, dataset_type: Literal['train', 'valid', 'test'], transforms: transforms.Compose|None=None, use_numpy=False, config_filename: str='dataset.json', **kwargs):
        super(NNUNetDataset, self).__init__(base_dir=base_dir, dataset_type=dataset_type, use_numpy=use_numpy)

        config_file = base_dir / config_filename
        uunet_config = json.load(config_file.open("r"))
        
        self.labels = uunet_config['labels']
        if dataset_type == 'train':
            self.config = uunet_config['training']
        elif dataset_type == 'test':
            self.config = uunet_config['test']
        elif dataset_type == 'valid':
            self.config = uunet_config['validation']

        self.transforms = transforms
        self.n = len(self.config)
