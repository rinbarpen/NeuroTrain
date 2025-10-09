from pathlib import Path
from torchvision import transforms
from typing import Literal

from .custom_dataset import CustomDataset, Betweens

import json5
# nnU-Net Style
"""
|-imageTr
|-imageTs
|-labelTr
|-dataset.json
"""
class NNUNetDataset(CustomDataset):

    def __init__(self, root_dir: Path, split: Literal['train', 'valid', 'test'], transforms: transforms.Compose|None=None, use_numpy=False, config_filename: str='dataset.json', **kwargs):
        # 将参数映射到父类期望的参数名
        super(NNUNetDataset, self).__init__(root_dir, split, **kwargs)

        config_file = root_dir / config_filename
        
        uunet_config = json5.load(config_file.open("r"))
        
        self.labels = uunet_config['labels']
        if split == 'train':
            self.config = uunet_config['training']
        elif split == 'test':
            self.config = uunet_config['test']
        elif split == 'valid':
            self.config = uunet_config['validation']

        self.transforms = transforms
        self.n = len(self.config)
