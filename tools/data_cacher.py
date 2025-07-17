# TODO: Not completely supported yet!
import os
import toml
import yaml
import json
import torch
from pathlib import Path
from argparse import ArgumentParser
from rich import print as rich_print
from rich.json import JSON
from rich.console import Console
# from src.utils.typed import Betweens
from src.utils.dataset import drive_dataset, bowl2018_dataset, chasedb1_dataset, isic2017_dataset, isic2018_dataset, stare_dataset
from src.utils.transform import VisionTransformersBuilder

c = {}

def set_config(config_file: str):
    global c
    config_type = os.path.splitext(config_file)[1]
    match config_type.lower():
        case '.toml':
            with open(config_file, 'r', encoding='utf-8') as f:
                c = toml.load(f)
        case '.yaml':
            with open(config_file, 'r', encoding='utf-8') as f:
                c = yaml.safe_load(f)
        case '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                c = json.load(f)
        case _:
            raise ValueError(f"NOT support {config_type}")

def get_transforms():
    builder = VisionTransformersBuilder()
    for k, v in c['transform'].items():
        match k.upper():
            case 'RESIZE':
                builder = builder.resize(tuple(v))
            case 'HFLIP':
                builder = builder.random_horizontal_flip(*v)
            case 'VFLIP':
                builder = builder.random_vertical_flip(*v)
            case 'ROTATION':
                builder = builder.random_rotation(*v)
            case 'INVERT':
                builder = builder.random_invert(*v)
            case 'CROP':
                builder = builder.crop(tuple(v))
            case 'NORMALIZE':
                builder = builder.normalize(*v)
            case 'TO_TENSOR':
                builder = builder.to_tensor()
            case 'PIL_TO_TENSOR':
                builder = builder.PIL_to_tensor()
            case 'CONVERT_IMAGE_DTYPE':
                TMAP = {'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64, 'bfloat16': torch.bfloat16,
                        'uint8': torch.uint8, 'uint16': torch.uint16, 'uint32': torch.uint32, 'uint64': torch.uint64,
                        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64}
                ttype = TMAP[v[0]]
                builder = builder.convert_image_dtype(ttype)
    return builder.build()

def to_numpy():
    transforms = get_transforms()

    dataset_name = c['dataset']['name']
    save_dir, base_dir = Path(c['dataset']['save_dir']), Path(c['dataset']['base_dir'])
    _betweens = c['dataset']['betweens']

    if 'valid' in _betweens.keys():
        betweens = {'train': tuple(_betweens['train']), 'test': tuple(_betweens['test']), 'valid': tuple(_betweens['valid'])}
    else:
        betweens = {'train': tuple(_betweens['train']), 'test': tuple(_betweens['test'])}

    config = c['config']

    match dataset_name.lower():
        case 'drive':
            return drive_dataset.DriveDataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case 'stare':
            return stare_dataset.StareDataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case 'isic2017':
            return isic2017_dataset.ISIC2017Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case 'isic2018':
            return isic2018_dataset.ISIC2018Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case 'bowl2018':
            return bowl2018_dataset.BOWL2018Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case 'chasedb1':
            return chasedb1_dataset.ChaseDB1Dataset.to_numpy(save_dir, base_dir, betweens, transforms=transforms, **config)
        case _:
            rich_print(f'[red]No target dataset: {dataset_name}')

def data_cacher(config_file: str):
    set_config(config_file)

    console = Console()
    json_data = JSON(json.dumps(c, indent=2))
    console.print(json_data)
    
    try:
        to_numpy()
        rich_print("[green]Dataset cached successfully.")
    except Exception as e:
        rich_print(f"[red]Error caching dataset: {e}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Cache dataset to numpy format")
    parser.add_argument('-c', '--config', type=str, default='configs/cacher/data_cacher.toml', help="Config file")
    parser.add_argument('-n', '--dataset_name', type=str, help="Name of the dataset")
    parser.add_argument('--save_dir', type=str, help="The directory saving numpy data")
    parser.add_argument('--base_dir', type=str, help="The directory of dataset")

    args = parser.parse_args()

    c['dataset']['name'] = args.dataset_name
    c['dataset']['save_dir'] = args.save_dir
    c['dataset']['base_dir'] = args.base_dir

    data_cacher(args.config)
