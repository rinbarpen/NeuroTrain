import torch
from argparse import ArgumentParser
from pathlib import Path

from utils.util import load_model

if __name__ == '__main__':
    parser = ArgumentParser('Model Params Extractor')
    parser.add_argument('-m', '--model', required=True, help='Model File')
    parser.add_argument('-o', '--output', required=True, help='Output Model File')

    args = parser.parse_args()

    model_path = Path(args.model)
    model_params = load_model(model_path, 'cpu')

    save_model_path = Path(args.output) / model_path.name
    with save_model_path.open(mode='r', encoding='utf-8') as f:
        torch.save(model_params, f)
