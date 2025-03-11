import subprocess
import json
from argparse import ArgumentParser
import logging
from pathlib import Path
import toml
import yaml

from utils.util import prepare_logger

if __name__ == '__main__':
    prepare_logger()

    parser = ArgumentParser("Train Pipeline")
    parser.add_argument('-c', '--config', required=True, help="pipeline config file")

    args = parser.parse_args()

    config_path = Path(args.config)
    match config_path.suffix:
        case '.toml':
            with config_path.open('r', encoding='utf-8') as f:
                config = toml.load(f)
        case '.yaml'|'.yml':
            with config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        case '.json':
            with config_path.open('r', encoding='utf-8') as f:
                config = json.load(args.config)
        case _:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")


    for name, config in config.items():
        config_file = config['config']
        ext_args = config['ext_args']
        
        try:
            process = subprocess.Popen(["main.py", "-c", config_file, *ext_args])
            process.wait()
        except Exception as e:
            logging.error(f"Name: {name}, Config: {config_path}, Exception: {e}")
