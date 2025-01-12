import subprocess
import json
from argparse import ArgumentParser
import logging

from utils.util import prepare_logger

if __name__ == '__main__':
    prepare_logger()

    parser = ArgumentParser("Train Pipeline")
    parser.add_argument('-c', '--config', required=True, help="pipeline config file")

    args = parser.parse_args()

    config = json.load(args.config)

    for i, train_config in enumerate(config):
        config_file = train_config['config']
        ext_args = train_config['ext_args']
        
        try:
            process = subprocess.Popen(["main.py", "-c", config_file, *ext_args])
            process.wait()
        except Exception as e:
            logging.error(f"Index: {i}, Config: {args.config}, Exception: {e}")
