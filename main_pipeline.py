import subprocess
import json
from argparse import ArgumentParser
import logging
from pathlib import Path
import toml
import yaml

from utils import prepare_logger

# def prepare_logger():
#     log_colors = {
#         'DEBUG': 'cyan',
#         'INFO': 'green',
#         'WARNING': 'yellow',
#         'ERROR': 'red',
#         'FATAL': 'bold_red',
#     }
#     formatter = colorlog.ColoredFormatter(
#         '%(log_color)s %(asctime)s %(levelname)s | %(name)s | %(message)s',
#         log_colors=log_colors
#     )

#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     os.makedirs('logs', exist_ok=True)
#     filename = os.path.join('logs', time.strftime('%Y-%m-%d %H_%M_%S.log', time.localtime()))
#     file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
#     file_handler.setFormatter(logging.Formatter(
#         '%(asctime)s %(levelname)s | %(name)s | %(message)s'
#     ))

#     log_level = logging.DEBUG
#     root_logger = logging.getLogger()
#     root_logger.setLevel(log_level)
#     root_logger.addHandler(console_handler)
#     root_logger.addHandler(file_handler)


if __name__ == '__main__':
    prepare_logger()

    parser = ArgumentParser("Train Pipeline")
    parser.add_argument('-c', '--config', required=True, help="pipeline config file")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        from config import PIPELINE_CONFIG_DIR
        config_path = Path(PIPELINE_CONFIG_DIR) / args.config

    match config_path.suffix:
        case '.toml':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = toml.load(f)
        case '.yaml'|'.yml':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = yaml.safe_load(f)
        case '.json':
            with config_path.open('r', encoding='utf-8') as f:
                pipeline_config = json.load(f)
        case _:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")

    for name, config in pipeline_config.items():
        config_file = config['config']
        ext_args = config['ext_args']

        try:
            process = subprocess.Popen(["main.py", "-c", config_file, *ext_args])
            process.wait()
        except Exception as e:
            logging.error(f"Name: {name}, Config: {config_file}, Exception: {e}")
