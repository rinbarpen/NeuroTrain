from argparse import ArgumentParser
from pathlib import Path
import colorlog

def parse_args():
    parser = ArgumentParser('')
    parser.add_argument('-m', '--model', required=True)

    args = parser.parse_args()

    return args

def load_config(filename: Path):
    from config import CONFIG
    import json
    with filename.open(mode='r', encoding='utf-8') as f:
        CONFIG = json.load(f)

    colorlog.info('Loading config from %s' % filename.absolute())

def dump_config(filename: Path):
    from config import CONFIG
    import json
    with filename.open(mode='w', encoding='utf-8') as f:
        json.dump(CONFIG, f)

    colorlog.info('Dumping config to %s' % filename.absolute())
