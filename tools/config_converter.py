import os.path as osp
import json
import toml
import yaml
from argparse import ArgumentParser
from enum import Enum

class FileType(Enum):
    NONE = 0,
    JSON = 1,
    YAML = 2,
    TOML = 3,


def get_file_type(filename: str):
    ext = osp.splitext(osp.basename(filename))[1]
    match ext:
        case '.json':
            return FileType.JSON
        case '.yaml'|'.yml':
            return FileType.YAML
        case '.toml':
            return FileType.TOML
    return FileType.NONE

if __name__ == '__main__':
    parser = ArgumentParser('Configuration Converter')
    parser.add_argument('-i', '--input', required=True, help='input configuration')
    parser.add_argument('-o', '--output', required=True, help='output configuration')

    args = parser.parse_args()
    input_type = get_file_type(args.input)
    output_type = get_file_type(args.output)

    match input_type:
        case FileType.JSON:
            with open(args.input, mode='r', encoding='utf-8') as f:
                config = json.load(f)
        case FileType.YAML:
            with open(args.input, mode='r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        case FileType.TOML:
            with open(args.input, mode='r', encoding='utf-8') as f:
                config = toml.load(f)

    match output_type:
        case FileType.JSON:
            json.dump(config, args.output)
        case FileType.YAML:
            with open(args.output, mode='w', encoding='utf-8') as f:
                yaml.dump(config, f)
        case FileType.TOML:
            with open(args.output, mode='w', encoding='utf-8') as f:
                toml.dump(config, f)
