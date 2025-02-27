import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--project')
args = parser.parse_args()

os.system(f'python runtime -m {args.project}')
os.system(f'venv')
