import shutil
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(prog='cleanup logs and output')
    parser.add_argument('--log', action='store_true', help='cleanup logs')
    parser.add_argument('--output', action='store_true', help='cleanup output')

    args = parser.parse_args()
    if args.log:
        shutil.rmtree('./logs', ignore_errors=True)
    if args.output:
        shutil.rmtree('./output', ignore_errors=True)
