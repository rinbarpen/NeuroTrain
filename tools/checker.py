from argparse import ArgumentParser
import torch


def main():
    parse_args()


def check_cuda():
    print(f"Cuda is available: {torch.cuda.is_available()}")
    print(f"The number of cuda is {torch.cuda.device_count()}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='check cuda')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
