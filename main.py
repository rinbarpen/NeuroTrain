import colorama
colorama.init()

from config import CONFIG, IS_TEST, IS_TRAIN, IS_INFERENCE
from model_operation import Trainer, Tester, Predictor
from options import parse_args
from utils.utils import prepare_logger

if __name__ == '__main__':
    prepare_logger()
    args = parse_args()

    if IS_TRAIN:
        handle = Trainer()
        handle.train()
        exit(0)

    if IS_TEST:
        handle = Tester()
        handle.test()
        exit(0)

    if IS_INFERENCE:
        handle = Predictor()
        handle.predict(args.input)
        exit(0)

    exit(1)
