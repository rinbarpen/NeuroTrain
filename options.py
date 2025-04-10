import os
from argparse import ArgumentParser
import logging
from pathlib import Path
import json
import toml
import yaml
import time
from pprint import pp

from config import PREDICT_MODE, TEST_MODE, TRAIN_MODE, get_config, set_config

def parse_args():
    parser = ArgumentParser('Neuro Train')
    # model_parser
    model_parser = parser.add_argument_group(title='Model Options', description='Model options')
    model_parser.add_argument('-m', '--model', type=str, help='Model name')
    model_parser.add_argument('-mc', '--model_config', type=dict, help='Model config')
    # Train
    train_parser = parser.add_argument_group(title='Train', description='Train options')
    train_parser.add_argument('-e', '--epoch', type=int, help='epoch')
    train_parser.add_argument('-lr', '--learning_rate', type=float, help='learning_rate')
    train_parser.add_argument('--eps', type=float, help='eps')
    train_parser.add_argument('--weight_decay', type=float, help='weight_decay')
    train_parser.add_argument('--save_every_n_epoch', type=int, required=False, help='save_every_n_epoch')
    train_parser.add_argument('-ab', '--augment_boost', type=str, help='augment_boost')
    ## amp mode: bf16 int8 ..
    # bfloat16, float16, 
    train_parser.add_argument('--amp', type=str, default='none', help='set amp mode')
    ## for lr_scheduler
    train_parser.add_argument('--lr_scheduler', action='store_true', default=False, help='lr_scheduler')
    train_parser.add_argument('--warmup', type=int, help='lr_scheduler epoch')
    train_parser.add_argument('--warmup_lr', type=float, help='lr_scheduler learning_rate')
    ## for early_stopping
    train_parser.add_argument('-es', '--early_stopping', action='store_true', default=False, help='early_stopping')
    train_parser.add_argument('--patience', type=int, default=3, help='patience')
    ## common
    train_parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch_size')
    train_parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    train_parser.add_argument('--data', type=str, help='dataset names')
    train_parser.add_argument('--data_dir', type=str, help='dataset directory')
    # Predict
    predict_parser = parser.add_argument_group(title='Predict Options', description='Predict options')
    predict_parser.add_argument('-i', '--input', type=str, help='input')
    # Common
    parser.add_argument('--wandb', action='store_true', default=False, help='setup wandb')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    parser.add_argument('-c', '--config', type=str, help='Configuration of Train or Test or Predict')
    # parser.add_argument('--dump', default=False)
    parser.add_argument('--seed', type=int, default=42, help='Seed of Train or Test or Predict')
    parser.add_argument('--device', type=str, default='cuda', help='Devices for Training or Testing or Predicting')
    parser.add_argument('--train', action='store_true', default=False, help='Train')
    parser.add_argument('--test', action='store_true', default=False, help='Test')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict')
    parser.add_argument('--output_dir', type=str, help='Output Directory')
    parser.add_argument('--continue_checkpoint', type=str, help='Load Model Checkpoint')
    parser.add_argument('--task', type=str, help='Task name')

    args = parser.parse_args()
    CONFIG = get_config()
    if args.config:
        config_file = Path(args.config)
        CONFIG = load_config(config_file)

    if args.seed:
        CONFIG['seed'] = args.seed
    if args.device:
        CONFIG['device'] = args.device
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    if args.task:
        CONFIG['task'] = args.task
    if args.verbose:
        CONFIG['private']['verbose'] = args.verbose

    if args.continue_checkpoint:
        CONFIG['model']['continue_checkpoint'] = args.continue_checkpoint
    if args.model:
        CONFIG['model']['name'] = args.model
        if args.model_config: # n_channels=1;n_classes=1
            for arg in args.model_config.split(';'):
                k, v = arg.split('=')[0], arg.split('=')[1]
                model_config = {k.strip(): v.strip()}
            CONFIG['model']['config'] = model_config
    if args.train:
        if args.data:
            CONFIG['train']['dataset']['name'] = args.data
        if args.data_dir:
            CONFIG['train']['dataset']['path'] = args.data_dir

        if args.batch_size:
            CONFIG['train']['batch_size'] = args.batch_size
        if args.epoch:
            CONFIG['train']['epoch'] = args.epoch
        if args.eps:
            CONFIG['train']['optimizer']['eps'] = args.eps
        if args.learning_rate:
            CONFIG['train']['optimizer']['learning_rate'] = args.learning_rate
        if args.augment_boost:
            CONFIG['train']['augment_boost']['enabled'] = True
            for arg in args.augment_boost.split(';'):
                k, v = arg.split('=')[0], arg.split('=')[1]
                augment_boost = {k.strip(): v.strip()}
            CONFIG['train']['augment_boost']['config'] = augment_boost
        if args.save_every_n_epoch:
            CONFIG['train']['save_every_n_epoch'] = args.save_every_n_epoch
        if args.weight_decay:
            CONFIG['train']['optimizer']['weight_decay'] = args.weight_decay
        if args.amp != 'none':
            CONFIG['train']['scaler']['enabled'] = True
            CONFIG['train']['scaler']['compute_type'] = args.amp
            logging.info('set up amp mode: {}'.format(args.amp))
        if args.early_stopping:
            CONFIG['train']['early_stopping']['enabled'] = True
            CONFIG['train']['early_stopping']['patience'] = args.patience
            logging.info('set up early_stopping with patience={}'.format(args.patience))
        if args.lr_scheduler:
            CONFIG['train']['lr_scheduler']['enabled'] = True
            if args.warmup:
                CONFIG['train']['lr_scheduler']['warmup'] = args.warmup
            if args.warmup_lr:
                CONFIG['train']['lr_scheduler']['warmup_lr'] = args.warmup_lr
    if args.test:
        if args.data:
            CONFIG['test']['dataset']['name'] = args.data
        if args.data_dir:
            CONFIG['test']['dataset']['path'] = args.data_dir
        if args.batch_size:
            CONFIG['test']['batch_size'] = args.batch_size
    if args.input:
        CONFIG['predict']['input'] = args.input
    if args.train:
        CONFIG['private']['mode'] |= TRAIN_MODE
    if args.test:
        CONFIG['private']['mode'] |= TEST_MODE
    if args.predict:
        CONFIG['private']['mode'] |= PREDICT_MODE

    run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    CONFIG['run_id'] = run_id

    if args.wandb:
        import wandb
        CONFIG['private']['wandb'] = args.wandb
        project = CONFIG['task']
        entity = CONFIG['entity']
        wandb.init(project=project, entity=entity, config=CONFIG)

    pp(CONFIG)
    set_config(CONFIG)

def load_config(filename: Path):
    match filename.suffix:
        case '.json':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = json.load(f)
        case '.yaml'|'.yml':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        case '.toml':
            with filename.open(mode='r', encoding='utf-8') as f:
                config = toml.load(f)

    logging.info(f'Loading config from {filename}')
    return config

def dump_config(filename: Path):
    CONFIG = get_config()
    match filename.suffix:
        case '.json':
            with filename.open(mode='w', encoding='utf-8') as f:
                json.dump(CONFIG, f)
        case '.toml':
            with filename.open(mode='w', encoding='utf-8') as f:
                toml.dump(CONFIG, f)
        case '.yaml'|'.yml':
            with filename.open(mode='w', encoding='utf-8') as f:
                yaml.safe_dump_all(CONFIG, f, allow_unicode=True, encoding='utf-8')
