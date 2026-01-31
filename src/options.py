import os
import sys
import time
import json
from pathlib import Path
from rich.json import JSON
from rich.console import Console
from argparse import ArgumentParser

import wandb

from src.config import PREDICT_MODE, TEST_MODE, TRAIN_MODE, set_config, load_config, SINGLE_CONFIG_DIR

def parse_args():
    parser = ArgumentParser('NeuroTrain')
    # model_parser
    model_parser = parser.add_argument_group(title='Model Options', description='Model options')
    model_parser.add_argument('-m', '--model', type=str, help='Model name')
    # Train
    train_parser = parser.add_argument_group(title='Train', description='Train options')
    train_parser.add_argument('-e', '--epoch', type=int, help='epoch')
    train_parser.add_argument('-lr', '--learning_rate', type=float, help='learning_rate')
    train_parser.add_argument('--weight_decay', type=float, help='weight_decay')
    train_parser.add_argument('--save_period', type=float, help='save_period')
    train_parser.add_argument('--grad_accumulation_steps', type=int, help='grad_accumulation_steps')
    ## amp mode: bf16 int8 ..
    # bfloat16, float16, 
    train_parser.add_argument('--amp', type=str, default='none', help='set amp mode: [float16, bfloat16]')
    ## for lr_scheduler
    train_parser.add_argument('--lr_scheduler', action='store_true', help='lr_scheduler')
    train_parser.add_argument('--warmup', type=int, help='warmup epoch')
    train_parser.add_argument('--warmup_lr', type=float, help='warmup learning_rate')
    ## for early_stopping
    train_parser.add_argument('--early_stopping', action='store_true', help='early_stopping')
    train_parser.add_argument('--patience', type=int, help='patience')
    ## common
    train_parser.add_argument('-b', '--batch_size', type=int, help='batch_size')
    train_parser.add_argument('--num_workers', type=int, help='dataset num_workers')
    train_parser.add_argument('--data', type=str, help='dataset names (not recommend to use)')
    train_parser.add_argument('--data_dir', type=str, help='dataset directory')
    # Predict
    predict_parser = parser.add_argument_group(title='Predict Options', description='Predict options')
    predict_parser.add_argument('-i', '--input', type=str, help='input')
    # Lora, TODO: Support it!
    lora_parser = parser.add_argument_group(title='Lora Options', description='Lora options')
    lora_parser.add_argument('--lora', action='store_true', help='use lora')
    lora_parser.add_argument('--lora_r', type=int, default=8, help='lora_r')
    lora_parser.add_argument('--lora_alpha', type=int, default=16, help='lora_alpha')
    lora_parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora_dropout')
    # target module lora finetuned default
    lora_parser.add_argument('--target_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'], help='target_modules for lora finetuned') 
    # Common
    parser.add_argument('--wandb', action='store_true', help='setup wandb')
    parser.add_argument('--verbose', action='store_true', help='setup verbose mode')
    parser.add_argument('--debug', action='store_true', help='setup debug mode')
    parser.add_argument('-c', '--config', type=str, help='configuration of Train or Test or Predict')
    # parser.add_argument('--dump', default=False)
    parser.add_argument('--seed', type=int, help='seed of Train or Test or Predict')
    # parser.add_argument('--device', type=str, nargs='+', default=[0], help='devices for Training or Testing or Predicting')
    parser.add_argument('--device', type=str, default='cuda:0', help='devices for Training or Testing or Predicting')
    parser.add_argument('--train', action='store_true', default=False, help='Train mode')
    parser.add_argument('--test', action='store_true', default=False, help='Test mode')
    parser.add_argument('--predict', action='store_true', default=False, help='Predict mode')
    parser.add_argument('--check', action='store_true', default=False, help='Check mode')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--continue_id', type=str, help='continue with $run_id$')
    parser.add_argument('--continue_from', type=str, help='full recovery from a run dir (e.g. runs/task/run_id)')
    parser.add_argument('--continue_checkpoint', type=str, help='continue checkpoint path')
    parser.add_argument('--pretrained', type=str, help='load model checkpoint')
    parser.add_argument('--task', type=str, help='task name')
    parser.add_argument('--run_id', type=str, help='run id')
    parser.add_argument('--monitor', action='store_true', default=False, help='Enable training monitor')
    parser.add_argument('--web_monitor', action='store_true', default=False, help='Enable web monitor dashboard')
    # for distributed launchers (torchrun/deepspeed)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed by launcher')

    # parser.add_argument('--data_cacher', action='store_true', help='tool: data cacher')
    # TODO: support for future!
    parser.add_argument('--metrics', nargs='+', default=[], help='metrics, support iou-family dice accuracy f1 recall auc mAP@[threshold] mF1@[threshold]')
    
    args = parser.parse_args()

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
        CONFIG['private']['log']['verbose'] = args.verbose
    if args.debug:
        CONFIG['private']['log']['debug'] = args.debug

    if args.data:
        CONFIG['dataset'] = {"name": args.data, "root_dir": args.data_dir}
    if args.num_workers:
        CONFIG['dataloader']['num_workers'] = args.num_workers
    if args.continue_from:
        # Full recovery from any run directory: set checkpoint/ext paths and recovery_dir
        continue_from = Path(args.continue_from)
        weights_dir = continue_from / "train" / "weights"
        recovery_dir = continue_from / "train" / "recovery"
        last_pt = weights_dir / "last.pt"
        stop_pt = weights_dir / "stop.pt"
        model_ckpt = last_pt if last_pt.exists() else (stop_pt if stop_pt.exists() else last_pt)
        ext_ckpt = weights_dir / "last.ext.pt" if last_pt.exists() else (weights_dir / "stop.ext.pt" if stop_pt.exists() else weights_dir / "last.ext.pt")
        CONFIG['model']['continue_checkpoint'] = str(model_ckpt)
        CONFIG['model']['continue_ext_checkpoint'] = str(ext_ckpt)
        CONFIG.setdefault('private', {})['recovery_dir'] = str(recovery_dir)
    elif args.continue_id:
        task_dir = os.path.join(CONFIG['output_dir'], CONFIG['task'])
        if args.continue_id.lower() == 'auto':
            continue_id = os.listdir(task_dir)[-1] # select the last run_id
        else:
            continue_id = args.continue_id
        checkpoint_filename = os.path.join(task_dir, continue_id, 'train', 'weights', 'last.pt')
        CONFIG['model']['continue_checkpoint'] = checkpoint_filename
        ext_checkpoint_filename = os.path.join(task_dir, continue_id, 'train', 'weights', 'last.ext.pt')
        CONFIG['model']['continue_ext_checkpoint'] = ext_checkpoint_filename
        CONFIG['run_id'] = continue_id

    if args.pretrained:
        CONFIG['model']['pretrained'] = args.pretrained
    if args.model:
        CONFIG['model']['name'] = args.model
    if args.train:
        if args.batch_size:
            CONFIG['train']['batch_size'] = args.batch_size
        if args.epoch:
            CONFIG['train']['epoch'] = args.epoch
        if args.grad_accumulation_steps:
            CONFIG['train']['grad_accumulation_steps'] = args.grad_accumulation_steps
        if args.learning_rate:
            CONFIG['train']['optimizer']['learning_rate'] = args.learning_rate
        if args.weight_decay:
            CONFIG['train']['optimizer']['weight_decay'] = args.weight_decay
        if args.save_period:
            CONFIG['train']['save_period'] = args.save_period
        if args.weight_decay:
            CONFIG['train']['optimizer']['weight_decay'] = args.weight_decay
        if args.amp != 'none':
            CONFIG['train']['scaler'] = {}
            CONFIG['train']['scaler']['compute_type'] = args.amp
            print('set up amp mode: {}'.format(args.amp))
        if args.early_stopping:
            CONFIG['train']['early_stopping'] = {}
            if args.patience:
                CONFIG['train']['early_stopping']['patience'] = args.patience
                print('set up early_stopping with patience={}'.format(args.patience))
        if args.lr_scheduler:
            CONFIG['train']['lr_scheduler'] = {}
            if args.warmup:
                CONFIG['train']['lr_scheduler']['warmup'] = args.warmup
            if args.warmup_lr:
                CONFIG['train']['lr_scheduler']['warmup_lr'] = args.warmup_lr
    if args.test:
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

    if args.monitor or args.web_monitor:
        CONFIG.setdefault('monitor', {})
        CONFIG['monitor']['enabled'] = True
    if args.web_monitor:
        CONFIG.setdefault('monitor', {})
        CONFIG['monitor']['web'] = True

    if args.run_id and not args.continue_checkpoint:
        CONFIG['run_id'] = args.run_id
    elif CONFIG['run_id'] is None or CONFIG['run_id'] == '':
        CONFIG['run_id'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    if args.check:
        CONFIG['private']['mode'] |= (TRAIN_MODE | TEST_MODE | PREDICT_MODE)
        CONFIG['train']['batch_size'] = 2
        CONFIG['train']['epoch'] = 3
        CONFIG['train']['save_period'] = 1

    if args.wandb:
        try:
            CONFIG['private']['wandb'] = True
            project = CONFIG['task']
            entity = CONFIG['entity']
            wandb.init(project=project, entity=entity, config=CONFIG)
        except Exception:
            CONFIG['private']['wandb'] = False
            print("wandb isn't installed, disable to launch wandb")
    
    console = Console()
    json_data = JSON(json.dumps(CONFIG, indent=2))
    console.print(json_data)
    set_config(CONFIG)
