from pathlib import Path

import logging
import torch
from torch import nn
import warnings
import torch.distributed as dist

warnings.filterwarnings("ignore")

from src.config import get_config, dump_config, is_predict, is_train, is_test
from src.options import parse_args
from src.engine import Trainer, Tester, Predictor
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader
from src.utils import (
    get_train_tools,
    get_train_criterion,
    load_model,
    load_model_ext,
    prepare_logger,
    set_seed,
    model_info,
    model_flops,
    str2dtype,
)
from src.utils.ddp_utils import (
    init_ddp_distributed,
    is_main_process,
    cleanup_ddp,
    setup_ddp_logging
)
from src.constants import TrainOutputFilenameEnv, ProjectFilenameEnv

if __name__ == "__main__":
    parse_args()
    
    c = get_config()
    
    # 检查是否使用 DDP
    use_ddp = c.get("ddp", {}).get("enabled", False)
    
    # 初始化 DDP 分布式环境
    if use_ddp:
        # 初始化分布式环境
        rank_info = init_ddp_distributed()
        local_rank = rank_info['local_rank']
        world_size = rank_info['world_size']
        
        # 设置 DDP 日志
        setup_ddp_logging(c.get("ddp", {}).get("log_level", "INFO"))
        
        # 只在主进程创建输出目录
        if is_main_process():
            output_dir = ProjectFilenameEnv().output_dir / c["task"] / c["run_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = ProjectFilenameEnv().output_dir / c["task"] / c["run_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = ProjectFilenameEnv().output_dir / c["task"] / c["run_id"]
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        predict_dir = output_dir / "predict"
        output_dir.mkdir(parents=True, exist_ok=True)
        local_rank = 0
        world_size = 1
    
    # 准备日志记录器
    if not use_ddp or is_main_process():
        prepare_logger(output_dir, ('train', 'test', 'predict'))

    if c["seed"] >= 0:
        set_seed(c["seed"])
    
    device = c["device"]
    is_multigpu = "," in device
    if is_multigpu and not use_ddp:
        dist.barrier()

    is_continue_mode = False  # use this mode if encountering crash while training

    model = get_model(c["model"]["name"], c["model"]["config"])
    model = model.to(device)
    pretrained_model = c["model"].get("pretrained")
    continue_checkpoint = c["model"].get("continue_checkpoint")
    if pretrained_model and pretrained_model != "":
        model_path = Path(pretrained_model)
        model_params = load_model(model_path, device)
        logging.info(f"Load model: {model_path}")
        model.load_state_dict(model_params)

    if continue_checkpoint and continue_checkpoint != "":
        model_path = Path(continue_checkpoint)
        model_params = load_model(model_path, device)
        logging.info(f"Load model: {model_path}, Now is in continue mode")
        model.load_state_dict(model_params)
        is_continue_mode = True

    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(use_valid=True)
    
    # 如果没有单独的验证集，使用测试集作为验证集
    if valid_loader is None and test_loader is not None:
        valid_loader = test_loader
        if is_main_process() or not use_ddp:
            logging.info("使用测试集作为验证集进行训练")
    
    if is_train():
        if use_ddp:
            # DDP 训练
            train_dir = output_dir / "train"
            train_dir.mkdir(exist_ok=True)
            
            if is_main_process():
                dump_config(train_dir / "config.json")
                dump_config(train_dir / "config.toml")
                dump_config(train_dir / "config.yaml")
                logger = logging.getLogger("train")
                logger.info(f"Dumping config[.json|.yaml|.toml] to the {train_dir}/config[.json|.yaml|.toml]")
            
            # 使用 DDP 包装模型
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
            
            # 创建标准训练器
            handler = Trainer(train_dir, model, is_continue_mode=is_continue_mode)
            
            # 获取训练工具
            tools = get_train_tools(model.module if hasattr(model, 'module') else model)
            optimizer = tools["optimizer"]
            lr_scheduler = tools["lr_scheduler"]
            scaler = tools["scaler"]
            criterion = get_train_criterion()
            
            # 设置训练器
            handler.setup_trainer(criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler)
            
            # 使用分布式采样器
            from torch.utils.data.distributed import DistributedSampler
            if train_loader is not None:
                train_sampler = DistributedSampler(train_loader.dataset)
                train_loader = torch.utils.data.DataLoader(
                    train_loader.dataset,
                    batch_size=c["train"]["batch_size"],
                    sampler=train_sampler,
                    num_workers=c["dataloader"].get("num_workers", 4),
                    pin_memory=c["dataloader"].get("pin_memory", True),
                    drop_last=c["dataloader"].get("drop_last", True)
                )
            
            if valid_loader is not None:
                valid_sampler = DistributedSampler(valid_loader.dataset, shuffle=False)
                valid_loader = torch.utils.data.DataLoader(
                    valid_loader.dataset,
                    batch_size=c["valid"].get("batch_size", c["train"]["batch_size"]),
                    sampler=valid_sampler,
                    num_workers=c["dataloader"].get("num_workers", 4),
                    pin_memory=c["dataloader"].get("pin_memory", True),
                    drop_last=False
                )
            
            handler.train(
                num_epochs=c["train"]["epoch"],
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                early_stop="early_stopping" in c["train"],
                last_epoch=0
            )
        else:
            # 标准训练
            train_dir = output_dir / "train"
            train_dir.mkdir(exist_ok=True)
            dump_config(train_dir / "config.json")
            dump_config(train_dir / "config.toml")
            dump_config(train_dir / "config.yaml")
            logger = logging.getLogger("train")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {train_dir}/config[.json|.yaml|.toml]")

            finished_epoch = 0
            tools = get_train_tools(model)
            optimizer = tools["optimizer"]
            lr_scheduler = tools["lr_scheduler"]
            scaler = tools["scaler"]
            criterion = get_train_criterion()

            if is_continue_mode and c["model"]["continue_ext_checkpoint"] != "":
                model_ext_path = Path(c["model"]["continue_ext_checkpoint"])
                model_ext_params = load_model_ext(model_ext_path, device)
                finished_epoch = model_ext_params["epoch"]
                logger.info(f"Continue Train from {finished_epoch}")

                try:
                    optimizer.load_state_dict(model_ext_params["optimizer"])
                    if lr_scheduler and model_ext_params["lr_scheduler"]:
                        lr_scheduler.load_state_dict(model_ext_params["lr_scheduler"])
                    if scaler and model_ext_params["scaler"]:
                        scaler.load_state_dict(model_ext_params["scaler"])
                except Exception as e:
                    logger.error(f"{e} WHILE LOADING EXT CHECKPOINT")

            handler = Trainer(train_dir, model, is_continue_mode=is_continue_mode)
            handler.setup_trainer(criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler)
            handler.train(
                num_epochs=c["train"]["epoch"],
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                early_stop="early_stopping" in c["train"],
                last_epoch=finished_epoch,
            )

    if is_test():
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        
        if is_main_process() or not use_ddp:
            dump_config(test_dir / "config.json")
            dump_config(test_dir / "config.yaml")
            dump_config(test_dir / "config.toml")
            logger = logging.getLogger("test")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {test_dir}/config[.json|.yaml|.toml]")

        if is_train() and not is_continue_mode:
            train_dir = output_dir / "train"  # 确保train_dir被定义
            env = TrainOutputFilenameEnv().register(train_dir=train_dir)
            model_path = env.output_best_model_filename
            if not model_path.exists():
                model_path = env.output_last_model_filename
            
            if model_path.exists():
                model_params = load_model(model_path, device)
                if is_main_process() or not use_ddp:
                    logger.info(f"Load model: {model_path}")
                model.load_state_dict(model_params)

        handler = Tester(test_dir, model)
        if test_loader is not None:
            handler.test(test_dataloader=test_loader)

    if is_predict():
        predict_dir = output_dir / "predict"
        predict_dir.mkdir(exist_ok=True)
        
        if is_main_process() or not use_ddp:
            dump_config(predict_dir / "config.json")
            dump_config(predict_dir / "config.toml")
            dump_config(predict_dir / "config.yaml")
            logger = logging.getLogger("predict")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {predict_dir}/config[.json|.yaml|.toml]")

        if is_train() and not is_continue_mode:
            train_dir = output_dir / "train"  # 确保train_dir被定义
            env = TrainOutputFilenameEnv().register(train_dir=train_dir)
            model_path = env.output_best_model_filename
            if not model_path.exists():
                model_path = env.output_last_model_filename
            
            if model_path.exists():
                model_params = load_model(model_path, device)
                if is_main_process() or not use_ddp:
                    logger.info(f"Load model: {model_path}")
                model.load_state_dict(model_params)

        handler = Predictor(predict_dir, model)
        input_path = Path(c["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handler.predict(inputs, **c["predict"]["config"])
        else:
            inputs = [input_path]
            handler.predict(inputs, **c["predict"]["config"])

    # get_input_size
    input_sizes = c["model"]["config"]["input_sizes"] # get a list
    dtypes = c["model"]["config"].get("dtypes")
    if dtypes is not None and len(dtypes) > 0:
        dtypes = [str2dtype(dtype) for dtype in dtypes]
    else:
        dtype = None

    # 清理 DDP 环境
    if use_ddp:
        cleanup_ddp()
    
    # 模型信息记录（只在主进程执行）
    if not use_ddp or is_main_process():
        model_info(output_dir, model, input_sizes, dtypes=dtypes)
        model_flops(output_dir, model, input_sizes)
