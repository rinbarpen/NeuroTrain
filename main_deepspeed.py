#!/usr/bin/env python3
"""
DeepSpeed训练主文件
专门用于DeepSpeed分布式训练，简化代码结构
"""

from pathlib import Path
import logging
import warnings
import torch
import torch.distributed as dist

warnings.filterwarnings("ignore")

from src.config import get_config, dump_config, is_predict, is_train, is_test, set_config
from src.options import parse_args
from src.engine.deepspeed_trainer import DeepSpeedTrainer
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader
from src.utils import (
    get_train_criterion,
    load_model,
    prepare_logger,
    set_seed,
    model_info,
    model_flops,
    str2dtype,
)
from src.utils.deepspeed_utils import (
    is_deepspeed_available,
    init_deepspeed_distributed,
    load_deepspeed_config,
    create_deepspeed_config,
    get_deepspeed_rank_info,
    is_main_process,
    setup_deepspeed_logging,
    create_deepspeed_dataloader,
    cleanup_deepspeed
)
from src.constants import TrainOutputFilenameEnv, ProjectFilenameEnv


def main():
    """DeepSpeed训练主函数"""
    # 解析命令行参数
    parse_args()
    
    # 获取配置
    c = get_config()
    
    # 检查DeepSpeed可用性
    if not is_deepspeed_available():
        raise ImportError("DeepSpeed is not available. Please install deepspeed: pip install deepspeed")
    
    # 初始化DeepSpeed分布式环境
    rank_info = init_deepspeed_distributed()
    local_rank = rank_info['local_rank']
    world_size = rank_info['world_size']
    # 设置当前进程的CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # 设置DeepSpeed日志
    setup_deepspeed_logging(c.get("deepspeed", {}).get("log_level", "INFO"))
    
    # 创建输出目录
    output_dir = ProjectFilenameEnv().output_dir / c["task"] / c["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备日志记录器（只在主进程）
    if is_main_process():
        prepare_logger(output_dir, ('train', 'test', 'predict'))
    
    # 设置随机种子
    if c["seed"] >= 0:
        set_seed(c["seed"])
    
    # 基于 local_rank 绑定设备，确保每个进程使用独立GPU
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else c.get("device", "cpu")
    c["device"] = device
    set_config(c)
    is_continue_mode = False  # 用于训练中断后继续的模式
    
    # 创建模型
    model = get_model(c["model"]["name"], c["model"]["config"])
    model = model.to(device)
    if dist.is_initialized():
        dist.barrier()
    
    # 加载预训练模型
    pretrained_model = c["model"].get("pretrained")
    if pretrained_model and pretrained_model != "":
        model_path = Path(pretrained_model)
        model_params = load_model(model_path, device)
        if is_main_process():
            logging.info(f"Load pretrained model: {model_path}")
        model.load_state_dict(model_params)
    
    # 加载继续训练的检查点
    continue_checkpoint = c["model"].get("continue_checkpoint")
    if continue_checkpoint and continue_checkpoint != "":
        model_path = Path(continue_checkpoint)
        model_params = load_model(model_path, device)
        if is_main_process():
            logging.info(f"Load checkpoint: {model_path}, Now is in continue mode")
        model.load_state_dict(model_params)
        is_continue_mode = True
    
    # 仅在rank0执行数据下载准备，其他rank关闭下载标志以避免并发
    try:
        if "dataset" in c and isinstance(c["dataset"], dict):
            dataset_cfg = c["dataset"].get("config", {}) or {}
            if not is_main_process():
                if "download" in dataset_cfg:
                    dataset_cfg["download"] = False
                    c["dataset"]["config"] = dataset_cfg
                    set_config(c)
    except Exception:
        pass
    if dist.is_initialized():
        dist.barrier()

    # 获取数据加载器
    train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(use_valid=True)
    if dist.is_initialized():
        dist.barrier()
    
    # 如果没有单独的验证集，使用测试集作为验证集
    if valid_loader is None and test_loader is not None:
        valid_loader = test_loader
        if is_main_process():
            logging.info("使用测试集作为验证集进行训练")
    
    # 训练模式
    if is_train():
        train_dir = output_dir / "train"
        train_dir.mkdir(exist_ok=True)
        
        if is_main_process():
            dump_config(train_dir / "config.json")
            dump_config(train_dir / "config.toml")
            dump_config(train_dir / "config.yaml")
            logger = logging.getLogger("train")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {train_dir}/config[.json|.yaml|.toml]")
        
        # 加载或创建DeepSpeed配置
        deepspeed_config = c.get("deepspeed", {}).get("config", None)
        if deepspeed_config:
            if isinstance(deepspeed_config, str):
                ds_config = load_deepspeed_config(deepspeed_config)
            else:
                ds_config = deepspeed_config
        else:
            # 使用默认配置
            ds_config = create_deepspeed_config(
                zero_stage=c.get("deepspeed", {}).get("zero_stage", 2),
                train_batch_size=c["train"]["batch_size"],
                micro_batch_size=c["train"]["batch_size"],
                gradient_accumulation_steps=c["train"].get("grad_accumulation_steps", 1),
                learning_rate=c["train"]["optimizer"]["learning_rate"],
                weight_decay=c["train"]["optimizer"]["weight_decay"],
                cpu_offload=c.get("deepspeed", {}).get("cpu_offload", False),
                fp16=c.get("deepspeed", {}).get("fp16", False),
                bf16=c.get("deepspeed", {}).get("bf16", False)
            )
        
        # 创建DeepSpeed训练器
        handler = DeepSpeedTrainer(
            output_dir=train_dir,
            model=model,
            deepspeed_config=ds_config,
            is_continue_mode=is_continue_mode,
            local_rank=local_rank
        )
        
        criterion = get_train_criterion()
        handler.setup_trainer(criterion=criterion)
        
        # 使用DeepSpeed数据加载器
        train_loader = create_deepspeed_dataloader(
            train_loader.dataset,
            batch_size=c["train"]["batch_size"],
            shuffle=True,
            num_workers=c["dataloader"].get("num_workers", 4),
            pin_memory=c["dataloader"].get("pin_memory", True),
            drop_last=c["dataloader"].get("drop_last", True)
        )
        
        if valid_loader:
            valid_loader = create_deepspeed_dataloader(
                valid_loader.dataset,
                batch_size=c.get("valid", {}).get("batch_size", c["train"]["batch_size"]),
                shuffle=False,
                num_workers=c["dataloader"].get("num_workers", 4),
                pin_memory=c["dataloader"].get("pin_memory", True),
                drop_last=False
            )
        
        # 开始训练
        handler.train(
            num_epochs=c["train"]["epoch"],
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            early_stop="early_stopping" in c["train"],
            last_epoch=0
        )
    
    # 测试模式
    if is_test():
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        
        if is_main_process():
            dump_config(test_dir / "config.json")
            dump_config(test_dir / "config.yaml")
            dump_config(test_dir / "config.toml")
            logger = logging.getLogger("test")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {test_dir}/config[.json|.yaml|.toml]")
        
        # 加载训练好的模型进行测试
        if is_train() and not is_continue_mode:
            env = TrainOutputFilenameEnv().register(train_dir=train_dir)
            model_path = env.output_best_model_filename
            if not model_path.exists():
                model_path = env.output_last_model_filename
            
            if model_path.exists():
                model_params = load_model(model_path, device)
                if is_main_process():
                    logging.info(f"Load model for testing: {model_path}")
                model.load_state_dict(model_params)
        
        # 创建测试器（这里需要根据实际情况调整）
        from src.engine import Tester
        handler = Tester(test_dir, model)
        handler.test(test_dataloader=test_loader)
    
    # 预测模式
    if is_predict():
        predict_dir = output_dir / "predict"
        predict_dir.mkdir(exist_ok=True)
        
        if is_main_process():
            dump_config(predict_dir / "config.json")
            dump_config(predict_dir / "config.toml")
            dump_config(predict_dir / "config.yaml")
            logger = logging.getLogger("predict")
            logger.info(f"Dumping config[.json|.yaml|.toml] to the {predict_dir}/config[.json|.yaml|.toml]")
        
        # 加载训练好的模型进行预测
        if is_train() and not is_continue_mode:
            env = TrainOutputFilenameEnv().register(train_dir=train_dir)
            model_path = env.output_best_model_filename
            if not model_path.exists():
                model_path = env.output_last_model_filename
            
            if model_path.exists():
                model_params = load_model(model_path, device)
                if is_main_process():
                    logging.info(f"Load model for prediction: {model_path}")
                model.load_state_dict(model_params)
        
        # 创建预测器（这里需要根据实际情况调整）
        from src.engine import Predictor
        handler = Predictor(predict_dir, model)
        input_path = Path(c["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handler.predict(inputs, **c["predict"]["config"])
        else:
            inputs = [input_path]
            handler.predict(inputs, **c["predict"]["config"])
    
    # 记录模型信息（只在主进程执行）
    if is_main_process():
        input_sizes = c["model"]["config"]["input_sizes"]
        dtypes = c["model"]["config"].get("dtypes")
        if dtypes is not None and len(dtypes) > 0:
            dtypes = [str2dtype(dtype) for dtype in dtypes]
        else:
            dtypes = None
        
        model_info(output_dir, model, input_sizes, dtypes=dtypes)
        model_flops(output_dir, model, input_sizes)
    
    # 清理前同步，避免退出阶段心跳/Store报错噪声
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
    # 清理DeepSpeed环境
    cleanup_deepspeed()


if __name__ == "__main__":
    main()
