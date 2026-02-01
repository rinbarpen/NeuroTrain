from pathlib import Path
import json

import copy
import logging
import os
import torch
from torch import nn
import warnings
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

warnings.filterwarnings("ignore")

from src.config import get_config, dump_config, is_predict, is_train, is_test
from src.options import parse_args
from src.engine import Trainer, Inferencer, Predictor, get_predictor, MultiModelPredictor
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader
from src.utils import (
    get_train_tools,
    get_train_criterion,
    load_model,
    load_model_ext,
    prepare_logger,
    set_seed,
    print_model_info_block,
    str2dtype,
)
from src.utils.ddp_utils import (
    init_ddp_distributed,
    is_main_process,
    cleanup_ddp,
    setup_ddp_logging
)
from src.constants import TrainOutputFilenameEnv, ProjectFilenameEnv
from src.pretrained_manager import get_pretrained_path, resolve_pretrained_to_file


def _should_auto_start_monitor(monitor_conf: dict) -> bool:
    auto_start_default = monitor_conf.get('enabled', False) or monitor_conf.get('web', False)
    auto_start = monitor_conf.get('auto_start_latest', auto_start_default)
    env_toggle = os.environ.get("NEUROTRAIN_MONITOR_AUTO_START")
    if env_toggle is not None:
        auto_start = env_toggle.strip().lower() not in {"0", "false", "off", "no"}
    return bool(auto_start)


def _resolve_monitor_base_url(monitor_conf: dict) -> str:
    base_url = monitor_conf.get('api_base') or os.environ.get("NEUROTRAIN_MONITOR_API")
    if base_url:
        return base_url.rstrip("/")
    return "http://127.0.0.1:5000"


def _resolve_monitor_timeout(monitor_conf: dict) -> float:
    timeout = monitor_conf.get('api_timeout')
    if timeout is None:
        timeout_env = os.environ.get("NEUROTRAIN_MONITOR_TIMEOUT")
        if timeout_env is not None:
            try:
                timeout = float(timeout_env)
            except ValueError:
                timeout = None
    if timeout is None:
        timeout = 2.0
    return float(timeout)


def _notify_monitor_auto_start(config: dict, output_dir: Path) -> None:
    monitor_conf = config.get('monitor') or {}
    if not _should_auto_start_monitor(monitor_conf):
        return
    try:
            import requests  # type: ignore
    except Exception:
        logging.getLogger('monitor').warning("requests 模块不可用，跳过监控自动通知")
        return

    base_url = _resolve_monitor_base_url(monitor_conf)
    start_path = monitor_conf.get('api_start_path', '/api/control/start')
    url = f"{base_url}{start_path}"
    timeout = _resolve_monitor_timeout(monitor_conf)

    payload = {
        "task": config.get("task"),
        "entity": config.get("entity"),
        "run_id": config.get("run_id"),
        "output_dir": str(output_dir),
        "pid": os.getpid(),
        "device": config.get("device"),
        "timestamp": datetime.utcnow().isoformat(),
        "modes": {
            "train": is_train(),
            "test": is_test(),
            "predict": is_predict(),
        },
    }

    train_conf = config.get("train") or {}
    if train_conf:
        payload["train"] = {
            "epoch": train_conf.get("epoch"),
            "batch_size": train_conf.get("batch_size"),
        }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logging.getLogger('monitor').info("已通知监控服务自动启动：%s", url)
    except Exception as exc:
        logging.getLogger('monitor').warning("监控自动启动通知失败：%s", exc)

def _wandb_alert_on_failure(exc: BaseException) -> None:
    """Send wandb alert when run fails, if wandb is enabled."""
    try:
        from src.config import get_config
        c = get_config()
        if c.get("private", {}).get("wandb") and c.get("private", {}).get("wandb_alerts", True):
            import wandb
            wandb.alert(title="Run failed", text=str(exc))
    except Exception:
        pass


def _run_main():
    """Main execution body; wrapped in try/except for wandb alert on failure."""
    c = get_config()
    async_workers = c['train'].get('async_workers')
    if async_workers is None or async_workers <= 0:
        async_workers = os.cpu_count() or 1
    async_executor = ThreadPoolExecutor(
        max_workers=async_workers,
        thread_name_prefix='trainer_async',
    )

    # 检查是否使用 DDP
    use_ddp = c.get("ddp", {}).get("enabled", False)
    device = c["device"]

    # 初始化 DDP 分布式环境
    if use_ddp:
        # 初始化分布式环境
        rank_info = init_ddp_distributed()
        local_rank = rank_info['local_rank']
        world_size = rank_info['world_size']
        device = rank_info['device']
        c['device'] = device

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
        _notify_monitor_auto_start(c, output_dir)

    if c["seed"] >= 0:
        set_seed(c["seed"])

    is_multigpu = "," in device
    if is_multigpu and not use_ddp:
        dist.barrier()

    is_continue_mode = False  # use this mode if encountering crash while training

    model = get_model(c["model"]["name"], c["model"]["config"])
    model = model.to(device)
    pretrained_model = c["model"].get("pretrained")
    continue_checkpoint = c["model"].get("continue_checkpoint")
    if pretrained_model and pretrained_model != "":
        p = Path(pretrained_model)
        if p.is_absolute() or p.exists():
            path_to_load = p if p.is_file() else resolve_pretrained_to_file(p)
        else:
            resolved = get_pretrained_path(
                pretrained_model,
                provider=c.get("model", {}).get("pretrained_provider"),
            )
            path_to_load = resolved if resolved.is_file() else resolve_pretrained_to_file(resolved)
        if path_to_load and path_to_load.is_file():
            model_params = load_model(path_to_load, device)
            logging.info(f"Load model: {path_to_load}")
            model.load_state_dict(model_params)
        else:
            logging.warning(
                "Pretrained could not be resolved to a weight file (use key from list_known_models or a .pt path): %s",
                pretrained_model,
            )

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

            # recovery_dir for full recovery from any run (e.g. --continue_from)
            recovery_dir_path = None
            if c.get("private", {}).get("recovery_dir"):
                recovery_dir_path = Path(c["private"]["recovery_dir"])

            # 创建标准训练器
            handler = Trainer(
                train_dir,
                model,
                is_continue_mode=is_continue_mode,
                recovery_dir=recovery_dir_path,
                async_executor=async_executor,
            )

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

            finished_epoch = 0
            if is_continue_mode and c["model"].get("continue_ext_checkpoint", ""):
                model_ext_path = Path(c["model"]["continue_ext_checkpoint"])
                if model_ext_path.exists():
                    model_ext_params = load_model_ext(model_ext_path, device)
                    finished_epoch = model_ext_params["epoch"]
                    if is_main_process():
                        logger = logging.getLogger("train")
                        logger.info(f"Continue Train from {finished_epoch}")
                    try:
                        optimizer.load_state_dict(model_ext_params["optimizer"])
                        if lr_scheduler and model_ext_params.get("lr_scheduler"):
                            lr_scheduler.load_state_dict(model_ext_params["lr_scheduler"])
                        if scaler and model_ext_params.get("scaler"):
                            scaler.load_state_dict(model_ext_params["scaler"])
                    except Exception as e:
                        logger = logging.getLogger("train")
                        logger.error(f"{e} WHILE LOADING EXT CHECKPOINT")
                elif finished_epoch == 0:
                    # No ext checkpoint: try recovery_info for last_epoch
                    recovery_dir = c.get("private", {}).get("recovery_dir") or (str(model_ext_path.parent.parent / "recovery") if model_ext_path.parent else None)
                    if recovery_dir:
                        pointer_file = Path(recovery_dir) / "recovery_info.json"
                        if pointer_file.exists():
                            try:
                                with pointer_file.open("r") as f:
                                    recovery_data = json.load(f)
                                finished_epoch = int(recovery_data.get("epoch", 0))
                                if is_main_process():
                                    logger = logging.getLogger("train")
                                    logger.info(f"Resumed last_epoch from recovery_info: {finished_epoch}")
                            except Exception as e:
                                if is_main_process():
                                    logging.getLogger("train").warning(f"Could not read epoch from recovery_info: {e}")

            handler.train(
                num_epochs=c["train"]["epoch"],
                train_dataloader=train_loader,
                valid_dataloader=valid_loader,
                early_stop="early_stopping" in c["train"],
                last_epoch=finished_epoch
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

                if is_continue_mode and c["model"].get("continue_ext_checkpoint", ""):
                    model_ext_path = Path(c["model"]["continue_ext_checkpoint"])
                    if model_ext_path.exists():
                        model_ext_params = load_model_ext(model_ext_path, device)
                        finished_epoch = model_ext_params["epoch"]
                        logger.info(f"Continue Train from {finished_epoch}")

                        try:
                            optimizer.load_state_dict(model_ext_params["optimizer"])
                            if lr_scheduler and model_ext_params.get("lr_scheduler"):
                                lr_scheduler.load_state_dict(model_ext_params["lr_scheduler"])
                            if scaler and model_ext_params.get("scaler"):
                                scaler.load_state_dict(model_ext_params["scaler"])
                        except Exception as e:
                            logger.error(f"{e} WHILE LOADING EXT CHECKPOINT")
                    elif finished_epoch == 0:
                        recovery_dir = c.get("private", {}).get("recovery_dir") or str(model_ext_path.parent.parent / "recovery")
                        pointer_file = Path(recovery_dir) / "recovery_info.json"
                        if pointer_file.exists():
                            try:
                                with pointer_file.open("r") as f:
                                    recovery_data = json.load(f)
                                finished_epoch = int(recovery_data.get("epoch", 0))
                                logger.info(f"Resumed last_epoch from recovery_info: {finished_epoch}")
                            except Exception as e:
                                logger.warning(f"Could not read epoch from recovery_info: {e}")

                recovery_dir_path = Path(c["private"]["recovery_dir"]) if c.get("private", {}).get("recovery_dir") else None
                handler = Trainer(
                    train_dir,
                    model,
                    is_continue_mode=is_continue_mode,
                    recovery_dir=recovery_dir_path,
                    async_executor=async_executor,
                )
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

        handler = Inferencer(test_dir, model)
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

        predict_runtime_conf = copy.deepcopy(c["predict"].get("config") or {})
        compare_conf = predict_runtime_conf.pop("compare", None)
        compare_enabled = bool(compare_conf and compare_conf.get("enabled", False))

        if compare_enabled:
            handler = MultiModelPredictor(predict_dir, compare_conf, base_config=c)
        else:
            handler = get_predictor(predict_dir, model, **predict_runtime_conf)

        input_path = Path(c["predict"]["input"])
        if input_path.is_dir():
            inputs = [filename for filename in input_path.iterdir()]
            handler.predict(inputs, **predict_runtime_conf)
        else:
            inputs = [input_path]
            handler.predict(inputs, **predict_runtime_conf)

    # get_input_size（缺失时跳过模型信息记录）
    input_sizes = c["model"]["config"].get("input_sizes")
    dtypes = c["model"]["config"].get("dtypes")
    if dtypes is not None and len(dtypes) > 0:
        dtypes = [str2dtype(dtype) for dtype in dtypes]
    else:
        dtypes = None

    # 清理 DDP 环境
    if use_ddp:
        cleanup_ddp()

    # 模型信息记录（只在主进程执行，且配置了 input_sizes 时）
    if (not use_ddp or is_main_process()) and input_sizes is not None:
        print_model_info_block(output_dir, model, input_sizes, dtypes=dtypes, device=c["device"])

    async_executor.shutdown(wait=True)


if __name__ == "__main__":
    parse_args()
    try:
        _run_main()
    except Exception as e:
        _wandb_alert_on_failure(e)
        raise
