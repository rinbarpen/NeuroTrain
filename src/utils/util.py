import logging
import os.path as osp
import os
import sys
import io
import time
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.amp.grad_scaler import GradScaler
import torch.distributed as dist
import torchinfo
import psutil
from typing import Sequence, Type, Optional, Dict, Any, Union, List, Tuple, Callable, TypeVar
from pathlib import Path
from fvcore.nn import flop_count, flop_count_table, FlopCountAnalysis, parameter_count, parameter_count_table

from src.config import get_config, get_config_value
from src.utils.typed import FilePath, ImageInstance
from src.utils.criterion import CombineCriterion, get_criterion

def prepare_logger(output_dir: Path, names: Sequence[str]|None=None):
    c = get_config_value('private.log', default={
        'debug': False,
        'verbose': True,
        'log_file_format': '%Y-%m-%d %H_%M_%S',
        'log_format': '%(asctime)s %(levelname)s | %(name)s | %(message)s',
    })
    assert c is not None

    filename = output_dir / f'{time.strftime(c["log_file_format"], time.localtime())}.log'
    file_handler = logging.FileHandler(filename, encoding='utf-8', delay=True)
    file_handler.setFormatter(logging.Formatter(c['log_format']))
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter(c['log_format']))

    def get_log_level():
        if c['debug']:
            return logging.DEBUG
        elif c['verbose']:
            return logging.INFO
        else:
            return logging.WARNING

    logging.basicConfig(level=get_log_level(),
                        format=c['log_format'],
                        handlers=[file_handler, console_handler],
                        force=True)
    if names is not None:
        level = get_log_level()
        for name in names:
            logger = logging.getLogger(name)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(level)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_tools(model: nn.Module):
    c = get_config()
    train_c = c['train']
    optimizer_c = train_c['optimizer']
    match optimizer_c['type'].lower():
        case 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])
        case 'adam' | _:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=optimizer_c['learning_rate'], weight_decay=optimizer_c['weight_decay'])

    if 'lr_scheduler' not in train_c:
        scheduler = None
    else:
        scheduler_c = train_c['lr_scheduler']
        match scheduler_c['type'].lower():
            case 'step':
                step_size = scheduler_c['step_size']
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            case 'mstep':
                milestones = scheduler_c['step_size'] # list
                if isinstance(milestones, int):
                    milestones = [milestones]
                gamma = scheduler_c.get('gamma', 0.1)
                scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
            case 'cos' | 'cosine_annealing':
                T_max = scheduler_c.get('T_max', train_c['epoch'])  # 周期
                eta_min = scheduler_c.get('eta_min', 0)  # 最小学习率
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            case 'cos_warm':
                T_0 = scheduler_c.get('T_0', 10)  # 首次周期
                T_mult = scheduler_c.get('T_mult', 2)  # 周期倍数
                eta_min = scheduler_c.get('eta_min', 0.001)  # 最小学习率
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
            case _:
                scheduler = None

    return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'scaler': GradScaler() if 'scaler' in train_c else None,
    }

def get_train_criterion():
    c = get_config()
    return CombineCriterion(*[get_criterion(cc) for cc in c['criterion']]) if 'criterion' in c else None

def save_model(path: FilePath, model: nn.Module, *, 
               ext_path: FilePath|None=None,
               optimizer=None, lr_scheduler=None, scaler=None, **kwargs):
    if not is_main_process():
        return
    model_cp = model.state_dict()
    path = Path(path)

    try:
        torch.save(model_cp, path)
    except FileExistsError as e:
        path = path.parent / (path.stem +
                              time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        torch.save(model_cp, path)

    if ext_path:
        ext_path = Path(ext_path)
        ext_cp = dict()
        if optimizer:
            ext_cp["optimizer"] = optimizer.state_dict()
        if lr_scheduler:
            ext_cp["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler:
            ext_cp["scaler"] = scaler.state_dict()
        for k, v in kwargs.items():
            ext_cp[k] = v
        try:
            torch.save(ext_cp, ext_path)
        except FileExistsError as e:
            ext_path = ext_path.parent / (ext_path.stem +
                                time.strftime("%Y%m%d_%H%M%S", time.localtime()))
            torch.save(ext_cp, ext_path)
def load_model(path: FilePath, map_location: str = 'cuda'):
    return torch.load(path, map_location)
def load_model_ext(ext_path: FilePath, map_location: str = 'cuda'):
    return torch.load(ext_path, map_location)

def save_numpy_data(path: FilePath, data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(path, str):
        path = Path(path)

    try:
        np.save(path, data)
    except FileNotFoundError as e:
        path.parent.mkdir(parents=True)
        np.save(path, data)
def load_numpy_data(path: FilePath):
    try:
        data = np.load(path)
        return data
    except FileNotFoundError as e:
        logging.error(f'File is not found: {e}')
        raise e

def model_info(output_dir: Path, model: nn.Module, input_sizes: Sequence[int]|Sequence[Sequence[int]], dtypes: Type|Sequence[Type]|None=None, device: str = 'cuda', *, rich_print=True):
    # Check if input_sizes is a sequence of integers (not nested)
    if isinstance(input_sizes, (list, tuple)) and len(input_sizes) > 0 and isinstance(input_sizes[0], int):
        input_sizes = [input_sizes]
    if dtypes is None:
        pass
    elif isinstance(dtypes, Type):
        dtypes = [dtypes] * len(input_sizes)
    else:
        assert len(dtypes) == len(input_sizes), 'dtypes and input_sizes must have the same length'

    model_stats = torchinfo.summary(model, input_size=input_sizes, dtypes=dtypes, device=device, verbose=0)

    summary_file = output_dir / 'model_summary.txt'
    with summary_file.open('w', encoding='utf-8') as f:
        f.write(str(model_stats))
    
    if rich_print:
        from rich import print
        print(str(model_stats))

def model_flops(output_dir: Path, model: nn.Module, input_sizes: Sequence[int]|Sequence[Sequence[int]], dtypes: Type|Sequence[Type]|None=None, device: str='cuda', *, rich_print=True) -> float:
    # Check if input_sizes is a sequence of integers (not nested)
    if isinstance(input_sizes, (list, tuple)) and len(input_sizes) > 0 and isinstance(input_sizes[0], int):
        input_sizes = [input_sizes]
    if dtypes is None:
        pass
    elif isinstance(dtypes, Type):
        dtypes = [dtypes] * len(input_sizes)
    else:
        assert len(dtypes) == len(input_sizes), 'dtypes and input_sizes must have the same length'

    input_tensors = [torch.randn(input_size).to(device) for input_size in input_sizes]
    input_tensors = tuple(input_tensors)
    model = model.to(device)

    analysis = FlopCountAnalysis(model, input_tensors)
    table = flop_count_table(analysis)
    total_flops = analysis.total()

    flop_count_file = output_dir / 'model_flop_count.txt'
    with flop_count_file.open('w', encoding='utf-8') as f:
        f.write(table)

    if rich_print:
        from rich import print
        print(table)
    
    return total_flops

# freeze_filter = lambda n: ("clip" in n) or ("bert" in n)
# c = {"lr": [None, 0.01]}

# freeze_layers(model, freeze_filter, **c)
# layer_filter: 
#  layer_name: str [input]
#  result: bool [output]
def freeze_layers(model: nn.Module, freeze_filter=lambda n: True, **kwargs):
    named_params = model.named_parameters()
    for n, p in named_params:
        if freeze_filter(n):
            p.requires_grad = False

def str2dtype(dtype: str) -> torch.dtype:
    match dtype:
        case 'float16' | 'f16':
            ttype = torch.float16
        case 'float32' | 'f32':
            ttype = torch.float32
        case 'float64' | 'f64':
            ttype = torch.float64
        case 'bfloat16' | 'bf16':
            ttype = torch.bfloat16
        case 'uint8' | 'u8':
            ttype = torch.uint8
        case 'uint16' | 'u16':
            ttype = torch.uint16
        case 'uint32' | 'u32':
            ttype = torch.uint32
        case 'uint64' | 'u64':
            ttype = torch.uint64
        case 'int8' | 'i8':
            ttype = torch.int8
        case 'int16' | 'i16':
            ttype = torch.int16
        case 'int32' | 'i32':
            ttype = torch.int32
        case 'int64' | 'i64':
            ttype = torch.int64
        case _:
            ttype = torch.get_default_dtype()
    return ttype


def run_async_task(
    executor: ThreadPoolExecutor | None,
    func: Callable[..., Any] | None,
    *args,
    logger: logging.Logger | None = None,
    **kwargs,
):
    if func is None:
        return

    def _wrapper():
        try:
            func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            target_logger = logger or logging.getLogger(__name__)
            func_name = getattr(func, '__qualname__', getattr(func, '__name__', repr(func)))
            target_logger.warning(f"Async task {func_name} failed: {exc}")

    if executor is None:
        _wrapper()
    else:
        executor.submit(_wrapper)


def reset_peak_memory_stats():
    """Reset GPU peak memory statistics if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _collect_memory_cost() -> Dict[str, float]:
    stats: Dict[str, float] = {}
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        stats['cpu_used_mb'] = memory_info.rss / (1024 ** 2)
    except Exception:
        stats['cpu_used_mb'] = 0.0
    try:
        stats['cpu_percent'] = psutil.virtual_memory().percent
    except Exception:
        stats['cpu_percent'] = 0.0

    if torch.cuda.is_available():
        try:
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            stats['gpu_allocated_mb'] = stats['gpu_reserved_mb'] = stats['gpu_max_allocated_mb'] = 0.0
    return stats


def log_memory_cost(stage: str, logger: Optional[logging.Logger] = None):
    """Log CPU/GPU memory cost information for a specific stage."""
    stats = _collect_memory_cost()
    logger = logger or logging.getLogger(stage.lower())
    parts = [
        f"CPU Used: {stats.get('cpu_used_mb', 0.0):.1f} MB ({stats.get('cpu_percent', 0.0):.1f}%)",
    ]
    if torch.cuda.is_available():
        parts.append(
            f"GPU Allocated: {stats.get('gpu_allocated_mb', 0.0):.1f} MB / "
            f"Reserved: {stats.get('gpu_reserved_mb', 0.0):.1f} MB / "
            f"Max: {stats.get('gpu_max_allocated_mb', 0.0):.1f} MB"
        )
    logger.info(f"{stage} Memory Cost | " + " | ".join(parts))


def tensor_print(tensor: torch.Tensor, name: str=''):
    print(f'{name}: {tensor.shape} {tensor.dtype} {tensor.device} | {tensor}')
    print()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0

def all_gather(data, world_size: int|None=None, device: str='cuda'):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
        world_size: int, default None. If None, use the world size from distributed training.
        device: str, default 'cuda'. The device to use for the gather operation.
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = world_size or get_world_size()
    if world_size == 1:
        return [data]

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1":
        from functools import lru_cache
        @lru_cache()
        def _get_global_gloo_group():
            """
            Return a process group based on gloo backend, containing all the ranks
            The result is cached.
            """

            if dist.get_backend() == "nccl":
                return dist.new_group(backend="gloo")

            return dist.group.WORLD

        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    assert isinstance(local_size.item(), int)
    local_size = int(local_size.item())

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        tensor = torch.split(tensor, [size, max_size - size], dim=0)[0]
        buffer = io.BytesIO(tensor.cpu().numpy().tobytes())
        obj = torch.load(buffer)
        data_list.append(obj)

    return data_list

def reduce_dict(input_dict, world_size: int|None=None, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        world_size: int, default None. If None, use the world size from distributed training.
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    # check if world_size is greater than 1
    world_size = world_size or get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        rank = proc_id
        world_size = ntasks
        gpu = proc_id % num_gpus
        dist_url = 'env://'
    else:
        print('Not using distributed mode')
        distributed = False
        return

    distributed = True

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()
    setup_for_distributed(rank == 0)

    return {
        'distributed': distributed,
        'dist_backend': dist_backend,
        'dist_url': dist_url,
        'world_size': world_size,
        'rank': rank,
        'gpu': gpu,
    }
