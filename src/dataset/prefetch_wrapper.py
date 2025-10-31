"""
数据集预读取包装器

使用单独的线程提前加载数据，减少训练等待时间。
"""

import threading
import queue
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class PrefetchDataset(Dataset):
    """
    数据集预读取包装器
    
    使用单独的线程提前加载数据，可以显著提升训练性能。
    当模型在处理当前batch时，预读取线程已经在加载下一个batch的数据。
    
    特性:
    - 单独的后台线程进行数据预读
    - 可配置的缓冲队列大小
    - 线程安全
    - 自动资源管理
    """
    
    def __init__(
        self,
        dataset: Dataset,
        buffer_size: int = 2,
        enable_prefetch: bool = True
    ):
        """
        初始化预读取数据集
        
        Args:
            dataset: 原始数据集
            buffer_size: 预读取缓冲区大小（提前加载多少个样本）
            enable_prefetch: 是否启用预读取
        """
        self.dataset = dataset
        self.buffer_size = max(1, buffer_size)
        self.enable_prefetch = enable_prefetch
        
        # 预读取相关
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_event = None
        self._current_index = 0
        self._lock = threading.Lock()
        
        if self.enable_prefetch:
            self._init_prefetch()
            logger.info(f"预读取已启用，缓冲区大小: {self.buffer_size}")
        else:
            logger.info("预读取已禁用")
    
    def _init_prefetch(self):
        """初始化预读取线程和队列"""
        self._prefetch_queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="DatasetPrefetchThread"
        )
        self._prefetch_thread.start()
    
    def _prefetch_worker(self):
        """预读取工作线程"""
        if self._stop_event is None or self._prefetch_queue is None:
            return
            
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    index = self._current_index
                
                # 检查索引是否有效
                try:
                    dataset_len = len(self.dataset)  # type: ignore
                except Exception:
                    dataset_len = 0
                    
                if index >= dataset_len:
                    # 等待一段时间后重试
                    self._stop_event.wait(0.1)
                    continue
                
                try:
                    # 从原始数据集加载数据
                    data = self.dataset[index]
                    
                    # 将数据放入队列（阻塞直到有空间）
                    if not self._stop_event.is_set():
                        self._prefetch_queue.put((index, data), timeout=1.0)
                    
                    # 移动到下一个索引
                    with self._lock:
                        self._current_index += 1
                
                except queue.Full:
                    # 队列满了，等待一段时间
                    self._stop_event.wait(0.1)
                except Exception as e:
                    logger.error(f"预读取线程出错: {e}")
                    self._stop_event.wait(0.1)
        
        except Exception as e:
            logger.error(f"预读取线程异常退出: {e}")
    
    def __len__(self):
        """返回数据集长度"""
        try:
            return len(self.dataset)  # type: ignore
        except Exception:
            return 0
    
    def __getitem__(self, index: int):
        """
        获取数据项
        
        Args:
            index: 数据索引
            
        Returns:
            数据项
        """
        if not self.enable_prefetch or self._prefetch_queue is None:
            # 预读取禁用，直接从原始数据集获取
            return self.dataset[index]
        
        # 检查是否是顺序访问
        with self._lock:
            if index == self._current_index - 1:
                # 顺序访问，尝试从预读取队列获取
                try:
                    prefetch_index, prefetch_data = self._prefetch_queue.get(timeout=5.0)
                    
                    if prefetch_index == index:
                        # 预读取的数据匹配
                        return prefetch_data
                    else:
                        # 预读取的数据不匹配，放回队列并从原始数据集获取
                        self._prefetch_queue.put((prefetch_index, prefetch_data))
                        logger.debug(f"预读取索引不匹配: 期望 {index}, 实际 {prefetch_index}")
                        return self.dataset[index]
                
                except queue.Empty:
                    # 队列为空，直接从原始数据集获取
                    logger.debug(f"预读取队列为空，直接加载 index={index}")
                    return self.dataset[index]
            
            else:
                # 非顺序访问（如shuffle后的第一次访问），重置预读取
                logger.debug(f"检测到非顺序访问，重置预读取: {index}")
                self._reset_prefetch(index)
                return self.dataset[index]
    
    def _reset_prefetch(self, start_index: int):
        """
        重置预读取起始位置
        
        Args:
            start_index: 新的起始索引
        """
        if self._prefetch_queue is None:
            return
            
        with self._lock:
            # 清空队列
            while not self._prefetch_queue.empty():
                try:
                    self._prefetch_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 设置新的起始索引
            self._current_index = start_index + 1
    
    def stop_prefetch(self):
        """停止预读取线程"""
        if self.enable_prefetch and self._stop_event:
            self._stop_event.set()
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=2.0)
            logger.info("预读取线程已停止")
    
    def __del__(self):
        """析构函数，确保线程正确停止"""
        self.stop_prefetch()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_prefetch()


class SequentialPrefetchDataset(Dataset):
    """
    顺序预读取数据集
    
    针对顺序访问优化的预读取实现，性能更好。
    适用于训练时不使用shuffle的场景。
    """
    
    def __init__(
        self,
        dataset: Dataset,
        buffer_size: int = 4,
        enable_prefetch: bool = True
    ):
        """
        初始化顺序预读取数据集
        
        Args:
            dataset: 原始数据集
            buffer_size: 预读取缓冲区大小
            enable_prefetch: 是否启用预读取
        """
        self.dataset = dataset
        self.buffer_size = max(1, buffer_size)
        self.enable_prefetch = enable_prefetch
        
        # 预读取缓冲
        self._buffer = {}
        self._lock = threading.Lock()
        self._prefetch_thread = None
        self._stop_event = None
        self._next_index = 0
        
        if self.enable_prefetch:
            self._init_prefetch()
            logger.info(f"顺序预读取已启用，缓冲区大小: {self.buffer_size}")
    
    def _init_prefetch(self):
        """初始化预读取"""
        self._stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="SequentialPrefetchThread"
        )
        self._prefetch_thread.start()
    
    def _prefetch_worker(self):
        """预读取工作线程"""
        if self._stop_event is None:
            return
            
        try:
            while not self._stop_event.is_set():
                index = None
                with self._lock:
                    # 检查缓冲区是否已满
                    if len(self._buffer) >= self.buffer_size:
                        pass  # 缓冲区已满，等待
                    else:
                        try:
                            dataset_len = len(self.dataset)  # type: ignore
                        except Exception:
                            dataset_len = 0
                            
                        if self._next_index < dataset_len:
                            index = self._next_index
                            self._next_index += 1
                
                if index is not None:
                    try:
                        # 加载数据
                        data = self.dataset[index]
                        
                        # 存入缓冲
                        with self._lock:
                            self._buffer[index] = data
                    
                    except Exception as e:
                        logger.error(f"预读取数据失败 (index={index}): {e}")
                else:
                    # 没有更多数据要预读，等待
                    self._stop_event.wait(0.1)
        
        except Exception as e:
            logger.error(f"预读取线程异常: {e}")
    
    def __len__(self):
        """返回数据集长度"""
        try:
            return len(self.dataset)  # type: ignore
        except Exception:
            return 0
    
    def __getitem__(self, index: int):
        """
        获取数据项
        
        Args:
            index: 数据索引
            
        Returns:
            数据项
        """
        if not self.enable_prefetch:
            return self.dataset[index]
        
        # 尝试从缓冲获取
        with self._lock:
            if index in self._buffer:
                data = self._buffer.pop(index)
                return data
        
        # 缓冲中没有，直接加载
        logger.debug(f"缓冲未命中，直接加载 index={index}")
        return self.dataset[index]
    
    def stop_prefetch(self):
        """停止预读取"""
        if self.enable_prefetch and self._stop_event:
            self._stop_event.set()
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=2.0)
            logger.info("顺序预读取线程已停止")
    
    def __del__(self):
        """析构函数"""
        self.stop_prefetch()


def create_prefetch_dataset(
    dataset: Dataset,
    buffer_size: int = 2,
    enable_prefetch: bool = True,
    mode: str = "auto"
) -> Dataset:
    """
    创建预读取数据集的工厂函数
    
    Args:
        dataset: 原始数据集
        buffer_size: 缓冲区大小
        enable_prefetch: 是否启用预读取
        mode: 预读取模式
            - "auto": 自动选择（默认使用通用模式）
            - "sequential": 顺序预读取（适合不shuffle的场景）
            - "general": 通用预读取（适合shuffle的场景）
    
    Returns:
        包装后的数据集
    """
    if not enable_prefetch:
        return dataset
    
    if mode == "sequential":
        return SequentialPrefetchDataset(dataset, buffer_size, enable_prefetch)
    elif mode == "general":
        return PrefetchDataset(dataset, buffer_size, enable_prefetch)
    else:  # auto
        # 默认使用通用模式
        return PrefetchDataset(dataset, buffer_size, enable_prefetch)

