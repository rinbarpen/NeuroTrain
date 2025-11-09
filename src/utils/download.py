"""
通用下载工具模块，支持断点续传

提供统一的下载接口，支持：
- 断点续传
- 下载进度显示
- 文件完整性校验
- 错误重试
"""
import os
import sys
import logging
import requests
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse
import hashlib
from tqdm import tqdm


logger = logging.getLogger(__name__)


def download_file(
    url: str,
    save_path: str | Path,
    chunk_size: int = 8192,
    resume: bool = True,
    verify_hash: bool = False,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = 'sha256',
    show_progress: bool = True,
    timeout: Optional[int] = None,
    max_retries: int = 3,
) -> Path:
    """
    下载文件，支持断点续传
    
    Args:
        url: 下载URL
        save_path: 保存路径（文件或目录）
        chunk_size: 下载块大小（字节）
        resume: 是否启用断点续传
        verify_hash: 是否验证文件哈希
        expected_hash: 期望的文件哈希值
        hash_algorithm: 哈希算法（'md5', 'sha1', 'sha256'）
        show_progress: 是否显示下载进度
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        
    Returns:
        下载文件的完整路径
        
    Raises:
        requests.RequestException: 下载失败
        ValueError: 文件哈希校验失败
    """
    save_path = Path(save_path)
    
    # 如果save_path是目录，从URL提取文件名
    if save_path.is_dir() or (not save_path.exists() and save_path.suffix == ''):
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'download'
        save_path = save_path / filename
    elif save_path.is_dir():
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'download'
        save_path = save_path / filename
    
    # 确保父目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否已存在且完整
    if save_path.exists():
        if verify_hash and expected_hash:
            if _verify_file_hash(save_path, expected_hash, hash_algorithm):
                logger.info(f"文件已存在且完整: {save_path}")
                return save_path
        else:
            # 如果没有哈希校验，尝试HEAD请求获取文件大小
            try:
                head_response = requests.head(url, timeout=timeout or 30, allow_redirects=True)
                if head_response.status_code == 200:
                    remote_size = int(head_response.headers.get('Content-Length', 0))
                    local_size = save_path.stat().st_size
                    if remote_size > 0 and local_size == remote_size:
                        logger.info(f"文件已存在且大小匹配: {save_path} ({local_size} bytes)")
                        return save_path
            except Exception:
                pass  # 如果HEAD请求失败，继续下载
    
    # 准备断点续传
    resume_pos = 0
    if resume and save_path.exists():
        resume_pos = save_path.stat().st_size
        if resume_pos > 0:
            logger.info(f"检测到未完成的下载，从 {resume_pos} 字节处继续: {save_path}")
    
    # 下载文件
    headers = {}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
    
    for attempt in range(max_retries):
        try:
            with requests.get(
                url,
                stream=True,
                headers=headers,
                timeout=timeout,
                allow_redirects=True
            ) as response:
                response.raise_for_status()
                
                # 检查服务器是否支持断点续传
                if resume_pos > 0:
                    if response.status_code == 206:  # Partial Content
                        logger.info("服务器支持断点续传")
                    elif response.status_code == 200:
                        logger.warning("服务器不支持断点续传，重新下载")
                        resume_pos = 0
                        save_path.unlink()  # 删除不完整的文件
                
                # 获取文件总大小
                total_size = int(response.headers.get('Content-Length', 0))
                if resume_pos > 0 and total_size > 0:
                    total_size += resume_pos
                
                # 打开文件进行写入
                mode = 'ab' if resume_pos > 0 else 'wb'
                with open(save_path, mode) as f:
                    if show_progress:
                        # 创建进度条
                        pbar = tqdm(
                            total=total_size,
                            initial=resume_pos,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"下载 {save_path.name}"
                        )
                    
                    # 下载数据块
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if show_progress:
                                pbar.update(len(chunk))
                    
                    if show_progress:
                        pbar.close()
                
                # 验证文件哈希（在下载完成后统一计算）
                if verify_hash and expected_hash:
                    file_hash = _calculate_file_hash(save_path, hash_algorithm)
                    
                    if file_hash.lower() != expected_hash.lower():
                        save_path.unlink()
                        raise ValueError(
                            f"文件哈希校验失败！\n"
                            f"期望: {expected_hash}\n"
                            f"实际: {file_hash}"
                        )
                    logger.info(f"文件哈希校验通过: {file_hash}")
                
                logger.info(f"下载完成: {save_path}")
                return save_path
                
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                continue
            else:
                logger.error(f"下载失败，已达到最大重试次数: {e}")
                raise
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            raise
    
    raise RuntimeError("下载失败")


def _calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """计算文件的哈希值"""
    hasher = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_file_hash(file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """验证文件哈希值"""
    try:
        actual_hash = _calculate_file_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()
    except Exception as e:
        logger.warning(f"计算文件哈希时出错: {e}")
        return False


def download_with_retry(
    url: str,
    save_path: str | Path,
    max_retries: int = 3,
    **kwargs
) -> Path:
    """
    带重试机制的下载函数
    
    Args:
        url: 下载URL
        save_path: 保存路径
        max_retries: 最大重试次数
        **kwargs: 传递给download_file的其他参数
        
    Returns:
        下载文件的完整路径
    """
    return download_file(url, save_path, max_retries=max_retries, **kwargs)


def download_multiple(
    urls: list[str | tuple[str, str | Path]],
    base_dir: str | Path,
    **kwargs
) -> list[Path]:
    """
    批量下载多个文件
    
    Args:
        urls: URL列表，可以是字符串或(url, save_path)元组
        base_dir: 基础保存目录
        **kwargs: 传递给download_file的其他参数
        
    Returns:
        下载文件的路径列表
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    for item in urls:
        if isinstance(item, tuple):
            url, save_path = item
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
        else:
            url = item
            save_path = base_dir
        
        try:
            file_path = download_file(url, save_path, **kwargs)
            downloaded_files.append(file_path)
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            raise
    
    return downloaded_files

