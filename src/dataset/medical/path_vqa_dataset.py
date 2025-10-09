import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
from fastparquet import ParquetFile
from pathlib import Path
from typing import Literal
import glob

from ..custom_dataset import CustomDataset

class PathVQADataset(CustomDataset):
    """
    PathVQA 数据集的 Dataset 类。
    """
    mapping = {
        'train': 'data/train-*.parquet',
        'test': 'data/test-*.parquet',
        'valid': 'data/valid-*.parquet'
    }
    def __init__(self, root_dir: Path, split: Literal['train', 'test', 'valid']):
        """
        初始化 PathVQADataset。

        Args:
            root_dir (Path): 数据集所在的根目录。
            split (Literal['train', 'test', 'valid']): 数据集划分（训练、测试或验证）。
        """
        super(PathVQADataset, self).__init__(root_dir, split)
        data_path_pattern = root_dir / self.mapping[split]
        data_files = glob.glob(str(data_path_pattern))
        
        dfs = [ParquetFile(f).to_pandas() for f in data_files]
        self.samples = pd.concat(dfs, ignore_index=True)

        self.n = len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取数据集中的单个样本。

        Args:
            idx (int): 样本索引。

        Returns:
            dict: 包含图像、问题和答案的字典。
        """
        item = self.samples.iloc[idx]
        
        # 加载图像
        image_bytes = item['image.bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        question = item['question']
        answer = item['answer']
        
        return {
            'image': image,
            'question': question,
            'answer': answer
        }

def collate_fn(batch, processor):
    """
    数据批处理函数。

    Args:
        batch (list): 一批数据样本。
        processor: 用于处理文本和图像的 processor。

    Returns:
        dict: 包含处理后的 input_ids, attention_mask, pixel_values 和 labels 的字典。
    """
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]

    texts = [f"USER: <image>\n{q}\nASSISTANT: {a}" for q, a in zip(questions, answers)]
    
    # 使用processor处理
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    labels = inputs["input_ids"].clone()
    
    # Masking user prompts
    response_template = "ASSISTANT:"
    
    for i in range(len(texts)):
        # Find where the assistant's response begins
        input_ids = inputs['input_ids'][i]
        
        # Find the tokenized representation of the response template
        template_ids = processor.tokenizer(response_template, add_special_tokens=False).input_ids
        
        # Search for the template in the input_ids
        start_idx = -1
        # Simple search for sublist
        for k in range(len(input_ids) - len(template_ids) + 1):
            if input_ids[k:k+len(template_ids)].tolist() == template_ids:
                start_idx = k + len(template_ids)
                break
        
        if start_idx != -1:
            labels[i, :start_idx] = -100
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': inputs['pixel_values'],
        'labels': labels,
    }

def get_pathvqa_dataloader(root_dir: Path, processor, batch_size: int, split: Literal['train', 'test', 'valid']='train', num_workers: int = 4):
    """
    获取 PathVQA 数据集的 Dataloader。

    Args:
        root_dir (Path): 数据集所在的根目录。
        processor: 用于处理文本和图像的 processor。
        batch_size (int): 批处理大小。
        split (Literal['train', 'test', 'valid'], optional): 数据集划分。默认为 'train'。
        num_workers (int, optional): 用于数据加载的进程数。默认为 4。

    Returns:
        DataLoader: PathVQA 数据集的 Dataloader。
    """
    dataset = PathVQADataset(root_dir, split, processor)
    dataloader = dataset.dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: collate_fn(x, processor))
    return dataloader