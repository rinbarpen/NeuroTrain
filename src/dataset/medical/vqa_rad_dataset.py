import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import io
from fastparquet import ParquetFile
from pathlib import Path
from typing import Literal, Union

from ..custom_dataset import CustomDataset

class VQARadDataset(CustomDataset):
    mapping = {
        'train': 'data/train-00000-of-00001-eb8844602202be60.parquet',
        'test': 'data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'
    }
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'test']):
        super(VQARadDataset, self).__init__(root_dir, split)
        data_path = self.root_dir / self.mapping[split]
        pf = ParquetFile(data_path)
        df = pf.to_pandas()
        self.samples = df

        self.n = len(self.samples)
    
    def __getitem__(self, idx):
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
    """数据批处理函数"""
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

def get_vqarad_dataloader(root_dir: Union[str, Path], processor, batch_size: int, split: Literal['train', 'test']='train', num_workers: int = 4):
    dataset = VQARadDataset(root_dir, split)
    dataloader = dataset.dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, processor)
    )
    return dataloader
