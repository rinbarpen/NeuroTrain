from .custom_dataset import CustomDataset
from .isic2016_dataset import ISIC2016Dataset

from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import glob
from PIL import Image
import io

Split = Literal['train', 'valid', 'test']

# {task_type}/{split}_{index}-{hash_id}
class ISIC2016ReasoningSegDataset(CustomDataset):
    mapping = {
        'train': '{task_type}/train*.parquet',
        'valid': '{task_type}/valid*.parquet',
        'test': '{task_type}/test*.parquet',
    }
    def __init__(self, base_dir, split: Split, task_type: str="Task1", processor=None):
        super(ISIC2016ReasoningSegDataset, self).__init__(base_dir, split)
        data_path_pattern = Path(self.base_dir) / self.mapping[self.dataset_type].format(task_type=task_type)
        self.files = glob.glob(str(data_path_pattern))
        
        dfs = [ParquetFile(f).to_pandas() for f in self.files]
        self.data = pd.concat(dfs, ignore_index=True)

        self.processor = processor

        self.n = len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        image_bytes = item['image.bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        mask_bytes = item['mask.bytes']
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        question = item['question']
        answer = item['answer']

        return {
            'image': image,
            'mask': mask,
            'question': question,
            'answer': answer
        }

def collate_fn(batch, processor):
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
        input_ids = inputs.input_ids[i]
        
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
        'gt_masks': [item['mask'] for item in batch],
    }

def get_isic2016_reasoning_seg_dataloader(base_dir: Path, processor, batch_size: int, split: Literal['train', 'test', 'valid']='train', num_workers: int = 4):
    dataset = ISIC2016ReasoningSegDataset(base_dir, split)
    dataloader = dataset.dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: collate_fn(x, processor))
    return dataloader