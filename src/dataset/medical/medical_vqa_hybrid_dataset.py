from .path_vqa_dataset import PathVQADataset
from .vqa_rad_dataset import VQARadDataset
from ..hybrid_dataset import HybridDataset

from typing import Literal, Type, TypeAlias, Sequence
from pathlib import Path
from torch.utils.data import DataLoader

vqa_dataset: TypeAlias = Type[PathVQADataset] | Type[VQARadDataset]
HybridVQADataset = HybridDataset[vqa_dataset]


def create_hybrid_vqa_dataset(
    path_vqa_base_dir: Path,
    vqa_rad_base_dir: Path,
    split: str = 'train',
) -> HybridVQADataset:
    return HybridVQADataset([
        PathVQADataset(path_vqa_base_dir, split),
        VQARadDataset(vqa_rad_base_dir, split),
    ])

def create_hybrid_vqa_dataloader(
    path_vqa_base_dir: Path,
    vqa_rad_base_dir: Path,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    *,
    processor = None,
) -> DataLoader:
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
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "labels": labels
        } 
    
    dataset = create_hybrid_vqa_dataset(path_vqa_base_dir, vqa_rad_base_dir, split)
    return dataset.dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, processor)
    )