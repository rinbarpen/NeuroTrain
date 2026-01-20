import torch
from pathlib import Path
import numpy as np
import yaml
from typing import Literal, Optional, List, Dict, Tuple, Union
import json
from PIL import Image
import logging

from .custom_dataset import CustomDataset, Betweens

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logging.warning("pycocotools is not installed. COCO dataset will not be available.")


class COCODataset(CustomDataset):
    """COCO数据集实现
    
    支持COCO格式的目标检测和实例分割任务。
    
    数据集结构:
        root_dir/
            annotations/
                instances_train2017.json
                instances_val2017.json
            train2017/
                000000000001.jpg
                ...
            val2017/
                000000000001.jpg
                ...
    
    Args:
        root_dir: COCO数据集根目录
        split: 数据集划分 ('train', 'val', 'test')
        year: COCO数据集年份 (默认2017)
        task: 任务类型 ('detection', 'segmentation', 'keypoint', 'caption')
        use_crowd: 是否使用crowd标注
        min_keypoints: 对于keypoint任务，最小关键点数量
        return_masks: 是否返回分割mask
        **kwargs: 其他配置参数
    """
    
    mapping = {
        "train": ("train", "train"),
        "valid": ("val", "val"),
        "test": ("test", "test")
    }
    
    # COCO类别ID到类别名称的映射
    COCO_CATEGORIES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
        48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
        53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
        58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    
    def __init__(
        self, 
        root_dir: Union[str, Path], 
        split: Literal['train', 'val', 'test'],
        year: str = '2017',
        task: Literal['detection', 'segmentation', 'keypoint', 'caption'] = 'detection',
        use_crowd: bool = False,
        min_keypoints: int = 0,
        return_masks: bool = True,
        transform=None,
        **kwargs
    ):
        """
        初始化COCO数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分
            year: COCO数据集年份
            task: 任务类型
            use_crowd: 是否使用crowd标注
            min_keypoints: 最小关键点数量
            return_masks: 是否返回分割mask
            transform: 数据变换
            **kwargs: 其他配置参数
        """
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError(
                "pycocotools is required for COCO dataset. "
                "Install it with: uv add pycocotools"
            )
        
        super(COCODataset, self).__init__(root_dir, split, **kwargs)
        
        self.year = year
        self.task = task
        self.use_crowd = use_crowd
        self.min_keypoints = min_keypoints
        self.return_masks = return_masks
        self.transform = transform
        
        # 构建数据路径
        self.root_dir = Path(root_dir)
        
        # 根据split确定COCO的split名称
        if split == 'valid':
            coco_split = 'val'
        else:
            coco_split = split
            
        self.img_dir = self.root_dir / f'{coco_split}{year}'
        
        # 确定annotation文件名
        if task == 'detection' or task == 'segmentation':
            ann_file = self.root_dir / 'annotations' / f'instances_{coco_split}{year}.json'
        elif task == 'keypoint':
            ann_file = self.root_dir / 'annotations' / f'person_keypoints_{coco_split}{year}.json'
        elif task == 'caption':
            ann_file = self.root_dir / 'annotations' / f'captions_{coco_split}{year}.json'
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # 检查文件是否存在
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Please download COCO{year} dataset and place it in {root_dir}"
            )
        
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"Please download COCO{year} dataset and place it in {root_dir}"
            )
        
        # 初始化COCO API
        logging.info(f"Loading COCO annotations from {ann_file}")
        self.coco = COCO(str(ann_file))
        
        # 获取所有图像ID
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # 过滤没有标注的图像
        self.img_ids = [
            img_id for img_id in self.img_ids 
            if len(self._get_ann_ids(img_id)) > 0
        ]
        
        self.n = len(self.img_ids)
        
        logging.info(f"Loaded {self.n} images from COCO {split} set")
    
    def _get_ann_ids(self, img_id: int) -> List[int]:
        """获取图像的标注ID列表"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None if self.use_crowd else False)
        return ann_ids
    
    def __getitem__(self, index: int) -> Dict:
        """获取指定索引的数据样本"""
        # 获取图像ID和信息
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 加载图像
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 获取标注
        ann_ids = self._get_ann_ids(img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 根据任务类型处理标注
        if self.task == 'detection' or self.task == 'segmentation':
            target = self._process_detection_anns(anns, img_info)
        elif self.task == 'keypoint':
            target = self._process_keypoint_anns(anns, img_info)
        elif self.task == 'caption':
            target = self._process_caption_anns(anns)
        else:
            target = {}
        
        # 应用变换
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            # 默认转换为tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'target': target,
            'metadata': {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'split': self.split,
                'task': self.task
            }
        }
    
    def _process_detection_anns(self, anns: List[Dict], img_info: Dict) -> Dict:
        """处理目标检测/实例分割标注"""
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # 跳过crowd标注(如果不使用)
            if not self.use_crowd and ann.get('iscrowd', 0) == 1:
                continue
            
            # 获取bbox [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # 获取类别
            labels.append(ann['category_id'])
            
            # 获取面积
            areas.append(ann['area'])
            
            # 获取iscrowd标记
            iscrowd.append(ann.get('iscrowd', 0))
            
            # 如果需要mask且存在分割标注
            if self.return_masks and self.task == 'segmentation' and 'segmentation' in ann:
                # 将RLE或polygon转换为mask
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    mask = self.coco.annToMask(ann)
                else:
                    # RLE format
                    mask = coco_mask.decode(ann['segmentation'])
                masks.append(mask)
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        
        if masks:
            target['masks'] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        
        return target
    
    def _process_keypoint_anns(self, anns: List[Dict], img_info: Dict) -> Dict:
        """处理关键点检测标注"""
        boxes = []
        labels = []
        keypoints = []
        num_keypoints = []
        
        for ann in anns:
            # 跳过crowd标注
            if not self.use_crowd and ann.get('iscrowd', 0) == 1:
                continue
            
            # 跳过关键点数量不足的标注
            if ann.get('num_keypoints', 0) < self.min_keypoints:
                continue
            
            # 获取bbox
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # 获取类别
            labels.append(ann['category_id'])
            
            # 获取关键点 [x1, y1, v1, x2, y2, v2, ...]
            kpts = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints.append(kpts)
            
            # 获取关键点数量
            num_keypoints.append(ann.get('num_keypoints', 0))
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'keypoints': torch.as_tensor(np.stack(keypoints), dtype=torch.float32) if keypoints else torch.zeros((0, 17, 3), dtype=torch.float32),
            'num_keypoints': torch.as_tensor(num_keypoints, dtype=torch.int64) if num_keypoints else torch.zeros((0,), dtype=torch.int64),
        }
        
        return target
    
    def _process_caption_anns(self, anns: List[Dict]) -> Dict:
        """处理图像描述标注"""
        captions = [ann['caption'] for ann in anns]
        
        target = {
            'captions': captions,
            'caption_ids': [ann['id'] for ann in anns]
        }
        
        return target
    
    @staticmethod
    def name() -> str:
        """返回数据集名称"""
        return "COCO"
    
    @staticmethod
    def metadata(task: str = 'detection', **kwargs) -> Dict:
        """获取COCO数据集元数据
        
        Args:
            task: 任务类型 ('detection', 'segmentation', 'keypoint', 'caption')
            **kwargs: 其他参数
            
        Returns:
            包含数据集元数据的字典
        """
        base_meta = {
            'num_classes': 80,  # COCO有80个目标类别
            'num_categories': 91,  # 包含background的总类别数
            'dataset_name': 'COCO',
            'num_train': 118287,
            'num_val': 5000,
            'num_test': 40670,
        }
        
        if task == 'detection':
            return {
                **base_meta,
                'task_type': 'detection',
                'metrics': ['mAP', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large', 'AR'],
            }
        elif task == 'segmentation':
            return {
                **base_meta,
                'task_type': 'segmentation',
                'metrics': ['mAP', 'mAP50', 'mAP75', 'mAP_mask', 'AR'],
            }
        elif task == 'keypoint':
            return {
                **base_meta,
                'task_type': 'keypoint',
                'num_keypoints': 17,
                'metrics': ['mAP', 'mAP50', 'mAP75', 'AR'],
            }
        elif task == 'caption':
            return {
                **base_meta,
                'task_type': 'caption',
                'metrics': ['BLEU', 'METEOR', 'ROUGE', 'CIDEr'],
            }
        else:
            return {
                **base_meta,
                'task_type': task,
                'metrics': [],
            }
    
    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):
        """获取训练数据集"""
        return COCODataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):
        """获取验证数据集"""
        return COCODataset(root_dir, 'val', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):
        """获取测试数据集"""
        # COCO没有官方测试集标注，使用验证集
        logging.warning("COCO test set annotations are not publicly available, using validation set instead")
        return COCODataset(root_dir, 'val', **kwargs)
    
    def get_category_name(self, category_id: int) -> str:
        """根据类别ID获取类别名称"""
        return self.COCO_CATEGORIES.get(category_id, 'unknown')
    
    def get_img_info(self, index: int) -> Dict:
        """获取图像信息"""
        img_id = self.img_ids[index]
        return self.coco.loadImgs(img_id)[0]
    
    def get_ann_info(self, index: int) -> List[Dict]:
        """获取标注信息"""
        img_id = self.img_ids[index]
        ann_ids = self._get_ann_ids(img_id)
        return self.coco.loadAnns(ann_ids)


class COCOSegmentationDataset(CustomDataset):
    """COCO语义分割数据集实现
    
    专门用于语义分割任务，将COCO的实例分割标注转换为语义分割掩码。
    与实例分割不同，语义分割不区分同一类别的不同实例，所有相同类别的像素
    都被标记为相同的类别ID。
    
    数据集结构:
        root_dir/
            annotations/
                instances_train2017.json
                instances_val2017.json
            train2017/
                000000000001.jpg
                ...
            val2017/
                000000000001.jpg
                ...
    
    Args:
        root_dir: COCO数据集根目录
        split: 数据集划分 ('train', 'val', 'test', 'valid')
        year: COCO数据集年份 (默认2017)
        use_crowd: 是否使用crowd标注
        ignore_index: 忽略的标签索引（用于损失计算）
        bg_index: 背景类别索引
        semantic_mode: 语义分割模式，True返回mask，False返回target字典
        transform: 数据变换函数
        target_size: 目标图像大小 (height, width)，None表示保持原始大小
        **kwargs: 其他配置参数
    """
    
    mapping = {
        "train": ("train", "train"),
        "valid": ("val", "val"),
        "test": ("test", "test")
    }
    
    # COCO类别ID到连续类别ID的映射（0为背景）
    COCO_CATEGORY_TO_IDX = {
        0: 0,  # background
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
        11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19,
        21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29,
        34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39,
        44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49,
        55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59,
        65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69,
        79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
    }
    
    # COCO类别名称
    COCO_CATEGORIES = {
        0: 'background',
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
        15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
        20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
        25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
        30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
        35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard',
        39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup',
        43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana',
        48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot',
        53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair',
        58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table',
        62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote',
        67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven',
        71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock',
        76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'
    }
    
    def __init__(
        self, 
        root_dir: Union[str, Path], 
        split: Literal['train', 'val', 'test', 'valid'],
        year: str = '2017',
        use_crowd: bool = True,
        ignore_index: int = 255,
        bg_index: int = 0,
        semantic_mode: bool = True,
        transform=None,
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ):
        """初始化COCO语义分割数据集"""
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError(
                "pycocotools is required for COCO segmentation dataset. "
                "Install it with: uv add pycocotools"
            )
        
        super(COCOSegmentationDataset, self).__init__(root_dir, split, **kwargs)
        
        self.year = year
        self.use_crowd = use_crowd
        self.ignore_index = ignore_index
        self.bg_index = bg_index
        self.semantic_mode = semantic_mode
        self.transform = transform
        self.target_size = target_size
        
        # 构建数据路径
        self.root_dir = Path(root_dir)
        
        # 根据split确定COCO的split名称
        if split == 'valid':
            coco_split = 'val'
        else:
            coco_split = split
            
        self.img_dir = self.root_dir / f'{coco_split}{year}'
        ann_file = self.root_dir / 'annotations' / f'instances_{coco_split}{year}.json'
        
        # 检查文件是否存在
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Please download COCO{year} dataset and place it in {root_dir}"
            )
        
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"Please download COCO{year} dataset and place it in {root_dir}"
            )
        
        # 初始化COCO API
        logging.info(f"Loading COCO annotations from {ann_file}")
        self.coco = COCO(str(ann_file))
        
        # 获取所有图像ID，过滤没有分割标注的图像
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.img_ids = [img_id for img_id in self.img_ids if self._has_segmentation(img_id)]
        
        self.n = len(self.img_ids)
        logging.info(f"Loaded {self.n} images with segmentation from COCO {split} set")
    
    def _has_segmentation(self, img_id: int) -> bool:
        """检查图像是否有分割标注"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None if self.use_crowd else False)
        if not ann_ids:
            return False
        anns = self.coco.loadAnns(ann_ids)
        return any('segmentation' in ann for ann in anns)
    
    def _get_ann_ids(self, img_id: int) -> List[int]:
        """获取图像的标注ID列表"""
        return self.coco.getAnnIds(imgIds=img_id, iscrowd=None if self.use_crowd else False)
    
    def __getitem__(self, index: int) -> Dict:
        """获取指定索引的数据样本"""
        # 获取图像ID和信息
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # 加载图像
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 获取原始图像大小
        orig_height, orig_width = img_info['height'], img_info['width']
        
        # 获取标注并生成语义分割掩码
        ann_ids = self._get_ann_ids(img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 创建语义分割掩码
        seg_mask = self._create_semantic_mask(anns, orig_height, orig_width)
        
        # 调整大小
        if self.target_size is not None:
            image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            seg_mask = Image.fromarray(seg_mask).resize(
                (self.target_size[1], self.target_size[0]), 
                Image.NEAREST
            )
            seg_mask = np.array(seg_mask)
        
        # 应用变换
        if self.transform is not None:
            image, seg_mask = self.transform(image, seg_mask)
        else:
            # 默认转换为tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            seg_mask = torch.from_numpy(seg_mask).long()
        
        result = {
            'image': image,
            'metadata': {
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'split': self.split,
            }
        }
        
        # 根据模式返回不同的格式
        if self.semantic_mode:
            result['mask'] = seg_mask
        else:
            result['target'] = seg_mask
        
        return result
    
    def _create_semantic_mask(self, anns: List[Dict], height: int, width: int) -> np.ndarray:
        """创建语义分割掩码"""
        # 初始化为背景
        seg_mask = np.full((height, width), self.bg_index, dtype=np.uint8)
        
        # 按面积排序，先绘制大的物体，后绘制小的物体
        anns_sorted = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
        
        for ann in anns_sorted:
            # 检查是否有分割标注
            if 'segmentation' not in ann:
                continue
            
            # 跳过crowd标注（如果不使用）
            if not self.use_crowd and ann.get('iscrowd', 0) == 1:
                continue
            
            # 获取类别ID并映射到连续索引
            category_id = ann['category_id']
            class_idx = self.COCO_CATEGORY_TO_IDX.get(category_id, 0)
            
            # 生成mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                mask = self.coco.annToMask(ann)
            elif isinstance(ann['segmentation'], dict):
                # RLE format
                mask = coco_mask.decode(ann['segmentation'])
            else:
                continue
            
            # 将mask应用到语义分割掩码
            seg_mask[mask > 0] = class_idx
        
        return seg_mask
    
    @staticmethod
    def name() -> str:
        """返回数据集名称"""
        return "COCO_Segmentation"
    
    @staticmethod
    def metadata(**kwargs) -> Dict:
        """获取COCO语义分割数据集元数据"""
        return {
            'num_classes': 81,  # 80个目标类别 + 1个背景类别
            'num_object_classes': 80,  # 不包含背景的类别数
            'dataset_name': 'COCO_Segmentation',
            'task_type': 'semantic_segmentation',
            'num_train': 118287,
            'num_val': 5000,
            'ignore_index': 255,
            'metrics': ['mIoU', 'pixel_accuracy', 'mean_accuracy', 'dice'],
            'class_names': list(COCOSegmentationDataset.COCO_CATEGORIES.values()),
        }
    
    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):
        """获取训练数据集"""
        return COCOSegmentationDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):
        """获取验证数据集"""
        return COCOSegmentationDataset(root_dir, 'val', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):
        """获取测试数据集"""
        logging.warning("COCO test set annotations are not publicly available, using validation set instead")
        return COCOSegmentationDataset(root_dir, 'val', **kwargs)
    
    def get_category_name(self, class_idx: int) -> str:
        """根据类别索引获取类别名称"""
        return self.COCO_CATEGORIES.get(class_idx, 'unknown')

