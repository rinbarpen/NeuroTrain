from pathlib import Path
from typing import Literal, Dict, List, Optional
from PIL import Image
import torch
import numpy as np
import logging
import pandas as pd

from .custom_dataset import CustomDataset

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools import mask as coco_mask  # type: ignore
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logging.warning("pycocotools is not installed. RLE mask decoding will be unavailable.")

logger = logging.getLogger(__name__)


class COCO2017Dataset(CustomDataset):
    """COCO2017数据集（基于Parquet的检测标注）
    
    基于由COCO JSON转换得到的 Parquet 文件加载检测任务数据。
    期望的 Parquet 文件为: `bbox_list_train.parquet`, `bbox_list_valid.parquet`, `bbox_list_test.parquet`，
    每行对应一张图像，包含列：
      - image_id: int
      - file_name: str
      - width: int
      - height: int
      - bboxes: List[List[float]]  每个元素为 [x, y, w, h]
      - labels: List[str]          类别名（可选）
      - category_ids: List[int]    COCO类别ID
      - num_objects: int
    
    图像仍从 `root_dir/coco2017/{train2017|val2017}` 读取。
    
    Args:
        root_dir: 数据集根目录（包含coco2017子目录，且优先在此处查找parquet）
        split: 数据集划分 ('train', 'val', 'test', 'valid')
        task: 任务类型（当前支持 'detection'）
        use_crowd: 兼容参数（无效，占位）
        min_keypoints: 兼容参数（无效，占位）
        return_masks: 兼容参数（无效，占位）
        transform: 数据变换函数
        parquet_path: 明确指定parquet文件路径（可选）
        **kwargs: 其他配置参数
    """
    mapping = {
        'train': 'train',
        'val': 'val',
        'valid': 'val',
        'test': 'val'  # COCO没有公开的test标注，使用val代替
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
        root_dir: Path,
        split: Literal['train', 'val', 'test', 'valid'],
        task: Literal['detection', 'segmentation', 'keypoint', 'caption'] = 'detection',
        use_crowd: bool = False,
        min_keypoints: int = 0,
        return_masks: bool = False,
        transform=None,
        parquet_path: Optional[Path] = None,
        **kwargs
    ):
        """初始化COCO2017数据集（基于Parquet）"""
        super(COCO2017Dataset, self).__init__(root_dir, split, **kwargs)

        if task != 'detection':
            logger.warning("COCO2017Dataset (Parquet) currently supports detection only. Other tasks are not implemented.")
        self.task = task
        self.use_crowd = use_crowd
        self.min_keypoints = min_keypoints
        self.return_masks = return_masks
        self.transform = transform

        # 路径
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / 'coco2017'

        # 图像目录 split 映射（用于选择 train2017/val2017）
        if split not in self.mapping:
            raise ValueError(f"Unsupported split: {split}. Must be one of {list(self.mapping.keys())}")
        coco_split = self.mapping[split]
        self.img_dir = self.data_dir / f'{coco_split}2017'
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"Please ensure COCO2017 images are in {self.img_dir}"
            )

        # Parquet 文件 split 映射（val -> valid）
        file_split_map = {
            'train': 'train',
            'val': 'valid',
            'valid': 'valid',
            'test': 'test',
        }
        file_split = file_split_map.get(split, split)
        default_parquet_name = f'bbox_list_{file_split}.parquet'

        # 查找 parquet 文件
        candidate_paths: List[Path] = []
        if parquet_path is not None:
            candidate_paths.append(Path(parquet_path))
        candidate_paths.extend([
            self.root_dir / default_parquet_name,
            self.root_dir.parent / default_parquet_name if self.root_dir.parent != self.root_dir else self.root_dir / default_parquet_name,
            Path.cwd() / default_parquet_name,
            self.data_dir / default_parquet_name,
            Path('data') / default_parquet_name,
        ])
        parquet_file: Optional[Path] = next((p for p in candidate_paths if p.exists()), None)
        if parquet_file is None:
            tried = "\n".join([f" - {p}" for p in candidate_paths])
            raise FileNotFoundError(
                f"Parquet file not found for split '{split}'. Tried:\n{tried}\n"
                f"Please generate it via notebook (bbox_list_{file_split}.parquet)."
            )

        logger.info(f"Loading COCO2017 (Parquet) from {parquet_file}")
        df = pd.read_parquet(parquet_file)

        # 保留必要列，并确保类型
        expected_cols = {'image_id', 'file_name', 'width', 'height', 'bboxes', 'category_ids'}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {sorted(missing)}")

        # 过滤图像存在的行
        def image_exists(row) -> bool:
            return (self.img_dir / row['file_name']).exists()

        try:
            df = df[df.apply(image_exists, axis=1)].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to filter by image existence: {e}")

        self.df = df
        self.img_ids = list(self.df['image_id'].astype(int).tolist())
        self.n = len(self.df)
        logger.info(f"Loaded {self.n} images from COCO2017 {split} set (Parquet)")
    
    def _get_ann_ids(self, img_id: int) -> List[int]:
        """获取图像的标注ID列表"""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None if self.use_crowd else False)
        return ann_ids
    
    def __len__(self):
        """返回数据集样本数量"""
        return self.n
    
    def __getitem__(self, index: int) -> Dict:
        """获取指定索引的数据样本（基于Parquet行）"""
        if index >= self.n:
            raise IndexError(f"Index {index} out of range for dataset of size {self.n}")

        row = self.df.iloc[index]
        img_id = int(row['image_id'])
        file_name: str = row['file_name']
        width = int(row['width'])
        height = int(row['height'])

        # 加载图像
        img_path = self.img_dir / file_name
        image = Image.open(img_path).convert('RGB')

        # 处理检测标注：bboxes为[x, y, w, h]，转为[x1, y1, x2, y2]
        bboxes: List[List[float]] = row['bboxes'] if isinstance(row['bboxes'], list) else []
        category_ids: List[int] = row['category_ids'] if isinstance(row['category_ids'], list) else []

        boxes_xyxy: List[List[float]] = []
        areas: List[float] = []
        for bb in bboxes:
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            boxes_xyxy.append([x, y, x + w, y + h])
            areas.append(w * h)

        labels_tensor = (
            torch.as_tensor(category_ids, dtype=torch.int64)
            if category_ids else torch.zeros((0,), dtype=torch.int64)
        )
        boxes_tensor = (
            torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            if boxes_xyxy else torch.zeros((0, 4), dtype=torch.float32)
        )
        areas_tensor = (
            torch.as_tensor(areas, dtype=torch.float32)
            if areas else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd_tensor = torch.zeros((labels_tensor.shape[0],), dtype=torch.int64)

        target: Dict = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'area': areas_tensor,
            'iscrowd': iscrowd_tensor,
        }

        # 变换
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return {
            'image': image,
            'target': target,
            'metadata': {
                'image_id': img_id,
                'file_name': file_name,
                'height': height,
                'width': width,
                'split': self.split,
                'task': self.task,
            },
        }
    
    # 旧的COCO JSON处理方法已移除；当前实现基于Parquet检测标注
    
    @staticmethod
    def name() -> str:
        """返回数据集名称"""
        return "COCO2017"
    
    @staticmethod
    def metadata(task: str = 'detection', **kwargs) -> Dict:
        """获取COCO2017数据集元数据"""
        base_meta = {
            'num_classes': 80,  # COCO有80个目标类别
            'num_categories': 91,  # 包含background的总类别数
            'dataset_name': 'COCO2017',
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
    def get_train_dataset(root_dir: Path, **kwargs):
        """获取训练数据集"""
        return COCO2017Dataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs):
        """获取验证数据集"""
        return COCO2017Dataset(root_dir, 'valid', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs):
        """获取测试数据集"""
        # COCO没有官方测试集标注，使用验证集
        logger.warning("COCO2017 test set annotations are not publicly available, using validation set instead")
        return COCO2017Dataset(root_dir, 'val', **kwargs)
    
    def get_category_name(self, category_id: int) -> str:
        """根据类别ID获取类别名称"""
        return self.COCO_CATEGORIES.get(category_id, 'unknown')
    
    def get_img_info(self, index: int) -> Dict:
        """获取图像信息（来自Parquet行）"""
        if index >= self.n:
            raise IndexError(f"Index {index} out of range for dataset of size {self.n}")
        row = self.df.iloc[index]
        return {
            'id': int(row['image_id']),
            'file_name': row['file_name'],
            'width': int(row['width']),
            'height': int(row['height']),
        }
    
    def get_ann_info(self, index: int) -> List[Dict]:
        """获取标注信息（检测）"""
        if index >= self.n:
            raise IndexError(f"Index {index} out of range for dataset of size {self.n}")
        row = self.df.iloc[index]
        bboxes: List[List[float]] = row['bboxes'] if isinstance(row['bboxes'], list) else []
        category_ids: List[int] = row['category_ids'] if isinstance(row['category_ids'], list) else []
        anns: List[Dict] = []
        for i, bb in enumerate(bboxes):
            cid = int(category_ids[i]) if i < len(category_ids) else -1
            anns.append({'bbox': bb, 'category_id': cid})
        return anns


if __name__ == '__main__':
    # 测试数据集
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 使用data目录作为根目录
    ds = COCO2017Dataset(root_dir=Path('data'), split='train', task='detection')
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else 'PIL Image'}")
        print(f"Target keys: {list(sample['target'].keys())}")
        print(f"Metadata: {sample['metadata']}")
