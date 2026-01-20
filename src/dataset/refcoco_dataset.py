import json
import torch
from pathlib import Path
from typing import Literal, Optional, Dict, List, Union
from PIL import Image
import numpy as np
import logging
import glob
import pandas as pd
from fastparquet import ParquetFile

from .custom_dataset import CustomDataset

logger = logging.getLogger(__name__)


class RefCOCODataset(CustomDataset):
    mapping = {
        'train': 'val',
        'valid': 'test',
        'val': 'test',
        'test': 'test'
    }
    """RefCOCO数据集"""
    def __init__(self, root_dir: Union[str, Path], split: Literal['train', 'val', 'test', 'valid'], **kwargs):
        super(RefCOCODataset, self).__init__(root_dir, split, **kwargs)

        self.root_dir = Path(root_dir)
        self.split = self.mapping[split]

        self.samples = self._load_samples()
        self.n = len(self.samples)

    def _load_samples(self):
        """加载样本"""
        pf = ParquetFile(self.root_dir / 'data' / f'{self.split}-*.parquet')
        df = pf.to_pandas()
        return df
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

# class BaseRefCOCODataset(CustomDataset):
#     """RefCOCO系列数据集基类
    
#     RefCOCO系列数据集用于指代表达理解（Referring Expression Comprehension）任务，
#     给定图像和文本描述，需要定位对应的目标区域。
    
#     数据集结构:
#         root_dir/
#             images/
#                 mscoco/
#                     train2014/
#                     val2014/
#             annotations/
#                 refs(unc).p      # 引用表达标注 (pickle格式)
#                 instances.json   # 实例标注 (JSON格式)
#     或
#         root_dir/
#             images/
#                 train2014/
#                 val2014/
#             annotations/
#                 refs(unc).p
#                 instances.json
    
#     Args:
#         root_dir: 数据集根目录
#         split: 数据集划分 ('train', 'val', 'test')
#         dataset_name: 数据集名称 ('refcoco', 'refcoco+', 'refcocog')
#         image_dir: 图像目录名，默认为'images'，子目录为'mscoco'或直接在images下
#         transform: 图像变换
#         **kwargs: 其他参数
#     """
    
#     mapping = {
#         'train': 'train',
#         'valid': 'val',
#         'val': 'val',
#         'test': 'test'
#     }
    
#     def __init__(
#         self,
#         root_dir: Path,
#         split: Literal['train', 'val', 'test', 'valid'],
#         dataset_name: str = 'refcoco',
#         image_dir: str = 'images',
#         transform=None,
#         **kwargs
#     ):
#         super(BaseRefCOCODataset, self).__init__(root_dir, split, **kwargs)
        
#         self.dataset_name = dataset_name.lower()
#         self.transform = transform
        
#         # 标准化split名称
#         if split == 'valid':
#             split = 'val'
        
#         # 构建路径
#         self.root_dir = Path(root_dir)
        
#         # 查找图像目录（支持多种结构）
#         img_base = self.root_dir / image_dir
        
#         # 检查是否是COCO2014目录结构（如 /media/rczx/Dataset/coco2014）
#         if (self.root_dir / 'images' / 'train2014').exists():
#             # 结构: root_dir/images/train2014/ (如 coco2014/images/train2014/)
#             self.img_dir = self.root_dir / 'images'
#         elif (img_base / 'mscoco').exists():
#             # 结构: images/mscoco/train2014/
#             self.img_dir = img_base / 'mscoco'
#         elif (img_base / 'train2014').exists() or (img_base / 'val2014').exists():
#             # 结构: images/train2014/, images/val2014/ (直接包含COCO图像)
#             self.img_dir = img_base
#         else:
#             # 默认使用image_dir指定的目录
#             self.img_dir = img_base
        
#         logger.info(f"Using image directory: {self.img_dir}")
        
#         # 首先尝试从parquet文件加载（新的数据格式）
#         # 检查是否有parquet文件
#         parquet_data = None
#         annotations_dir = self.root_dir / 'annotations'  # 提前定义，供后续使用
#         if HAS_PANDAS:
#             # 检查独立的RefCOCO目录结构
#             possible_refcoco_dirs = []
#             parent_dir = self.root_dir.parent if self.root_dir.parent != self.root_dir else None
#             if parent_dir:
#                 # 根据dataset_name匹配对应的目录
#                 dir_mapping = {
#                     'refcoco': ['RefCOCO', 'refcoco'],
#                     'refcoco+': ['RefCOCOplus', 'RefCOCO+', 'refcoco+'],
#                     'refcocog': ['RefCOCOg', 'refcocog'],
#                 }
#                 dir_names = dir_mapping.get(self.dataset_name, ['RefCOCO', 'RefCOCOplus', 'RefCOCOg'])
#                 for dir_name in dir_names:
#                     refcoco_dir = parent_dir / dir_name
#                     if refcoco_dir.exists() and refcoco_dir.is_dir():
#                         possible_refcoco_dirs.append(refcoco_dir)
            
#             # 查找parquet文件
#             parquet_patterns = []
#             # 在当前root_dir下的data目录
#             data_dir = self.root_dir / 'data'
#             if data_dir.exists():
#                 parquet_patterns.extend([
#                     str(data_dir / f'{split}-*.parquet'),
#                     str(data_dir / f'{split}*.parquet'),
#                 ])
            
#             # 在独立RefCOCO目录的data目录
#             for refcoco_dir in possible_refcoco_dirs:
#                 refcoco_data_dir = refcoco_dir / 'data'
#                 if refcoco_data_dir.exists():
#                     parquet_patterns.extend([
#                         str(refcoco_data_dir / f'{split}-*.parquet'),
#                         str(refcoco_data_dir / f'{split}*.parquet'),
#                     ])
            
#             # 查找匹配的parquet文件
#             parquet_files = []
#             for pattern in parquet_patterns:
#                 parquet_files.extend(glob.glob(pattern))
#             parquet_files = sorted(set(parquet_files))  # 去重并排序
            
#             if parquet_files:
#                 # 加载所有parquet文件并合并
#                 dfs = []
#                 for pf in parquet_files:
#                     try:
#                         if HAS_PANDAS:
#                             df = pd.read_parquet(pf)
#                             dfs.append(df)
#                             logger.info(f"Loaded parquet file: {pf} ({len(df)} rows)")
#                     except Exception as e:
#                         logger.warning(f"Failed to load parquet file {pf}: {e}")
                
#                 if dfs and HAS_PANDAS:
#                     parquet_data = pd.concat(dfs, ignore_index=True)
#                     logger.info(f"Loaded {len(parquet_data)} samples from {len(parquet_files)} parquet files")
        
#         # 如果parquet加载成功，使用parquet数据
#         if parquet_data is not None and HAS_PANDAS and len(parquet_data) > 0:
#             self.refs = parquet_data.to_dict('records')
#             logger.info(f"Using parquet format: {len(self.refs)} samples")
#         else:
#             # 回退到原有的pickle/JSON格式加载
#             annotations_dir = self.root_dir / 'annotations'
            
#             # 可能的标注文件位置（包括独立的RefCOCO目录结构）
#             possible_refcoco_dirs = []
#             parent_dir = self.root_dir.parent if self.root_dir.parent != self.root_dir else None
#             if parent_dir:
#                 for dir_name in ['RefCOCO', 'RefCOCOplus', 'RefCOCOg', 'refcoco', 'refcoco+', 'refcocog']:
#                     refcoco_dir = parent_dir / dir_name
#                     if refcoco_dir.exists() and refcoco_dir.is_dir():
#                         possible_refcoco_dirs.append(refcoco_dir)
            
#             pickle_file = None
#             possible_pickle_files = [
#                 annotations_dir / 'refs(unc).p',
#                 annotations_dir / 'refs(google).p',
#                 annotations_dir / 'refs.p',
#                 annotations_dir / f'{self.dataset_name}_refs(unc).p',
#                 annotations_dir / f'{self.dataset_name}_refs(google).p',
#                 annotations_dir / f'{self.dataset_name}_refs.p',
#                 self.root_dir / 'refs(unc).p',
#                 self.root_dir / 'refs(google).p',
#                 self.root_dir / f'{self.dataset_name}_refs(unc).p',
#                 self.root_dir / f'{self.dataset_name}_refs(google).p',
#                 self.root_dir / 'refs.p',
#             ]
            
#             for refcoco_dir in possible_refcoco_dirs:
#                 possible_pickle_files.extend([
#                     refcoco_dir / 'annotations' / 'refs(unc).p',
#                     refcoco_dir / 'annotations' / 'refs(google).p',
#                     refcoco_dir / 'annotations' / 'refs.p',
#                     refcoco_dir / 'refs(unc).p',
#                     refcoco_dir / 'refs(google).p',
#                     refcoco_dir / 'refs.p',
#                 ])
            
#             # 根据数据集类型优先选择对应的文件
#             if self.dataset_name == 'refcoco':
#                 priority_files = [
#                     annotations_dir / 'refs(unc).p',
#                     annotations_dir / 'refs(google).p',
#                 ]
#             elif self.dataset_name == 'refcoco+':
#                 priority_files = [
#                     annotations_dir / 'refs(unc).p',
#                 ]
#             elif self.dataset_name == 'refcocog':
#                 priority_files = [
#                     annotations_dir / 'refs(google).p',
#                 ]
#             else:
#                 priority_files = []
            
#             for pf in priority_files + possible_pickle_files:
#                 if pf.exists():
#                     pickle_file = pf
#                     break
        
#             json_file = None
#             possible_json_files = [
#                 annotations_dir / f'{self.dataset_name}.json',
#                 annotations_dir / 'refs.json',
#                 self.root_dir / f'{self.dataset_name}.json',
#                 self.root_dir / 'refs.json',
#             ]
#             for refcoco_dir in possible_refcoco_dirs:
#                 possible_json_files.extend([
#                     refcoco_dir / 'annotations' / f'{self.dataset_name}.json',
#                     refcoco_dir / 'annotations' / 'refs.json',
#                     refcoco_dir / f'{self.dataset_name}.json',
#                     refcoco_dir / 'refs.json',
#                 ])
            
#             for jf in possible_json_files:
#                 if jf.exists():
#                     json_file = jf
#                     break
        
#             # 加载标注
#             if pickle_file is not None and pickle_file.exists():
#                 import pickle
#                 with open(pickle_file, 'rb') as f:
#                     self.refs = pickle.load(f)
#                 logger.info(f"Loaded {len(self.refs)} references from {pickle_file}")
#             elif json_file is not None and json_file.exists():
#                 with open(json_file, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     if 'refs' in data:
#                         self.refs = data['refs']
#                     elif isinstance(data, list):
#                         self.refs = data
#                     else:
#                         raise ValueError(f"Unknown JSON format in {json_file}")
#                 logger.info(f"Loaded {len(self.refs)} references from {json_file}")
#             else:
#                 tried_paths = []
#                 tried_paths.extend([str(pf) for pf in priority_files + possible_pickle_files])
#                 tried_paths.extend([str(jf) for jf in possible_json_files])
                
#                 error_msg = (
#                     f"RefCOCO annotation file not found for dataset '{self.dataset_name}'.\n"
#                     f"Tried parquet files and {len(tried_paths)} other possible locations:\n"
#                 )
#                 for i, path in enumerate(tried_paths[:20], 1):
#                     error_msg += f"  {i}. {path}\n"
#                 if len(tried_paths) > 20:
#                     error_msg += f"  ... and {len(tried_paths) - 20} more locations\n"
#                 error_msg += (
#                     f"\nPlease ensure RefCOCO dataset is properly downloaded.\n"
#                     f"Expected formats:\n"
#                     f"  - Parquet files in data/ directory (e.g., train-*.parquet, val-*.parquet)\n"
#                     f"  - Pickle files: refs(unc).p or refs(google).p\n"
#                     f"  - JSON files: refs.json or {self.dataset_name}.json\n"
#                     f"\nRoot directory: {self.root_dir}"
#                 )
#                 raise FileNotFoundError(error_msg)
        
#         # 加载实例标注（用于获取bbox等信息）
#         # 支持多种可能的文件位置和命名
#         instances_file = None
#         possible_instances_files = [
#             annotations_dir / 'instances.json',
#             annotations_dir / 'instances_train2014.json',
#             annotations_dir / 'instances_val2014.json',
#             annotations_dir / 'instances_test2014.json',
#         ]
        
#         # 根据split尝试加载对应的instances文件
#         if split == 'train':
#             train_file = annotations_dir / 'instances_train2014.json'
#             if train_file.exists():
#                 instances_file = train_file
#         elif split == 'val':
#             val_file = annotations_dir / 'instances_val2014.json'
#             if val_file.exists():
#                 instances_file = val_file
#         elif split == 'test':
#             test_file = annotations_dir / 'instances_test2014.json'
#             if test_file.exists():
#                 instances_file = test_file
        
#         # 如果按split找不到，尝试通用文件
#         if instances_file is None or not instances_file.exists():
#             for pf in possible_instances_files:
#                 if pf.exists():
#                     instances_file = pf
#                     break
        
#         if instances_file and instances_file.exists():
#             with open(instances_file, 'r', encoding='utf-8') as f:
#                 instances_data = json.load(f)
#                 # 构建image_id到image_info的映射
#                 self.images_dict = {img['id']: img for img in instances_data.get('images', [])}
#                 # 构建ann_id到annotation的映射
#                 self.annotations_dict = {ann['id']: ann for ann in instances_data.get('annotations', [])}
#             logger.info(f"Loaded instances from {instances_file}: {len(self.images_dict)} images, {len(self.annotations_dict)} annotations")
#         else:
#             logger.warning(f"Instances file not found. Tried: {possible_instances_files}, bbox information may be incomplete")
#             self.images_dict = {}
#             self.annotations_dict = {}
        
#         # 过滤当前split的refs（对于parquet格式，通常已经按split分好了，但还是要过滤一下）
#         self.samples = self._filter_refs_by_split(self.refs, split)
#         self.n = len(self.samples)
        
#         logger.info(f"Loaded {self.n} samples from {self.dataset_name} {split} set")
    
#     def _filter_refs_by_split(self, refs: List[Dict], split: str) -> List[Dict]:
#         """根据split过滤refs"""
#         filtered = []
        
#         # 处理parquet格式的数据，可能已经是DataFrame的records格式
#         for ref in refs:
#             # refs通常包含split信息
#             ref_split = ref.get('split', '').lower() if isinstance(ref.get('split'), str) else ''
            
#             # 对于parquet格式，split可能已经在文件名中，或者数据本身就是对应split的
#             # 如果split字段为空或不存在，根据split参数决定是否包含
#             if ref_split:
#                 if split == 'train' and ref_split in ['train', 'training']:
#                     filtered.append(ref)
#                 elif split == 'val' and ref_split in ['val', 'validation', 'valid']:
#                     filtered.append(ref)
#                 elif split == 'test' and ref_split in ['test', 'testing', 'testa', 'testb']:
#                     filtered.append(ref)
#             else:
#                 # 如果没有split字段，假设所有数据都属于当前split（parquet文件已经按split分组）
#                 # 这种情况下，如果refs来自对应的parquet文件，就全部接受
#                 filtered.append(ref)
        
#         return filtered
    
#     def _get_image_path(self, image_id: int, file_name: Optional[str] = None) -> Path:
#         """获取图像路径"""
#         # 首先尝试从images_dict获取文件名
#         if image_id in self.images_dict:
#             file_name = self.images_dict[image_id].get('file_name', file_name)
        
#         if file_name is None:
#             file_name = f"{image_id:012d}.jpg"
        
#         # 尝试不同的可能路径
#         possible_dirs = [
#             self.img_dir / 'train2014',
#             self.img_dir / 'val2014',
#             self.img_dir,
#             self.root_dir / 'train2014',
#             self.root_dir / 'val2014',
#         ]
        
#         for img_dir in possible_dirs:
#             img_path = img_dir / file_name
#             if img_path.exists():
#                 return img_path
        
#         # 如果都找不到，返回第一个可能路径（让调用者处理错误）
#         return possible_dirs[0] / file_name
    
#     def _get_bbox(self, ref: Dict) -> Optional[List[float]]:
#         """从ref中获取边界框"""
#         # 处理parquet格式（可能是Series）
#         if HAS_PANDAS:
#             try:
#                 import pandas as pd
#                 if isinstance(ref, pd.Series):
#                     ref = ref.to_dict()
#             except:
#                 pass
        
#         # ref可能直接包含bbox
#         bbox = ref.get('bbox') or ref.get('bounding_box') or ref.get('box')
#         if bbox is not None:
#             # 处理不同格式的bbox
#             if isinstance(bbox, list):
#                 if len(bbox) == 4:
#                     # 可能是 [x, y, w, h] 或 [x1, y1, x2, y2]
#                     if bbox[2] < bbox[0]:  # w < x，说明是 [x, y, w, h] 格式
#                         x, y, w, h = bbox
#                         return [x, y, x + w, y + h]
#                     else:  # 已经是 [x1, y1, x2, y2] 格式
#                         return bbox
#             elif isinstance(bbox, (tuple, np.ndarray)):
#                 bbox = list(bbox)
#                 if len(bbox) == 4:
#                     if bbox[2] < bbox[0]:
#                         x, y, w, h = bbox
#                         return [x, y, x + w, y + h]
#                     else:
#                 return bbox
        
#         # 或者通过ann_id获取
#         ann_id = ref.get('ann_id') or ref.get('annotation_id')
#         if ann_id and ann_id in self.annotations_dict:
#             ann = self.annotations_dict[ann_id]
#             if 'bbox' in ann:
#                 bbox = ann['bbox']
#                 if isinstance(bbox, list) and len(bbox) == 4:
#                     x, y, w, h = bbox
#                     return [x, y, x + w, y + h]
        
#         return None
    
#     def __getitem__(self, index: int) -> Dict:
#         """获取指定索引的数据样本"""
#         ref = self.samples[index]
        
#         # 处理parquet格式的数据（可能是字典或Series）
#         if HAS_PANDAS:
#             try:
#                 import pandas as pd
#                 if isinstance(ref, pd.Series):
#                     ref = ref.to_dict()
#             except:
#                 pass
        
#         # 获取图像信息
#         image_id = ref.get('image_id')
#         if image_id is None:
#             # 有些数据集可能使用'image'字段
#             image_info = ref.get('image')
#             if isinstance(image_info, dict):
#                 image_id = image_info.get('id')
#             elif HAS_PANDAS:
#                 try:
#                     import pandas as pd
#                     if pd.api.types.is_dict_like(image_info):
#                         image_id = image_info.get('id')
#                 except:
#                     pass
        
#         if image_id is None:
#             raise ValueError(f"Could not find image_id in ref: {list(ref.keys()) if isinstance(ref, dict) else ref}")
        
#         # 获取图像
#         file_name = ref.get('file_name') or ref.get('filename') or ref.get('image_file')
#         img_path = self._get_image_path(image_id, file_name)
#         if not img_path.exists():
#             raise FileNotFoundError(f"Image not found: {img_path}")
        
#         image = Image.open(img_path).convert('RGB')
        
#         # 获取引用表达文本（支持多种字段名）
#         sentence = None
#         # 尝试不同的字段名
#         for field in ['sentence', 'text', 'caption', 'expression', 'ref', 'sent', 'raw']:
#             val = ref.get(field)
#             if val:
#                 if isinstance(val, list) and len(val) > 0:
#                     # 如果是列表，取第一个元素
#                     if isinstance(val[0], dict):
#                         sentence = val[0].get('raw') or val[0].get('text') or val[0].get('sentence')
#                     else:
#                         sentence = str(val[0])
#                 elif isinstance(val, str):
#                     sentence = val
#                 elif isinstance(val, dict):
#                     sentence = val.get('raw') or val.get('text') or val.get('sentence')
                
#                 if sentence:
#                     break
        
#         # 如果还是找不到，尝试sentences字段
#         if not sentence:
#             sentences = ref.get('sentences', [])
#             if isinstance(sentences, list) and len(sentences) > 0:
#                 if isinstance(sentences[0], dict):
#                     sentence = sentences[0].get('raw') or sentences[0].get('text')
#                 else:
#                     sentence = str(sentences[0])
        
#         if not sentence:
#             sentence = ''  # 如果实在找不到，使用空字符串
        
#         # 获取边界框
#         bbox = self._get_bbox(ref)
        
#         # 应用变换
#         if self.transform is not None:
#             image = self.transform(image)
#         else:
#             # 默认转换为tensor
#             image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
#         result = {
#             'image': image,
#             'text': sentence,
#             'image_id': image_id,
#             'ref_id': ref.get('ref_id', ref.get('id', index)),
#             'metadata': {
#                 'image_id': image_id,
#                 'ref_id': ref.get('ref_id', ref.get('id', index)),
#                 'file_name': img_path.name,
#                 'split': self.split,
#                 'dataset': self.dataset_name,
#             }
#         }
        
#         if bbox is not None:
#             result['bbox'] = torch.tensor(bbox, dtype=torch.float32)
#             result['metadata']['bbox'] = bbox
        
#         return result
    
#     @staticmethod
#     def name() -> str:
#         """返回数据集名称"""
#         return "RefCOCO"
    
#     @staticmethod
#     def metadata(**kwargs) -> Dict:
#         """获取数据集元数据"""
#         return {
#             'task_type': 'referring_expression_comprehension',
#             'metrics': ['accuracy', 'iou', 'precision', 'recall'],
#             'description': 'Referring Expression Comprehension dataset',
#         }
    
#     @staticmethod
#     def get_train_dataset(root_dir: Path, **kwargs):
#         """获取训练数据集"""
#         return BaseRefCOCODataset(root_dir, 'train', **kwargs)
    
#     @staticmethod
#     def get_valid_dataset(root_dir: Path, **kwargs):
#         """获取验证数据集"""
#         return BaseRefCOCODataset(root_dir, 'val', **kwargs)
    
#     @staticmethod
#     def get_test_dataset(root_dir: Path, **kwargs):
#         """获取测试数据集"""
#         return BaseRefCOCODataset(root_dir, 'test', **kwargs)


# class RefCOCODataset(BaseRefCOCODataset):
#     """RefCOCO数据集
    
#     RefCOCO是第一个大规模的指代表达理解数据集，基于COCO数据集构建。
#     包含19,994幅图像，142,209个引用表达，50,000个目标。
#     """
    
#     def __init__(self, root_dir: Path, split: Literal['train', 'val', 'test', 'valid'], **kwargs):
#         super(RefCOCODataset, self).__init__(
#             root_dir, split, dataset_name='refcoco', **kwargs
#         )
    
#     @staticmethod
#     def name() -> str:
#         return "RefCOCO"
    
#     @staticmethod
#     def metadata(**kwargs) -> Dict:
#         base_meta = BaseRefCOCODataset.metadata(**kwargs)
#         base_meta.update({
#             'num_images': 19994,
#             'num_refs': 142209,
#             'num_objects': 50000,
#         })
#         return base_meta


# class RefCOCOPlusDataset(BaseRefCOCODataset):
#     """RefCOCO+数据集
    
#     RefCOCO+是RefCOCO的扩展版本，主要区别是禁止使用位置词汇（如left, right等）。
#     包含19,992幅图像，141,564个引用表达，49,856个目标。
#     """
    
#     def __init__(self, root_dir: Path, split: Literal['train', 'val', 'test', 'valid'], **kwargs):
#         super(RefCOCOPlusDataset, self).__init__(
#             root_dir, split, dataset_name='refcoco+', **kwargs
#         )
    
#     @staticmethod
#     def name() -> str:
#         return "RefCOCO+"
    
#     @staticmethod
#     def metadata(**kwargs) -> Dict:
#         base_meta = BaseRefCOCODataset.metadata(**kwargs)
#         base_meta.update({
#             'num_images': 19992,
#             'num_refs': 141564,
#             'num_objects': 49856,
#         })
#         return base_meta


# class RefCOCOgDataset(BaseRefCOCODataset):
#     """RefCOCOg数据集
    
#     RefCOCOg使用更长的描述性表达，通常比RefCOCO和RefCOCO+更长。
#     包含25,799幅图像，95,010个引用表达，49,856个目标。
#     """
    
#     def __init__(self, root_dir: Path, split: Literal['train', 'val', 'test', 'valid'], **kwargs):
#         super(RefCOCOgDataset, self).__init__(
#             root_dir, split, dataset_name='refcocog', **kwargs
#         )
    
#     @staticmethod
#     def name() -> str:
#         return "RefCOCOg"
    
#     @staticmethod
#     def metadata(**kwargs) -> Dict:
#         base_meta = BaseRefCOCODataset.metadata(**kwargs)
#         base_meta.update({
#             'num_images': 25799,
#             'num_refs': 95010,
#             'num_objects': 49856,
#         })
#         return base_meta

