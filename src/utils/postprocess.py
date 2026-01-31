import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Any
from PIL import Image

def postprocess_binary_classification(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对二分类输出进行后处理
    # 假设outputs为[N, C]，C为类别数，targets同shape
    # 返回二值化后的targets和outputs
    targets = targets.long()
    # 对输出进行sigmoid激活，然后二值化
    outputs = (F.sigmoid(outputs) >= 0.5).long()
    return targets, outputs

def postprocess_multiclass_classification(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对多分类输出进行后处理
    # 假设outputs为[N, C]，C为类别数，targets为[N]或[N, 1]的类别索引
    # 返回处理后的targets和outputs
    targets = targets.long()
    if targets.dim() > 1 and targets.size(1) > 1:
        # 如果targets是one-hot编码，转换为类别索引
        targets = torch.argmax(targets, dim=1)
    # 对输出进行softmax，然后取最大值的索引
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    return targets, outputs

def postprocess_regression(targets: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 对回归输出进行后处理
    # 回归任务通常不需要特殊处理，直接返回原始值
    # 但可以确保数据类型一致
    return targets.float(), outputs.float()

def postprocess_binary_segmentation(targets: torch.Tensor, outputs: torch.Tensor, is_multitask=False) -> tuple[torch.Tensor, torch.Tensor]:
    if is_multitask:
        targets = targets[:, 1:, ...]
        outputs = outputs[:, 1:, ...]
    targets = targets.bool()
    outputs = F.sigmoid(outputs) >= 0.5
    return targets, outputs

def postprocess_instance_segmentation(targets: torch.Tensor, outputs: torch.Tensor, is_multitask=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对实例分割输出进行后处理。
    假设outputs为[N, C, H, W]，C为类别数，targets同shape。
    返回二值化后的targets和outputs。
    """
    if is_multitask:
        outputs = outputs[:, 1:, ...]
        targets = targets[:, 1:, ...]
    # softmax后取最大类别
    outputs = torch.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1, keepdim=True)
    targets = torch.argmax(targets, dim=1, keepdim=True)
    return targets, outputs

def postprocess_panoptic_segmentation(
    semantic_logits: torch.Tensor, 
    center_heatmap: torch.Tensor, 
    offset_map: torch.Tensor,
    center_threshold: float = 0.1,
    nms_kernel: int = 7,
    top_k: int = 200,
    stuff_area: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对Panoptic-DeepLab输出进行后处理，生成全景分割结果。
    
    Args:
        semantic_logits: 语义分割logits，形状 (B, n_classes+1, H, W)
        center_heatmap: 中心点热力图，形状 (B, 1, H, W)，值域 [0, 1]
        offset_map: 偏移图，形状 (B, 2, H, W)
        center_threshold: 中心点检测阈值
        nms_kernel: 非极大值抑制的核大小
        top_k: 保留的最大实例数量
        stuff_area: stuff类别的最小面积阈值
        
    Returns:
        tuple: (semantic_seg, panoptic_seg)
            - semantic_seg: 语义分割结果，形状 (B, H, W)
            - panoptic_seg: 全景分割结果，形状 (B, H, W)
    """
    device = semantic_logits.device
    batch_size, num_classes, height, width = semantic_logits.shape
    
    # 1. 语义分割：对logits应用softmax并取argmax
    semantic_prob = F.softmax(semantic_logits, dim=1)  # (B, n_classes+1, H, W)
    semantic_seg = torch.argmax(semantic_prob, dim=1)  # (B, H, W)
    
    # 初始化全景分割结果
    panoptic_seg = semantic_seg.clone()
    
    # 对每个batch单独处理
    for b in range(batch_size):
        # 2. 中心点检测：使用非极大值抑制找到实例中心
        centers = _find_instance_centers(
            center_heatmap[b, 0],  # (H, W)
            threshold=center_threshold,
            nms_kernel=nms_kernel,
            top_k=top_k
        )
        
        if len(centers) == 0:
            continue
            
        # 3. 实例分组：根据偏移量将像素分配给实例
        instance_map = _group_pixels_to_instances(
            offset_map[b],  # (2, H, W)
            centers,
            semantic_seg[b],  # (H, W)
            semantic_prob[b]  # (n_classes+1, H, W)
        )
        
        # 4. 后处理：移除小实例，合并stuff类别
        panoptic_seg[b] = _postprocess_instances(
            instance_map,
            semantic_seg[b],
            stuff_area=stuff_area,
            num_classes=num_classes
        )
    
    return semantic_seg, panoptic_seg


def _find_instance_centers(
    center_heatmap: torch.Tensor,
    threshold: float = 0.1,
    nms_kernel: int = 7,
    top_k: int = 200
) -> list[tuple[int, int]]:
    """
    在中心点热力图中找到实例中心。
    
    Args:
        center_heatmap: 中心点热力图，形状 (H, W)
        threshold: 中心点检测阈值
        nms_kernel: 非极大值抑制的核大小
        top_k: 保留的最大中心点数量
        
    Returns:
        list: 中心点坐标列表 [(y, x), ...]
    """
    # 应用阈值过滤
    heatmap = center_heatmap.clone()
    heatmap[heatmap < threshold] = 0
    
    # 非极大值抑制
    pad = nms_kernel // 2
    hmax = F.max_pool2d(
        heatmap.unsqueeze(0).unsqueeze(0),
        kernel_size=nms_kernel,
        stride=1,
        padding=pad
    ).squeeze()
    
    # 找到局部最大值
    keep = (heatmap == hmax) & (heatmap > 0)
    
    # 获取中心点坐标和置信度
    y_coords, x_coords = torch.where(keep)
    scores = heatmap[y_coords, x_coords]
    
    # 按置信度排序并保留top_k
    if len(scores) > top_k:
        _, indices = torch.topk(scores, top_k)
        y_coords = y_coords[indices]
        x_coords = x_coords[indices]
    
    return list(zip(y_coords.cpu().numpy(), x_coords.cpu().numpy()))


def _group_pixels_to_instances(
    offset_map: torch.Tensor,
    centers: list[tuple[int, int]],
    semantic_seg: torch.Tensor,
    semantic_prob: torch.Tensor
) -> torch.Tensor:
    """
    根据偏移量将像素分配给最近的实例中心。
    
    Args:
        offset_map: 偏移图，形状 (2, H, W)
        centers: 实例中心坐标列表
        semantic_seg: 语义分割结果，形状 (H, W)
        semantic_prob: 语义分割概率，形状 (n_classes+1, H, W)
        
    Returns:
        torch.Tensor: 实例分割图，形状 (H, W)，0为背景，>0为实例ID
    """
    height, width = semantic_seg.shape
    instance_map = torch.zeros_like(semantic_seg)
    
    if len(centers) == 0:
        return instance_map
    
    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=offset_map.device),
        torch.arange(width, device=offset_map.device),
        indexing='ij'
    )
    
    # 计算每个像素指向的位置
    # offset_map[0] 是 dy，offset_map[1] 是 dx
    pointed_y = y_grid.float() + offset_map[0]  # (H, W)
    pointed_x = x_grid.float() + offset_map[1]  # (H, W)
    
    # 对每个中心点，找到指向它的像素
    for instance_id, (center_y, center_x) in enumerate(centers, start=1):
        # 计算每个像素指向位置与中心点的距离
        dist_y = pointed_y - center_y
        dist_x = pointed_x - center_x
        distance = torch.sqrt(dist_y ** 2 + dist_x ** 2)
        
        # 找到距离最近且属于前景的像素
        # 排除背景类别（类别0）
        foreground_mask = semantic_seg > 0
        
        # 使用距离阈值（例如8像素）来限制实例范围
        distance_threshold = 8.0
        valid_mask = (distance < distance_threshold) & foreground_mask
        
        # 为了避免重叠，只分配给当前未分配的像素
        unassigned_mask = instance_map == 0
        final_mask = valid_mask & unassigned_mask
        
        instance_map[final_mask] = instance_id
    
    return instance_map


def _postprocess_instances(
    instance_map: torch.Tensor,
    semantic_seg: torch.Tensor,
    stuff_area: int = 2048,
    num_classes: int = 1
) -> torch.Tensor:
    """
    后处理实例分割结果：移除小实例，处理stuff类别。
    
    Args:
        instance_map: 实例分割图，形状 (H, W)
        semantic_seg: 语义分割结果，形状 (H, W)
        stuff_area: stuff类别的最小面积阈值
        num_classes: 前景类别数量
        
    Returns:
        torch.Tensor: 后处理后的全景分割结果
    """
    panoptic_seg = semantic_seg.clone()
    
    # 获取所有实例ID
    instance_ids = torch.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]  # 排除背景
    
    # 为每个实例分配唯一的全景ID
    # 全景ID编码：class_id * 1000 + instance_id
    for instance_id in instance_ids:
        instance_mask = instance_map == instance_id
        
        # 获取该实例的主要语义类别
        instance_semantic = semantic_seg[instance_mask]
        if len(instance_semantic) == 0:
            continue
            
        # 使用众数作为实例的语义类别
        main_class = torch.mode(instance_semantic).values.item()
        
        # 检查实例大小，移除过小的实例
        instance_area = instance_mask.sum().item()
        if instance_area < 32:  # 最小实例面积阈值
            continue
            
        # 生成全景ID：类别ID * 1000 + 实例ID
        panoptic_id = main_class * 1000 + instance_id.item()
        panoptic_seg[instance_mask] = panoptic_id
    
    # 处理stuff类别（大面积的背景区域）
    # 对于没有被分配给任何实例的区域，如果面积足够大，保持其语义标签
    unassigned_mask = instance_map == 0
    for class_id in range(1, num_classes + 1):
        class_mask = (semantic_seg == class_id) & unassigned_mask
        class_area = class_mask.sum().item()
        
        if class_area >= stuff_area:
            # stuff类别使用类别ID * 1000作为全景ID
            panoptic_seg[class_mask] = class_id * 1000
    
    return panoptic_seg




def select_postprocess_fn(name: str) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]|None:
    if "segment" in name:
        if "instance" in name:
            return postprocess_instance_segmentation
        elif "panoptic" in name:
            return postprocess_panoptic_segmentation
        else:
            return postprocess_binary_segmentation
    elif "class" in name:
        if "multi" in name or "multiple" in name:
            return postprocess_multiclass_classification
        else:
            return postprocess_binary_classification
    elif "regress" in name:
        return postprocess_regression
    return None


# --- Predict-stage postprocess: model output -> saveable format (per sample) ---

def _ensure_pred_4d(pred: torch.Tensor) -> torch.Tensor:
    """(1, H, W) or (1, C, H, W) -> (1, C, H, W)."""
    if pred.dim() == 3:
        return pred.unsqueeze(1)
    return pred


def predict_postprocess_binary_segmentation(
    pred: torch.Tensor,
    original_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """Pred logits [1,1,H,W] or [1,H,W] -> PIL Image L (0/255)."""
    pred = _ensure_pred_4d(pred.detach())
    prob = F.sigmoid(pred)
    mask = (prob >= 0.5).float()
    mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask, mode="L")
    if original_size:
        img = img.resize((original_size[1], original_size[0]), Image.NEAREST)
    return img


def predict_postprocess_instance_segmentation(
    pred: torch.Tensor,
    original_size: Tuple[int, int] | None = None,
) -> Image.Image:
    """Pred logits [1, C, H, W] -> PIL Image L (class indices)."""
    pred = _ensure_pred_4d(pred.detach())
    prob = F.softmax(pred, dim=1)
    mask = torch.argmax(prob, dim=1, keepdim=False)
    mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)
    img = Image.fromarray(mask, mode="L")
    if original_size:
        img = img.resize((original_size[1], original_size[0]), Image.NEAREST)
    return img


def predict_postprocess_binary_classification(pred: torch.Tensor) -> Tuple[int, float]:
    """Pred logits [1, 1] or [1, C] -> (class_id, prob)."""
    pred = pred.detach().float().squeeze()
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    prob = F.sigmoid(pred).cpu().numpy()
    p = float(prob[0]) if len(prob) == 1 else float(prob[1])
    class_id = 1 if p >= 0.5 else 0
    return class_id, p


def predict_postprocess_multiclass_classification(pred: torch.Tensor) -> Tuple[int, np.ndarray]:
    """Pred logits [1, C] -> (class_id, probs)."""
    pred = pred.detach().float().squeeze(0)
    probs = F.softmax(pred, dim=0).cpu().numpy()
    class_id = int(np.argmax(probs))
    return class_id, probs


def predict_postprocess_regression(pred: torch.Tensor) -> float:
    """Pred [1, ...] -> scalar."""
    return float(pred.detach().cpu().numpy().ravel()[0])


def get_predict_postprocess_fn(name: str) -> Callable[..., Any] | None:
    """Return a predict-stage function: (pred, optional original_size) -> saveable (Image, scalar, or (id, prob))."""
    if not name:
        return None
    name_lower = name.lower()
    if "segment" in name_lower:
        if "instance" in name_lower:
            return predict_postprocess_instance_segmentation
        return predict_postprocess_binary_segmentation
    if "class" in name_lower:
        if "multi" in name_lower or "multiple" in name_lower:
            return predict_postprocess_multiclass_classification
        return predict_postprocess_binary_classification
    if "regress" in name_lower:
        return predict_postprocess_regression
    return None
