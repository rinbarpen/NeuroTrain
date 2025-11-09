import torch
from torch import nn
import torch.nn.functional as F

def dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor, # y_pred 预期为 logits
    *,
    class_axis: int = 1,
    smooth: float = 1e-6,
    # 新增参数，指定是否在内部进行激活
    apply_activation: bool = True,
    activation_type: str = 'sigmoid', # 'sigmoid' for binary/multilabel, 'softmax' for multiclass
) -> torch.Tensor:
    if class_axis < 0:
        class_axis += y_true.ndim

    # 对预测值应用激活函数
    if apply_activation:
        if activation_type == 'sigmoid':
            y_pred = torch.sigmoid(y_pred)
        elif activation_type == 'softmax':
            # 对于 softmax，需要知道类别维度
            y_pred = torch.softmax(y_pred, dim=class_axis)
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}")

    reduction_dims = tuple(i for i in range(y_true.ndim) if i != class_axis)

    intersection = (y_true * y_pred).sum(dim=reduction_dims)
    union = (y_true + y_pred).sum(dim=reduction_dims)
    union = torch.where(union == 0, intersection, union)

    dice_scores = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_scores.mean()

def kl_divergence_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, *, epsilon: float = 1e-7
):
    # 确保输入为概率分布
    y_true = torch.clamp(y_true, epsilon, 1.0 - epsilon)
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # 计算 KL 散度: sum(p * log(p/q))
    kl_div = torch.sum(y_true * torch.log(y_true / y_pred), dim=1)
    return torch.mean(kl_div)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float=1.0, reduction: str='batchmean'):
    tearcher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    # probs曲线更加平滑
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1).detach()
    # kl_div的学生输入必须是log的，而教师的不需要log
    kl = F.kl_div(
        student_log_probs,
        tearcher_probs,
        reduction=reduction
    )
    # 让loss曲线更加突出
    return (temperature ** 2) * kl


class CombineCriterion(nn.Module):
    def __init__(self, *loss_fns):
        super(CombineCriterion, self).__init__()
        self.loss_fns = loss_fns

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        return torch.stack([loss_fn(targets, preds) for loss_fn in self.loss_fns]).sum()

class Loss(nn.Module):
    def __init__(self, loss_fn: nn.Module|None=None, weight: float=1.0):
        super(Loss, self).__init__()
        self.loss_fn = loss_fn
        self.weight = weight
    
    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None:
            raise NotImplementedError("Loss function is not defined.")
        return self.loss_fn(targets, preds) * self.weight


class DiceLoss(Loss):
    def __init__(self, weight: float=1.0, activation_type: str='sigmoid'): # 增加 activation_type 参数
        super(DiceLoss, self).__init__(weight=weight)
        self.activation_type = activation_type

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = dice_loss(targets, preds, apply_activation=True, activation_type=self.activation_type)
        return loss * self.weight

class KLLoss(Loss):
    def __init__(self, weight: float=1.0):
        super(KLLoss, self).__init__(weight=weight)

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = kl_divergence_loss(targets, preds)
        return loss * self.weight

class ContrastiveLoss(Loss):
    """
    对比学习损失函数，用于Region-Text对齐
    
    支持对称损失（text2region + region2text）和单边损失
    """
    def __init__(self, weight: float=1.0, temperature: float=0.07, symmetric: bool=True):
        """
        Args:
            weight: 损失权重
            temperature: 温度参数，用于缩放相似度
            symmetric: 是否使用对称损失（双边损失）
        """
        super(ContrastiveLoss, self).__init__(weight=weight)
        self.temperature = temperature
        self.symmetric = symmetric
    
    def forward(self, targets: torch.Tensor, preds: dict) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            targets: 不需要（对比损失使用batch内的配对）
            preds: 字典，包含以下键:
                - 'logits_per_text': (B, B) 文本到区域的相似度矩阵
                - 'logits_per_region': (B, B) 区域到文本的相似度矩阵（可选）
        
        Returns:
            对比学习损失
        """
        logits_per_text = preds.get('logits_per_text')
        logits_per_region = preds.get('logits_per_region')
        
        if logits_per_text is None:
            raise ValueError("preds must contain 'logits_per_text'")
        
        batch_size = logits_per_text.shape[0]
        device = logits_per_text.device
        labels = torch.arange(batch_size, device=device)
        
        # 文本到区域的损失
        text_loss = F.cross_entropy(logits_per_text, labels)
        
        if self.symmetric and logits_per_region is not None:
            # 对称损失：区域到文本的损失
            region_loss = F.cross_entropy(logits_per_region, labels)
            total_loss = (text_loss + region_loss) / 2
        else:
            total_loss = text_loss
        
        return total_loss * self.weight


class DistillationLoss(Loss):
    def __init__(self, temperature: float=1.0, weight: float=1.0):
        super(DistillationLoss, self).__init__(weight=weight)
        self.temperature = temperature

    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        loss = distillation_loss(
            student_logits=preds,
            teacher_logits=targets,
            temperature=self.temperature
        )
        return loss * self.weight

class SemanticSegmentationLoss(Loss):
    """
    语义分割损失函数，结合交叉熵损失和Dice损失
    """
    def __init__(self, weight: float=1.0, ce_weight: float=1.0, dice_weight: float=1.0, 
                 ignore_index: int=-100, label_smoothing: float=0.0):
        super(SemanticSegmentationLoss, self).__init__(weight=weight)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        
    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            targets: [N, H, W] 语义分割标签
            preds: [N, C, H, W] 语义分割预测logits
        """
        # 交叉熵损失
        ce_loss = self.ce_loss(preds, targets)
        
        # Dice损失 - 需要将targets转换为one-hot编码
        num_classes = preds.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dice_loss_val = dice_loss(targets_one_hot, preds, apply_activation=True, activation_type='softmax')
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss_val
        return total_loss * self.weight


class InstanceSegmentationLoss(Loss):
    """
    实例分割损失函数，包含中心点检测损失和偏移回归损失
    """
    def __init__(self, weight: float=1.0, center_weight: float=1.0, offset_weight: float=1.0):
        super(InstanceSegmentationLoss, self).__init__(weight=weight)
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        
    def forward(self, targets: dict, preds: dict) -> torch.Tensor:
        """
        Args:
            targets: dict包含'center'和'offset'键
                - center: [N, 1, H, W] 中心点热力图标签
                - offset: [N, 2, H, W] 偏移量标签
            preds: dict包含'center'和'offset'键
                - center: [N, 1, H, W] 中心点预测
                - offset: [N, 2, H, W] 偏移量预测
        """
        # 中心点检测损失 - 使用focal loss处理类别不平衡
        center_loss = self._focal_loss(preds['center'], targets['center'])
        
        # 偏移回归损失 - 使用L1损失，只在前景区域计算
        offset_loss = self._offset_loss(preds['offset'], targets['offset'], targets['center'])
        
        total_loss = self.center_weight * center_loss + self.offset_weight * offset_loss
        return total_loss * self.weight
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float=0.25, gamma: float=2.0) -> torch.Tensor:
        """
        Focal Loss用于处理中心点检测的类别不平衡问题
        """
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss
            
        return loss.mean()
    
    def _offset_loss(self, pred_offset: torch.Tensor, target_offset: torch.Tensor, 
                    center_mask: torch.Tensor) -> torch.Tensor:
        """
        偏移量回归损失，只在前景区域计算
        """
        # 创建前景掩码（中心点附近区域）
        mask = (center_mask > 0.1).float()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_offset.device, requires_grad=True)
        
        # L1损失
        loss = F.l1_loss(pred_offset * mask, target_offset * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-6)
        
        return loss


class BoundingBoxRegressionLoss(Loss):
    """
    边界框回归损失函数，使用Smooth L1损失
    """
    def __init__(self, weight: float=1.0, beta: float=1.0):
        super(BoundingBoxRegressionLoss, self).__init__(weight=weight)
        self.beta = beta
        
    def forward(self, targets: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            targets: [N, 4] 或 [N, num_boxes, 4] 目标边界框 (x1, y1, x2, y2)
            preds: [N, 4] 或 [N, num_boxes, 4] 预测边界框
            mask: [N] 或 [N, num_boxes] 有效框的掩码
        """
        loss = F.smooth_l1_loss(preds, targets, reduction='none', beta=self.beta)
        
        if mask is not None:
            # 只计算有效框的损失
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
            
        return loss * self.weight


class ClassificationLoss(Loss):
    """
    分类损失函数，支持二分类和多分类
    """
    def __init__(self, weight: float=1.0, num_classes: int=2, label_smoothing: float=0.0,
                 focal_alpha: float=-1, focal_gamma: float=2.0):
        super(ClassificationLoss, self).__init__(weight=weight)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
    def forward(self, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            targets: [N] 或 [N, num_classes] 分类标签
            preds: [N, num_classes] 分类预测logits
        """
        if self.focal_alpha >= 0 and self.num_classes > 2:
            # 使用Focal Loss
            loss = self._focal_loss(preds, targets)
        else:
            # 使用标准损失
            if self.num_classes == 2:
                # 二分类
                if targets.dim() == 1:
                    targets = targets.float()
                    preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds[:, 1]
                loss = self.loss_fn(preds, targets)
            else:
                # 多分类
                loss = self.loss_fn(preds, targets)
                
        return loss * self.weight
    
    def _focal_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        多分类Focal Loss
        """
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


class PanopticLoss(Loss):
    """
    全景分割损失函数，结合语义分割、实例分割和一致性损失
    """
    def __init__(self, weight: float=1.0, semantic_weight: float=1.0, 
                 instance_weight: float=1.0, consistency_weight: float=0.1):
        super(PanopticLoss, self).__init__(weight=weight)
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.consistency_weight = consistency_weight
        
        # 初始化子损失函数
        self.semantic_loss = SemanticSegmentationLoss()
        self.instance_loss = InstanceSegmentationLoss()
        
    def forward(self, targets: dict, preds: dict) -> torch.Tensor:
        """
        Args:
            targets: dict包含以下键:
                - semantic: [N, H, W] 语义分割标签
                - center: [N, 1, H, W] 中心点热力图标签
                - offset: [N, 2, H, W] 偏移量标签
            preds: dict包含以下键:
                - semantic: [N, C, H, W] 语义分割预测logits
                - center: [N, 1, H, W] 中心点预测
                - offset: [N, 2, H, W] 偏移量预测
        """
        # 语义分割损失
        semantic_loss = self.semantic_loss(targets['semantic'], preds['semantic'])
        
        # 实例分割损失
        instance_targets = {'center': targets['center'], 'offset': targets['offset']}
        instance_preds = {'center': preds['center'], 'offset': preds['offset']}
        instance_loss = self.instance_loss(instance_targets, instance_preds)
        
        # 一致性损失 - 确保语义分割和实例分割的一致性
        consistency_loss = self._consistency_loss(targets, preds)
        
        total_loss = (self.semantic_weight * semantic_loss + 
                     self.instance_weight * instance_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss * self.weight
    
    def _consistency_loss(self, targets: dict, preds: dict) -> torch.Tensor:
        """
        计算语义分割和实例分割之间的一致性损失
        """
        # 获取语义分割预测
        semantic_pred = torch.softmax(preds['semantic'], dim=1)
        
        # 获取实例中心预测
        center_pred = torch.sigmoid(preds['center'])
        
        # 计算thing类别的一致性（假设thing类别索引大于某个阈值）
        # 这里简化处理，实际应用中需要根据具体的类别定义调整
        thing_classes = semantic_pred[:, 1:, :, :]  # 假设第0类是背景
        thing_semantic = thing_classes.sum(dim=1, keepdim=True)
        
        # 一致性损失：实例中心应该在thing类别区域内
        consistency_loss = F.mse_loss(center_pred * thing_semantic, center_pred)
        
        return consistency_loss
        
def get_criterion(c: dict):
    """
    根据配置字典创建相应的损失函数
    
    Args:
        c: 损失函数配置字典，包含type、weight和config字段
        
    Returns:
        相应的损失函数实例
    """
    c_type = c['type'].lower()
    weight = c.get('weight', 1)
    cc = c.get('config', {})

    # 基础损失函数
    if 'dice' in c_type:
        return DiceLoss(weight, **cc)
    elif 'bce' in c_type:
        return Loss(nn.BCEWithLogitsLoss(**cc), weight)
    elif 'ce' in c_type:
        return Loss(nn.CrossEntropyLoss(**cc), weight)
    
    # 分割相关损失函数
    elif 'semantic' in c_type or 'semantic_seg' in c_type:
        return SemanticSegmentationLoss(weight, **cc)
    elif 'instance' in c_type or 'instance_seg' in c_type:
        return InstanceSegmentationLoss(weight, **cc)
    elif 'panoptic' in c_type:
        return PanopticLoss(weight, **cc)
    
    # 检测相关损失函数
    elif 'bbox' in c_type or 'regression' in c_type:
        return BoundingBoxRegressionLoss(weight, **cc)
    elif 'classification' in c_type or 'cls' in c_type:
        return ClassificationLoss(weight, **cc)
    
    # 其他损失函数
    elif 'kl' in c_type:
        return KLLoss(weight, **cc)
    elif 'distillation' in c_type:
        return DistillationLoss(weight=weight, **cc)
    elif 'contrastive' in c_type:
        return ContrastiveLoss(weight=weight, **cc)

    return None
