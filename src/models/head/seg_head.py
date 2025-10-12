import torch
from torch import nn
import torch.nn.functional as F

class SemanticSegHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(SemanticSegHead, self).__init__()
        self.is_multiclass = n_classes > 1
        if self.is_multiclass:
            n_classes += 1 # add background class
            self.act = nn.Softmax(dim=1)
        else:
            self.act = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1, stride=1, padding=0)
        
        self._init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conv.bias, 0)

class InstanceSegHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, roi_size: int, hidden_dim: int = 1024):
        super(InstanceSegHead, self).__init__()
        self.is_multiclass = n_classes > 1
        if self.is_multiclass:
            all_n_classes = n_classes + 1

        cls_dim = reg_dim = in_channels * roi_size * roi_size

        self.reg_cls_linear = nn.Sequential(
            nn.Linear(reg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mask_head = nn.Conv2d(in_channels, all_n_classes, kernel_size=1) # first is background
        self.reg_head = nn.Linear(hidden_dim, 4 * n_classes), # (x, y, w, h) 
        self.cls_head = nn.Linear(hidden_dim, all_n_classes), # which cls for every roi

        self._init_weights()

    def forward(self, x):
        mask = self.mask_head(x)
        rc_x = self.reg_cls_linear(x.flatten(start_dim=1))
        reg = self.reg_head(rc_x) # (batch, 4 * n_classes)
        cls = self.cls_head(rc_x) # (batch, all_n_classes)
        return mask, reg, cls

    def _init_weights(self):
        """初始化网络权重"""
        # 分类层使用正态分布初始化
        nn.init.normal_(self.cls_head[-1].weight, std=0.01)
        nn.init.constant_(self.cls_head[-1].bias, 0)
        
        # 回归层使用正态分布初始化，偏置初始化为0
        nn.init.normal_(self.reg_head[-1].weight, std=0.001)
        nn.init.constant_(self.reg_head[-1].bias, 0)

        # 掩码层使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.mask_head.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.mask_head.bias, 0)

class PanopticSegHead(nn.Module):
    """Panoptic-DeepLab风格的全景分割头部
    
    实现了Panoptic-DeepLab论文中的三分支架构：
    1. 语义分割分支：预测每个像素的类别
    2. 实例中心分支：预测每个像素是否为实例中心
    3. 偏移回归分支：预测每个像素到其所属实例中心的偏移量
    """
    
    def __init__(self, in_channels: int, n_classes: int):
        """初始化Panoptic-DeepLab头部
        
        Args:
            in_channels: 输入特征图的通道数
            n_classes: 语义分割的类别数（不包括背景）
        """
        super(PanopticSegHead, self).__init__()
        self.n_classes = n_classes
        self.is_multiclass = n_classes > 1
        assert self.is_multiclass, "PanopticSegHead only support multiclass segmentation"
        
        # 语义分割分支：预测每个像素的类别概率（包括背景类）
        # 输出通道数为 n_classes + 1（背景类）
        self.semantic_head = nn.Conv2d(
            in_channels, 
            n_classes + 1, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        # 实例中心分支：预测每个像素是否为实例中心（类无关，1个通道）
        # 输出范围 [0, 1]，表示中心点的置信度
        self.center_head = nn.Conv2d(
            in_channels, 
            1, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        # 偏移回归分支：预测每个像素到实例中心的偏移向量（类无关，2个通道：dx, dy）
        # 输出范围 [-∞, +∞]，表示相对偏移量
        self.offset_head = nn.Conv2d(
            in_channels, 
            2, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        # 初始化权重
        self._init_weights()
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征图，形状为 (B, C, H, W)
            
        Returns:
            tuple: (semantic_logits, center_heatmap, offset_map)
                - semantic_logits: 语义分割logits，形状 (B, n_classes+1, H, W)
                - center_heatmap: 中心点热力图，形状 (B, 1, H, W)，值域 [0, 1]
                - offset_map: 偏移图，形状 (B, 2, H, W)，值域 [-∞, +∞]
        """
        # 语义分割分支：输出原始logits，不应用激活函数
        # 这样可以在训练时使用CrossEntropyLoss，推理时再应用softmax
        semantic_logits = self.semantic_head(x)
        
        # 实例中心分支：使用sigmoid激活，输出中心点置信度
        # sigmoid确保输出在[0, 1]范围内，表示每个像素是实例中心的概率
        center_heatmap = torch.sigmoid(self.center_head(x))
        
        # 偏移回归分支：不使用激活函数，直接输出偏移量
        # 偏移量可以是正负值，表示到实例中心的相对位置
        # 通常会在训练时使用L1或SmoothL1损失
        offset_map = self.offset_head(x)
        
        return semantic_logits, center_heatmap, offset_map
    
    def _init_weights(self):
        """初始化网络权重
        
        使用Panoptic-DeepLab论文中推荐的初始化策略：
        - 语义分割头：Kaiming正态分布初始化
        - 中心点头：正态分布初始化，偏置设为负值以降低初始激活
        - 偏移头：正态分布初始化，较小的标准差
        """
        # 语义分割头：使用Kaiming初始化
        nn.init.kaiming_normal_(self.semantic_head.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.semantic_head.bias, 0)
        
        # 中心点头：使用正态分布初始化，偏置设为负值
        # 负偏置有助于在训练初期降低false positive
        nn.init.normal_(self.center_head.weight, std=0.01)
        nn.init.constant_(self.center_head.bias, -2.19)  # ln(0.1/0.9) ≈ -2.19
        
        # 偏移头：使用较小标准差的正态分布初始化
        # 较小的初始权重有助于稳定训练
        nn.init.normal_(self.offset_head.weight, std=0.001)
        nn.init.constant_(self.offset_head.bias, 0)