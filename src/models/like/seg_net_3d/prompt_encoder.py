from torch import nn
import torch

from typing import Optional, List, Tuple
from einops import rearrange
from models.position_encoding import SpatialPositionEmbedding

# window_size is the batch size of masks
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, image_size: tuple[int, int], patched_image_size: tuple[int, int], mask_channels: int, n_masks: int, n_tokens_per_mask: int) -> None:
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patched_image_size = patched_image_size
        self.n_masks = n_masks
        self.n_mask_tokens = n_tokens_per_mask * n_masks

        self.pe = SpatialPositionEmbedding(embed_dim // 2)

        self.blank_embedding = nn.Embedding(1, embed_dim)
        self.point_encoder = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(2)]
        )
        self.bbox_encoder = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(2)]
        )

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(n_masks, mask_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(mask_channels // 4),
            nn.GELU(),
            nn.Conv2d(mask_channels // 4, mask_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(mask_channels),
            nn.GELU(),
            nn.Conv2d(mask_channels, embed_dim, kernel_size=1),
        )
        # self.no_mask_embedding = nn.Embedding(1, embed_dim)

    def forward(self, points: Optional[tuple[torch.Tensor, torch.Tensor]], bboxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # sparse tokens
        b = self._get_batch_size(points, bboxes)
        sparse_tokens = torch.empty((b, 0, self.embed_dim), device=self._get_device()) # (B, N, D)

        if points:
            coords, labels = points
            point_tokens = self._embed_points(coords, labels, pad=(bboxes is None))
            sparse_tokens = torch.cat([sparse_tokens, point_tokens], dim=1)

        if bboxes:
            bbox_tokens = self._embed_boxes(bboxes)
            sparse_tokens = torch.cat([sparse_tokens, bbox_tokens], dim=1)

        # dense tokens
        if masks:
            # blank fill
            b, n, h, w, d = masks.shape
            n_blank = self.window_size - d
            blanks = torch.zeros(b, n, h, w, n_blank)
            masks = torch.cat([blanks, masks], dim=-1) # b, self.n_masks, h, w, ws
        else:
            h, w = self.image_size
            blanks = torch.zeros(b, self.n_masks, h, w, self.window_size)
            masks = blanks # b, self.n_masks, h, w, ws

        masks = rearrange(masks, 'b c h w d -> b d c h w')
        dense_tokens = self._embed_masks(masks)
        masks = rearrange(masks, 'b d c h w -> b c h w d')

        return sparse_tokens, dense_tokens # (B, N, D), (B, Dim, H, W, window_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5 # shift to center of pixel

        if pad:
            # B, N, 2
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            
        point_embed = self.pe.forward_with_coords(points, self.image_size)
        point_embed[labels == -1] = 0.0

        point_embedding = self.point_encoder(points, self.input_image_size)

        point_embedding[labels == -1] += self.not_a_point_embed.weight 
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes += 0.5 # shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_tokens = self.pe.forward_with_coords(coords, self.image_size)
        corner_tokens[:, 0, :] += self.bbox_encoder[0].weight
        corner_tokens[:, 1, :] += self.bbox_encoder[1].weight
        return corner_tokens

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        mask_tokens = self.mask_encoder(masks)
        return mask_tokens

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points:
            return points[0].shape[0]
        elif boxes:
            return boxes.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device
