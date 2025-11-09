import torch
import torch.nn.functional as F


def compute_similarity_matrix(text_features: torch.Tensor, region_features: torch.Tensor, temperature: float=0.07, *, normalize: bool=True) -> torch.Tensor:
    """
    Compute the similarity matrix between text and region features.
    """
    if normalize:
        text_features = F.normalize(text_features, dim=-1)
        region_features = F.normalize(region_features, dim=-1)
    return text_features @ region_features.T / temperature

def compute_logits_per_text(text_features: torch.Tensor, image_features: torch.Tensor, temperature: float=0.07, *, normalize: bool=True) -> torch.Tensor:
    """
    Compute the logits per text.
    """
    return compute_similarity_matrix(text_features, image_features, temperature, normalize=normalize)

def compute_logits_per_image(image_features: torch.Tensor, text_features: torch.Tensor, temperature: float=0.07, *, normalize: bool=True) -> torch.Tensor:
    """
    Compute the logits per image.
    """
    return compute_similarity_matrix(image_features, text_features, temperature, normalize=normalize)

def compute_text_probabilities(logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    得到文本的概率分布

    Args:
        logits_per_text: (B, N) 文本与区域的对数相似度矩阵
    Returns:
        probs: (B, N) 概率分布
    """
    return F.softmax(logits_per_text, dim=-1)

def compute_image_probabilities(logits_per_image: torch.Tensor) -> torch.Tensor:
    """
    得到图像的概率比

    Args:
        logits_per_image: (B, N) 图像与文本的对数相似度矩阵
    Returns:
        probs: (B, N) 概率分布
    """
    return F.softmax(logits_per_image, dim=-1)
