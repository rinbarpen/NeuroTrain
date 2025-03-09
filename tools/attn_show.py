import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

from toolkit.vit_attn_show import show_mask_on_image

def vit_image_on_mask(input_path, output_path, show=False):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (256, 256))
    np_img = np.array(img)[:, :, ::-1]

    mask = show_mask_on_image(img, np.random.rand(256, 256))
    
    # 使用 matplotlib 显示图像和 mask
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np_img)
    plt.title("Input Image")
    plt.axis('off')  # 关闭坐标轴
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Mask Image")
    plt.axis('off')  # 关闭坐标轴
    
    plt.savefig(output_path)  # 保存图像
    
    if show:
        plt.show()

def grad_attn_image(image: np.ndarray, attentions: torch.Tensor, gradients: torch.Tensor, discard_ratio: float=0.9) -> np.ndarray:
    # attentions: (B, C, W, H)
    image_width, image_height = image.shape[:2]
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):                
            weights = grad
            attention_fused = (attention*weights).mean(axis=0)
            attention_fused = F.relu(attention_fused, inplace=True)

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_fused.flatten(1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            # TODO: Check it
            I = torch.eye(attention_fused.size(-1))
            a = (attention_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    mask = result[0, 0, 1 :]
    mask = mask.reshape(image_width, image_height).numpy()
    mask = mask / np.max(mask)
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, required=True, help='Path to the output image')
    parser.add_argument('--show', action='store_true', help='Show the image using plt.show()')
    
    args = parser.parse_args()
    
    vit_image_on_mask(args.input, args.output, args.show)
