from PIL import Image
import cv2
from pathlib import Path
from collections.abc import Sequence
from typing import Literal, List
from curl_cffi import requests
import io
import math

def chat_template(role: str, text: str, image: str | None = None, images: Sequence[str] | None = None):
    content = [{"type": "text", "text": text}]

    if image or images:
        image_list: list[str] = []
        if image:
            image_list.append(image)
        if images:
            image_list.extend(img for img in images if img)
        for _ in image_list:
            content.append({"type": "image"})

    conversation = {
        'role': role,
        'content': content,
    }

    return conversation

def image_to_PIL(image: Image.Image|cv2.Mat|str|Path, *, image_format: Literal['JPEG', 'PNG'] = 'JPEG') -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, cv2.Mat):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mode='RGB')
    elif isinstance(image, str):
        if image.startswith('http'):
            return Image.open(requests.get(image, stream=True).raw).convert('RGB')
        else:
            return Image.open(image).convert('RGB')
    elif isinstance(image, Path):
        return Image.open(image).convert('RGB')

def image_to_b64(image: Image.Image|cv2.Mat|str|Path, *, image_format: Literal['JPEG', 'PNG'] = 'JPEG') -> str:
    """
    将图像转换为base64编码的字符串
    """
    import base64
    if isinstance(image, Image.Image):
        with io.BytesIO() as output:
            image.save(output, format=image_format)
            return base64.b64encode(output.getvalue()).decode('utf-8')
    elif isinstance(image, cv2.Mat):
        _, buffer = cv2.imencode(f'.{image_format.lower()}', image)
        return base64.b64encode(buffer).decode('utf-8')
    elif isinstance(image, str):
        if image.startswith('http'):
            with requests.get(image) as response:
                return base64.b64encode(response.content).decode('utf-8')
        else:
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Path):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None
