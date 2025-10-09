from pathlib import Path
import base64
import mimetypes
import os.path as osp
from PIL import Image

# OPENAI FORMAT
class ConversationHistory:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.messages = []
    
    def add_user_message(self, message: str, image: str=None):
        # 先放入文本内容
        content = [
            {
                'type': 'text',
                'text': message
            }
        ]

        if image is not None:
            if image.startswith('https') or image.startswith('http'):
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': image
                    }
                })
            else:
                # 推断 MIME 类型，推断失败则按 PNG 处理（多数模型/SDK需要 image/* 前缀）
                mime, _ = mimetypes.guess_type(image)
                if mime is None:
                    mime = 'image/png'
                img = Image.open(image)
                b64 = base64.b64encode(img.tobytes()).decode('utf-8')
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime};base64,{b64}'
                    }
                })

        self.messages.append({
            'role': 'user',
            'content': content
        })
    
    def add_ai_message(self, message: str):
        self.messages.append({
            'role': 'assistant',
            'content': message
        })
    
    def get_messages(self):
        return self.messages
    
    def get_last_message(self):
        return self.messages[-1]

    def build(self):
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            }
        ] + self.messages
        return messages

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
