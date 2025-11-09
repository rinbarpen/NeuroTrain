import base64
import mimetypes
from collections.abc import Sequence

# OPENAI FORMAT
class ConversationHistory:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.messages = []
    
    def add_user_message(self, message: str, image: str | None = None, images: Sequence[str] | None = None):
        content = [{
            'type': 'text',
            'text': message
        }]

        for img in self._normalize_images(image, images):
            content.append(self._build_image_content(img))

        self.messages.append({
            'role': 'user',
            'content': content
        })
    
    def add_ai_message(self, message: str, image: str | None = None, images: Sequence[str] | None = None):
        if image or images:
            content = [{
                'type': 'text',
                'text': message
            }]
            for img in self._normalize_images(image, images):
                content.append(self._build_image_content(img))
        else:
            content = message

        self.messages.append({'role': 'assistant', 'content': content})
    
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

    @staticmethod
    def _normalize_images(image: str | None, images: Sequence[str] | None) -> list[str]:
        image_list: list[str] = []
        if image:
            image_list.append(image)
        if images:
            image_list.extend(img for img in images if img)
        return image_list

    @staticmethod
    def _build_image_content(image: str):
        if image.startswith('http'):
            return {
                'type': 'image_url',
                'image_url': {
                    'url': image
                }
            }

        mime, _ = mimetypes.guess_type(image)
        if mime is None:
            mime = 'image/png'
        with open(image, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return {
            'type': 'image_url',
            'image_url': {
                'url': f'data:{mime};base64,{b64}'
            }
        }
