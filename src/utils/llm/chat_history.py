from src.utils.llm.llm_utils import chat_template

class ChatHistory:
    def __init__(self, processor):
        self.processor = processor
        self.history = []
    
    def add_message(self, role: str, text: str, image: str|None=None):
        conversation = chat_template(role, text, image)
        self.history.append(conversation)
        return self

    def add_user_message(self, text: str, image: str|None=None):
        self.add_message('user', text, image)
        return self

    def add_assistant_message(self, text: str, image: str|None=None):
        self.add_message('assistant', text, image)
        return self

    def __len__(self):
        return len(self.history)

    def get_history(self):
        return self.history
    
    def last_message(self):
        if self.history and len(self.history) > 0:
            return self.history[-1]['content'][0]['text']
        return None
    
    def to_openai(self):
        return self.history

    def to_llava(self, *, processor=None):
        if processor is not None:
            return processor.add_chat_template(self.history, add_generation_prompt=True)
        
        t = ""
        for message in self.history:
            if message['role'] == 'user':
                t += "<image>USER:{}".format(message['content'][0]['text'])
            else:
                t += "ASSISTANT:{}".format(message['content'][0]['text'])
        return t

    def to_mistral_instruct(self):
        t = ""
        for message in self.history:
            if message['role'] == 'user':
                t += "<s>[INST]{}[/INST]".format(message['content'][0]['text'])
            else:
                t += "{}</s>".format(message['content'][0]['text'])
        return t
