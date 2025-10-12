from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

from src.config import PRETRAINED_MODEL_DIR

class Model:
    def __init__(self, model_id='meta-llama/Llama-2-7b-hf', 
                 extract_layers: tuple[int, int] = (3, 11),
                 num_classes: int = 10,
                 *, cache_dir=PRETRAINED_MODEL_DIR, device='cuda', dtype=torch.float16):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map=device, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.device = device
        self.dtype = dtype

        self.extract_layers = extract_layers
        
        hidden_size = self.model.config.hidden_size
        num_attention_heads = self.model.config.num_attention_heads
        
        self.attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True, dtype=dtype)
        self.classifier = LinearClassifier(hidden_size, num_classes)
        
        self.init()
        
    def init(self):
        self.model.eval()
        self.attn.eval()
        self.classifier.eval()
        self.model.to(self.device)
        self.attn.to(self.device)
        self.classifier.to(self.device)

    def extract_features(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        # Ensure hidden states are outputted
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states
    
    def forward(self, prompt: str):
        with torch.no_grad():
            features = self.extract_features(prompt)
            early_feats, late_feats = features[self.extract_layers[0]], features[self.extract_layers[1]]
            
            # Use early_feats as query, and late_feats as key/value for attention
            attn_output, _ = self.attn(query=early_feats, key=late_feats, value=late_feats)
            
            # Pool output and classify from the last token
            pooled_output = attn_output[:, -1, :]
            logits = self.classifier(pooled_output)
            
        return logits