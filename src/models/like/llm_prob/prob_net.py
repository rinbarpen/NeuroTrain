import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from ...llm.transformers import *

class FeatureGuidanceModule(nn.Module):
    """
    特征引导模块：使用早期特征作为查询来引导和优化晚期特征
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 交叉注意力机制：早期特征作为Query，晚期特征作为Key和Value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 门控机制，控制早期特征对晚期特征的影响程度
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, early_features: torch.Tensor, late_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            early_features: 早期特征 [batch_size, seq_len, hidden_dim]
            late_features: 晚期特征 [batch_size, seq_len, hidden_dim]
        Returns:
            guided_features: 引导后的特征 [batch_size, seq_len, hidden_dim]
        """
        # 使用早期特征作为查询，晚期特征作为键值对
        guided_features, attention_weights = self.cross_attention(
            query=early_features,
            key=late_features,
            value=late_features
        )
        
        # 特征融合：将原始晚期特征与引导特征结合
        concatenated = torch.cat([late_features, guided_features], dim=-1)
        fused_features = self.fusion_layer(concatenated)
        
        # 门控机制：控制融合程度
        gate_weights = self.gate(concatenated)
        final_features = gate_weights * fused_features + (1 - gate_weights) * late_features
        
        return final_features

class MultiLayerFeatureExtractor(nn.Module):
    """
    多层特征提取器：从LLM的不同层提取特征
    """
    def __init__(self, model, extract_layers: List[int]):
        super().__init__()
        self.model = model
        self.extract_layers = sorted(extract_layers)  # 确保层索引是排序的
        self.features = {}
        
        # 注册钩子函数来提取中间层特征
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """注册钩子函数来提取指定层的特征"""
        def get_activation(name):
            def hook(model, input, output):
                # 处理不同类型的输出
                if hasattr(output, 'last_hidden_state'):
                    self.features[name] = output.last_hidden_state
                elif isinstance(output, tuple):
                    self.features[name] = output[0]  # 通常第一个元素是hidden states
                else:
                    self.features[name] = output
            return hook
        
        # 为指定层注册钩子
        if hasattr(self.model, 'layers') or hasattr(self.model, 'transformer'):
            # 处理transformer架构
            layers = getattr(self.model, 'layers', None) or getattr(self.model, 'transformer').layers
            for layer_idx in self.extract_layers:
                if layer_idx < len(layers):
                    hook = layers[layer_idx].register_forward_hook(
                        get_activation(f'layer_{layer_idx}')
                    )
                    self.hooks.append(hook)
    
    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播并提取多层特征"""
        self.features.clear()
        
        # 执行前向传播
        if hasattr(self.model, '__call__'):
            output = self.model(x, return_dict=True)
        else:
            output = self.model.forward(x, return_dict=True)
        
        return output, self.features.copy()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class ProbNet(nn.Module):
    """
    增强的ProbNet：实现早期特征引导晚期特征的机制
    """
    def __init__(self, 
                 model_id=LLAMA3_MODEL_ID_8B_INSTRUCT,
                 extract_layers: Optional[List[int]] = None,
                 hidden_dim: int = 4096,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # 构建基础LLM模型
        self.llm, self.tokenizer, self.processor = build_transformers(model_id)
        
        # 如果模型构建失败，创建一个占位符
        if self.llm is None:
            print(f"Warning: Failed to load model {model_id}, using placeholder")
            self.llm = nn.Linear(hidden_dim, hidden_dim)  # 占位符
        
        # 确定要提取特征的层
        if extract_layers is None:
            # 默认提取早期、中期和晚期层
            total_layers = self._get_total_layers()
            extract_layers = [
                total_layers // 4,      # 早期层 (1/4)
                total_layers // 2,      # 中期层 (1/2)
                total_layers * 3 // 4   # 晚期层 (3/4)
            ]
        
        self.extract_layers = extract_layers
        
        # 多层特征提取器
        self.feature_extractor = MultiLayerFeatureExtractor(self.llm, extract_layers)
        
        # 特征引导模块列表
        self.guidance_modules = nn.ModuleList([
            FeatureGuidanceModule(hidden_dim, num_heads, dropout)
            for _ in range(len(extract_layers) - 1)  # 每相邻两层之间一个引导模块
        ])
        
        # 最终特征融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(extract_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def _get_total_layers(self) -> int:
        """获取模型的总层数"""
        if hasattr(self.llm, 'config') and hasattr(self.llm.config, 'num_hidden_layers'):
            return self.llm.config.num_hidden_layers
        elif hasattr(self.llm, 'layers'):
            return len(self.llm.layers)
        elif hasattr(self.llm, 'transformer') and hasattr(self.llm.transformer, 'layers'):
            return len(self.llm.transformer.layers)
        else:
            return 12  # 默认值
    
    def generate(self, text: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        生成文本响应，类似于调用llama
        
        Args:
            text: 输入文本
            max_length: 最大生成长度
            temperature: 生成温度
        
        Returns:
            生成的文本响应
        """
        if self.tokenizer is None:
            return "Error: Tokenizer not available"
        
        try:
            # 对输入文本进行编码
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids']
            
            # 执行特征引导的前向传播
            with torch.no_grad():
                results = self.forward_with_guidance(input_ids)
                
                # 如果有引导后的输出，使用它；否则使用原始输出
                if 'output' in results and results['output'] is not None:
                    # 这里需要将特征转换回logits进行生成
                    # 由于我们修改了特征，需要通过LLM的输出层来生成文本
                    if hasattr(self.llm, 'lm_head'):
                        logits = self.llm.lm_head(results['output'])
                    elif hasattr(self.llm, 'output'):
                        logits = self.llm.output(results['output'])
                    else:
                        # 如果找不到输出层，使用原始输出
                        logits = results['original_output'].logits if hasattr(results['original_output'], 'logits') else results['original_output']
                else:
                    # 使用原始LLM生成
                    output = self.llm.generate(
                        input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    # 移除输入部分，只返回生成的部分
                    input_text_length = len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
                    return generated_text[input_text_length:].strip()
                
                # 使用修改后的logits进行采样生成
                if isinstance(logits, torch.Tensor):
                    # 应用温度
                    logits = logits / temperature
                    
                    # 获取最后一个token的logits
                    next_token_logits = logits[0, -1, :]
                    
                    # 采样下一个token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # 简单的贪心解码（这里可以扩展为更复杂的生成策略）
                    generated_ids = [next_token.item()]
                    
                    # 解码生成的token
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    return generated_text
                else:
                    return "Error: Unable to generate text from guided features"
                    
        except Exception as e:
            return f"Error during generation: {str(e)}"
    
    def __call__(self, text: str, **kwargs) -> str:
        """
        使ProbNet可以像函数一样调用，类似于llama
        
        Args:
            text: 输入文本
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本响应
        """
        return self.generate(text, **kwargs)
    
    def forward_with_guidance(self, x) -> Dict[str, torch.Tensor]:
        """
        前向传播：实现早期特征引导晚期特征的机制
        
        Returns:
            Dict包含:
            - 'output': 最终输出
            - 'guided_features': 引导后的特征列表
            - 'attention_weights': 注意力权重（如果需要可视化）
        """
        # 提取多层特征
        final_output, layer_features = self.feature_extractor(x)
        
        # 按层索引排序特征
        sorted_features = []
        for layer_idx in self.extract_layers:
            layer_key = f'layer_{layer_idx}'
            if layer_key in layer_features:
                sorted_features.append(layer_features[layer_key])
            else:
                # 如果某层特征缺失，使用最终输出作为占位符
                if hasattr(final_output, 'last_hidden_state'):
                    sorted_features.append(final_output.last_hidden_state)
                else:
                    sorted_features.append(final_output)
        
        # 如果没有提取到足够的特征，直接返回原始输出
        if len(sorted_features) < 2:
            return {
                'output': final_output,
                'guided_features': sorted_features,
                'attention_weights': None
            }
        
        # 使用早期特征引导晚期特征
        guided_features = [sorted_features[0]]  # 第一层特征保持不变
        
        for i in range(len(sorted_features) - 1):
            early_feat = sorted_features[i]      # 早期特征作为查询
            late_feat = sorted_features[i + 1]   # 晚期特征作为键值
            
            # 应用特征引导
            guided_feat = self.guidance_modules[i](early_feat, late_feat)
            guided_features.append(guided_feat)
        
        # 融合所有引导后的特征
        if len(guided_features) > 1:
            # 确保所有特征具有相同的序列长度
            min_seq_len = min(feat.size(1) for feat in guided_features)
            aligned_features = [feat[:, :min_seq_len, :] for feat in guided_features]
            
            # 拼接并融合
            concatenated_features = torch.cat(aligned_features, dim=-1)
            fused_output = self.final_fusion(concatenated_features)
        else:
            fused_output = guided_features[0]
        
        return {
            'output': fused_output,
            'guided_features': guided_features,
            'original_output': final_output,
            'layer_features': layer_features
        }
    
    def forward(self, x):
        """
        保持原有的forward接口兼容性
        """
        return self.forward_with_guidance(x)
    
    def __del__(self):
        """析构函数：清理钩子"""
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.remove_hooks()

if __name__ == '__main__':
    prob_net = ProbNet()
    from pprint import pp
    pp(prob_net)
    x = torch.randn(1, 1024, 768)
    output = prob_net(x)
    pp(output)
