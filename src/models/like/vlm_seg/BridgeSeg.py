import torch
from torch import nn
import torch.nn.functional as F

from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel
from peft import get_peft_model, LoraConfig
from einops import rearrange, einsum
# from segment_anything import build_sam_vit_h
from transformers import SamModel, SamProcessor
from transformers.models.sam.modeling_sam import SamAttention

from .TopKSelector import TopKSelector
from .LLinear import LLinear

SamAttention.dropout_p = 0.0

class BridgeSeg(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

        self.device = c.get('device', 'cuda')

        self.processor = LlavaProcessor.from_pretrained(c['llm_model_id'], 
            padding_side='right',
            trust_remote_code=True, cache_dir=c['cache_dir']['processor'], use_fast=False)

        self.llm = LlavaForConditionalGeneration.from_pretrained(c['llm_model_id'], 
            trust_remote_code=True, cache_dir=c['cache_dir']['llm'], 
            device_map='auto', torch_dtype=c.get('dtype', torch.float16))

        llm_hidden_dim = self.llm.config.text_config.hidden_size

        self.vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', 
            torch_dtype=c.get('dtype', torch.float16))

        self.sam = SamModel.from_pretrained(c['llm_model_id'], 
            trust_remote_code=True, cache_dir=c['cache_dir']['sam'], 
            device_map='auto', torch_dtype=c.get('dtype', torch.float16))
        seg_hidden_dim = self.sam.config.vision_config.hidden_size

        # self.seg_master = build_sam_vit_h(checkpoint=c['cache_dir']['sam'])
        # seg_hidden_dim = self.seg_master.image_encoder.hidden_dim
        self.adapter = nn.Sequential(
            nn.Linear(seg_hidden_dim, seg_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(seg_hidden_dim // 4, seg_hidden_dim),
            nn.Dropout(0.0),
        )

        self.selector = TopKSelector(k=c.get('k', 3), dim=llm_hidden_dim)
        self.ls_linear = LLinear(llm_hidden_dim, seg_hidden_dim)

        if c.get('extra_tokens', None) is not None:
            self.add_new_tokens(c['extra_tokens'])
        
        self.init_modules()

    def add_new_tokens(self, new_tokens):
        original_vocab_size = len(self.processor.tokenizer)
        self.processor.tokenizer.add_tokens(new_tokens)
        new_vocab_size = len(self.processor.tokenizer)
        self.llm.resize_token_embeddings(new_vocab_size)
        self.llm.config.vocab_size = new_vocab_size

        if hasattr(self.llm, 'get_input_embeddings'):
            input_embeddings = self.llm.get_input_embeddings()
            with torch.no_grad():
                # 用现有embedding的均值初始化新的embedding
                mean_embedding = input_embeddings.weight[:original_vocab_size].mean(dim=0)
                input_embeddings.weight[original_vocab_size:] = mean_embedding.expand(
                    new_vocab_size - original_vocab_size, -1
                )

    def init_modules(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.sam.parameters():
            param.requires_grad = False
        
        # 对LLM应用LoRA
        if self.c.get('train_llm', False):
            self.llm.lm_head.requires_grad = True
            llm_lora_config = LoraConfig(
                r=self.c.get('lora_r', 8),
                lora_alpha=self.c.get('lora_alpha', 16),
                target_modules=self.c.get('lora_target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                lora_dropout=self.c.get('lora_dropout', 0.1),
                bias="all",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, llm_lora_config)
        
        # 对MaskDecoder应用LoRA
        if self.c.get('train_sam_mask_decoder', False):
            if self.c.get('train_sam_lora', False):
                # 使用LoRA微调MaskDecoder
                mask_decoder_lora_config = LoraConfig(
                    r=self.c.get('mask_decoder_lora_r', 16),
                    lora_alpha=self.c.get('mask_decoder_lora_alpha', 32),
                    target_modules=self.c.get('mask_decoder_target_modules', 
                                        ["transformer.layers.0.self_attn.q_proj", 
                                         "transformer.layers.0.self_attn.k_proj",
                                         "transformer.layers.0.self_attn.v_proj", 
                                         "transformer.layers.0.self_attn.out_proj",
                                         "transformer.layers.1.self_attn.q_proj", 
                                         "transformer.layers.1.self_attn.k_proj",
                                         "transformer.layers.1.self_attn.v_proj", 
                                         "transformer.layers.1.self_attn.out_proj",
                                         "transformer.layers.0.mlp.lin1", 
                                         "transformer.layers.0.mlp.lin2",
                                         "transformer.layers.1.mlp.lin1", 
                                         "transformer.layers.1.mlp.lin2",
                                         "output_upscaling.0", 
                                         "output_upscaling.2",
                                         "output_hypernetworks_mlps.0.layers.0", 
                                         "output_hypernetworks_mlps.0.layers.2",
                                         "iou_prediction_head.layers.0", 
                                         "iou_prediction_head.layers.2"]),
                    lora_dropout=self.c.get('mask_decoder_lora_dropout', 0.1),
                    bias="all",
                    task_type="FEATURE_EXTRACTION",
                )
                self.sam.mask_decoder = get_peft_model(self.sam.mask_decoder, mask_decoder_lora_config)
            else:
                # 全参数微调MaskDecoder
                for name, param in self.sam.mask_decoder.named_parameters():
                    param.requires_grad = True
        
        for param in self.adapter.parameters():
            param.requires_grad = True
        for param in self.selector.parameters():
            param.requires_grad = True
        for param in self.ls_linear.parameters():
            param.requires_grad = True
        
        self.adapter.train()
        self.selector.train()
        self.ls_linear.train()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor, pixel_values: torch.FloatTensor, resize_list: list[tuple[int, int]], original_size_list: list[tuple[int, int]], masks_list: torch.FloatTensor):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        image_embeddings = self.get_image_embeddings(pixel_values)

        outputs = self.llm(input_ids, attention_mask=attention_mask,  return_hidden_states=True)
        llm_loss = outputs.loss
        output_hidden_states = outputs.hidden_states
        llm_hidden_states = self.select_hidden_states(output_hidden_states, input_ids, self.processor.tokenizer.convert_tokens_to_ids('<SEG>'))

        llm_hidden_states = self.selector(llm_hidden_states)
        text_mask_embeddings = self.ls_linear(llm_hidden_states)
        low_res_masks, iou_predictions = self.fuse(text_mask_embeddings, image_embeddings)

        masks = torch.stack([
            self.sam.postprocess_masks(
                mask,
                input_size=resize,
                original_size=original_size,
            ) for mask, resize, original_size in zip(low_res_masks, resize_list, original_size_list)
        ], dim=0)

        loss = self.calc_loss({
            'pred_masks': masks,
            'llm': llm_loss,
        }, {
            'gt_masks': masks_list,
        })

        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks,
            'loss': loss,
        }

    def get_image_embeddings(self, pixel_values: torch.FloatTensor):
        image_embeddings = self.sam.vision_encoder(pixel_values)
        return image_embeddings

    def select_hidden_states(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, token_id: int):
        token_index = (input_ids[:, 1:] == token_id)
        token_index = F.pad(token_index, (255, 1))

        hidden_states = hidden_states[torch.any(token_index, dim=1)]
        return hidden_states

    def fuse(self, text_mask_embeddings: torch.FloatTensor, image_embeddings: torch.FloatTensor, **kwargs):
        text_mask_embeddings = rearrange(text_mask_embeddings, 'b k s d -> b (k s) d')

        image_embeddings = self.adapter(image_embeddings)

        # (B, N, D)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=kwargs.get('points', None),
            boxes=kwargs.get('boxes', None),
            masks=kwargs.get('masks', None),
        )
        if text_mask_embeddings is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_mask_embeddings], dim=1)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True, # Can directly multimask output in many SEGs?
        )
        return low_res_masks, iou_predictions
