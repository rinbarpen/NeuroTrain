
def get_model(model_name: str, config: dict):
    match model_name:
        case 'unet_neck':
            from .like.unet_neck import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'unet':
            from .sample.unet import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'simple-net':
            from .sample.simple_net import SimpleNet
            model = SimpleNet()
            return model
        case 'huggingface' | 'hf':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            tokenizer_c = config['tokenizer']
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_c['model'],
                cache_dir=tokenizer_c['cache_dir'],
                use_fast=tokenizer_c['use_fast'],
            )
            model_c = config['model']
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[model_c.get('torch_dtype', "bfloat16")]
            model = AutoModelForCausalLM.from_pretrained(
                model_c['model'],
                cache_dir=model_c['cache_dir'],
                trust_remote_code=model_c['trust_remote_code'],
                torch_dtype=torch_dtype,
                device_map=model_c['device_map'],
            )
            return {
                "tokenizer": tokenizer,
                "model": model,
            }

    raise ValueError(f'No supported model: {model_name}')
