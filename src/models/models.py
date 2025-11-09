import torch

from src.constants import PRETRAINED_MODEL_DIR

def get_model(model_name: str, config: dict):
    default_dtype = torch.float16
    
    match model_name.lower():
        case "clip":
            from .llm.clip import CLIP
            model = CLIP(
                model_name=config.get('model_name', 'openai/clip-vit-base-patch32'), 
                cache_dir=config.get('cache_dir', PRETRAINED_MODEL_DIR),
                device=config.get('device', 'cuda'), 
                dtype=config.get('dtype', default_dtype)
            )
            return model
        case 'unet':
            from .sample.unet import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'simple-net':
            from .sample.simple_net import SimpleNet
            model = SimpleNet()
            return model
        case 'torchvision':
            import torchvision.models as models
            import torch.nn as nn
            
            # 获取模型架构名称
            arch = config.get('arch', 'resnet18')
            pretrained = config.get('pretrained', True)
            n_classes = config.get('n_classes', 1000)
            
            # 创建模型
            if hasattr(models, arch):
                if pretrained:
                    # 使用新的weights参数（torchvision >= 0.13）
                    try:
                        weights = getattr(models, f'{arch.upper()}_Weights').DEFAULT if pretrained else None
                        model = getattr(models, arch)(weights=weights)
                    except AttributeError:
                        # 兼容旧版本torchvision
                        model = getattr(models, arch)(pretrained=pretrained)
                else:
                    model = getattr(models, arch)(pretrained=False)
                
                # 修改分类器层以适应目标类别数
                if hasattr(model, 'fc'):  # ResNet, DenseNet等
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, n_classes)
                elif hasattr(model, 'classifier'):  # VGG, AlexNet等
                    if isinstance(model.classifier, nn.Sequential):
                        in_features = model.classifier[-1].in_features
                        model.classifier[-1] = nn.Linear(in_features, n_classes)
                    else:
                        in_features = model.classifier.in_features
                        model.classifier = nn.Linear(in_features, n_classes)
                elif hasattr(model, 'head'):  # Vision Transformer等
                    in_features = model.head.in_features
                    model.head = nn.Linear(in_features, n_classes)
                
                # 如果是灰度图像，修改第一层
                if config.get('n_channels', 3) == 1:
                    if hasattr(model, 'conv1'):  # ResNet等
                        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, 
                                              kernel_size=model.conv1.kernel_size,
                                              stride=model.conv1.stride,
                                              padding=model.conv1.padding,
                                              bias=False)
                    elif hasattr(model, 'features') and hasattr(model.features[0], 'out_channels'):  # VGG等
                        first_conv = model.features[0]
                        model.features[0] = nn.Conv2d(1, first_conv.out_channels,
                                                     kernel_size=first_conv.kernel_size,
                                                     stride=first_conv.stride,
                                                     padding=first_conv.padding)
                
                return model
            else:
                raise ValueError(f'Unsupported torchvision model architecture: {arch}')
                
        case 'timm':
            try:
                import timm
            except ImportError:
                raise ImportError("timm library is required for timm models. Install with: pip install timm")
            
            # 获取模型配置
            model_name_timm = config.get('model_name', 'resnet18')
            pretrained = config.get('pretrained', True)
            n_classes = config.get('n_classes', 1000)
            in_chans = config.get('n_channels', 3)
            
            # 创建模型
            model = timm.create_model(
                model_name_timm,
                pretrained=pretrained,
                num_classes=n_classes,
                in_chans=in_chans
            )
            
            return model
            
        case 'huggingface' | 'hf':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer_c = config['tokenizer']
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_c['model'],
                trust_remote_code=tokenizer_c.get('trust_remote_code', False),
                cache_dir=tokenizer_c.get('cache_dir', PRETRAINED_MODEL_DIR),
                use_fast=tokenizer_c.get('use_fast', False),
            )
            model_c = config['model']
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[model_c.get('torch_dtype', "bfloat16")]
            model = AutoModelForCausalLM.from_pretrained(
                model_c['model'],
                cache_dir=model_c.get('cache_dir', PRETRAINED_MODEL_DIR),
                trust_remote_code=model_c.get('trust_remote_code', False),
                torch_dtype=torch_dtype,
                device_map=model_c.get('device_map', 'auto'),
            )
            return {
                "tokenizer": tokenizer,
                "model": model,
            }
        
        case 'emoe_refcoco' | 'emoe-refcoco':
            from .like.emoe.refcoco_model import EMOE_RefCOCO, EMOERefCOCOModelWrapper
            
            base_model = EMOE_RefCOCO(
                backbone=config.get('backbone', 'vit_base_patch16_224'),
                vit_hidden_dim=config.get('vit_hidden_dim', 768),
                num_heads=config.get('num_heads', 8),
                num_experts=config.get('num_experts', 4),
                expert_hidden_dim=config.get('expert_hidden_dim'),
                k=config.get('k', 2),
                sparse=config.get('sparse', True),
                dropout=config.get('dropout', 0.1),
                text_encoder_name=config.get('text_encoder_name', 'openai/clip-vit-base-patch32'),
                text_encoder_dim=config.get('text_encoder_dim', 512),
                alignment_dim=config.get('alignment_dim', 512),
                temperature=config.get('temperature', 0.07),
                cache_dir=config.get('cache_dir', PRETRAINED_MODEL_DIR),
            )
            # 使用包装器以适配训练器的数据格式
            model = EMOERefCOCOModelWrapper(base_model)
            return model

    raise ValueError(f'No supported model: {model_name}')
