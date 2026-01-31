import torch

from src.paths import get_pretrained_dir

def get_model(model_name: str, config: dict):
    _cache_dir = config.get("cache_dir") or str(get_pretrained_dir())
    default_dtype = torch.float16
    
    match model_name.lower():
        case "clip":
            from .like.llm.clip import CLIP
            from .like.llm.transformers import MODEL_ID_CATALOG
            
            variant = config.get('variant', 'base').lower()
            model_id = config.get('model_name')
            
            if model_id is None:
                variants = {
                    'base': MODEL_ID_CATALOG['image_encoder']['clip'][0],
                    'base-p16': MODEL_ID_CATALOG['image_encoder']['clip'][1],
                    'large-p14': MODEL_ID_CATALOG['image_encoder']['clip'][2],
                    'medclip': MODEL_ID_CATALOG['biomedical']['medclip'][0],
                    'biomedclip': MODEL_ID_CATALOG['biomedical']['biomedclip'][0],
                }
                model_id = variants.get(variant, variants['base'])

            model = CLIP(
                model_name=model_id, 
                cache_dir=_cache_dir,
                dtype=config.get('dtype', default_dtype)
            )
            return model
        
        case 'dino':
            from .like.llm.clip import CLIP  # CLIP wrapper can often be used for DINO features too if compatible, or use a generic wrapper
            from .like.llm.transformers import MODEL_ID_CATALOG
            
            variant = config.get('variant', 'v1-base').lower()
            model_id = config.get('model_name')
            
            if model_id is None:
                variants = {
                    'v1-base': MODEL_ID_CATALOG['image_encoder']['dino'][0],
                    'v2-base': MODEL_ID_CATALOG['image_encoder']['dinov2'][0],
                    'v2-large': MODEL_ID_CATALOG['image_encoder']['dinov2'][1],
                    'v3-s16': MODEL_ID_CATALOG['image_encoder']['dinov3'][0],
                    'v3-b16': MODEL_ID_CATALOG['image_encoder']['dinov3'][1],
                    'v3-l16': MODEL_ID_CATALOG['image_encoder']['dinov3'][2],
                }
                model_id = variants.get(variant, variants['v1-base'])
            
            # Using CLIP wrapper as a generic transformer wrapper if applicable, 
            # or we might need a specific DINO wrapper if one exists.
            # For now, let's assume the user can use 'huggingface' case for generic HF models.
            # But let's provide a specific dino case if possible.
            from .like.llm.transformers import build_transformers
            model, tokenizer, processor = build_transformers(
                model_id=model_id,
                cache_dir=_cache_dir,
                device=config.get('device', 'cuda'),
                torch_dtype=config.get('dtype', default_dtype)
            )
            return {"model": model, "tokenizer": tokenizer, "processor": processor}

        case 'vlm':
            from .like.llm.transformers import MODEL_ID_CATALOG, build_transformers
            
            variant = config.get('variant', 'llava').lower()
            model_id = config.get('model_name')
            
            if model_id is None:
                variants = {
                    'llava': MODEL_ID_CATALOG['vlm']['llava'][0],
                    'llava-13b': MODEL_ID_CATALOG['vlm']['llava'][1],
                    'qwen2-vl': MODEL_ID_CATALOG['vlm']['qwen2_vl'][0],
                    'phi3.5-vision': MODEL_ID_CATALOG['vlm']['phi_vision'][0],
                    'florence2': MODEL_ID_CATALOG['vlm']['florence2'][0],
                }
                model_id = variants.get(variant, variants['llava'])
            
            model, tokenizer, processor = build_transformers(
                model_id=model_id,
                cache_dir=_cache_dir,
                device=config.get('device', 'auto'),
                torch_dtype=config.get('dtype', default_dtype)
            )
            return {"model": model, "tokenizer": tokenizer, "processor": processor}

        case 'llm':
            from .like.llm.transformers import MODEL_ID_CATALOG, build_transformers
            
            variant = config.get('variant', 'llama3.1-8b').lower()
            model_id = config.get('model_name')
            
            if model_id is None:
                variants = {
                    'llama3-8b': MODEL_ID_CATALOG['llm']['llama3'][0],
                    'llama3.1-8b': MODEL_ID_CATALOG['llm']['llama3.1'][0],
                    'llama3.2-1b': MODEL_ID_CATALOG['llm']['llama3.2'][0],
                    'llama3.2-3b': MODEL_ID_CATALOG['llm']['llama3.2'][2],
                }
                model_id = variants.get(variant, variants['llama3.1-8b'])
            
            model, tokenizer, _ = build_transformers(
                model_id=model_id,
                cache_dir=_cache_dir,
                device=config.get('device', 'auto'),
                torch_dtype=config.get('dtype', default_dtype)
            )
            return {"model": model, "tokenizer": tokenizer}
        case 'unet':
            from .sample.unet import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'simple-net':
            from .sample.simple_net import SimpleNet
            model = SimpleNet()
            return model
        case 'vit':
            from .sample.ViT import (
                ViT, 
                vit_tiny_patch16_224, 
                vit_small_patch16_224,
                vit_base_patch16_224, 
                vit_base_patch32_224,
                vit_large_patch16_224, 
                vit_huge_patch14_224
            )
            
            variant = config.get('variant', 'base').lower()
            n_classes = config.get('n_classes', 1000)
            
            variants = {
                'tiny': vit_tiny_patch16_224,
                'small': vit_small_patch16_224,
                'base': vit_base_patch16_224,
                'base-p16': vit_base_patch16_224,
                'base-p32': vit_base_patch32_224,
                'large': vit_large_patch16_224,
                'huge': vit_huge_patch14_224,
            }
            
            if variant in variants:
                model = variants[variant](num_classes=n_classes, **config.get('kwargs', {}))
            else:
                model = ViT(
                    image_size=config.get('image_size', 224),
                    patch_size=config.get('patch_size', 16),
                    num_classes=n_classes,
                    dim=config.get('dim', 768),
                    depth=config.get('depth', 12),
                    heads=config.get('heads', 12),
                    mlp_dim=config.get('mlp_dim', 3072),
                    **config.get('kwargs', {})
                )
            return model
        case 'vit_grid':
            from .transformer.vit_grid import (
                ViTGrid,
                vit_grid_tiny_patch16_224,
                vit_grid_small_patch16_224,
                vit_grid_base_patch16_224,
                vit_grid_base_patch32_224,
            )
            variant = config.get('variant', 'base').lower()
            n_classes = config.get('n_classes', 1000)
            variants = {
                'tiny': vit_grid_tiny_patch16_224,
                'small': vit_grid_small_patch16_224,
                'base': vit_grid_base_patch16_224,
                'base-p16': vit_grid_base_patch16_224,
                'base-p32': vit_grid_base_patch32_224,
            }
            if variant in variants:
                model = variants[variant](num_classes=n_classes, **config.get('kwargs', {}))
            else:
                model = ViTGrid(
                    image_size=config.get('image_size', 224),
                    patch_size=config.get('patch_size', 16),
                    num_classes=n_classes,
                    dim=config.get('dim', 768),
                    depth=config.get('depth', 12),
                    heads=config.get('heads', 12),
                    mlp_dim=config.get('mlp_dim', 3072),
                    channels=config.get('channels', 3),
                    **config.get('kwargs', {}),
                )
            return model
        case 'swin':
            from .transformer.swin_transformers.swin_transformer import (
                SwinTransformer,
                swin_tiny_patch4_window7_224,
                swin_small_patch4_window7_224,
                swin_base_patch4_window7_224,
                swin_large_patch4_window7_224,
                swin_base_patch4_window12_384,
                swin_large_patch4_window12_384
            )
            variant = config.get('variant', 'tiny').lower()
            n_classes = config.get('n_classes', 1000)
            variants = {
                'tiny': swin_tiny_patch4_window7_224,
                'small': swin_small_patch4_window7_224,
                'base': swin_base_patch4_window7_224,
                'large': swin_large_patch4_window7_224,
                'base-384': swin_base_patch4_window12_384,
                'large-384': swin_large_patch4_window12_384,
            }
            if variant in variants:
                model = variants[variant](num_classes=n_classes, **config.get('kwargs', {}))
            else:
                model = SwinTransformer(num_classes=n_classes, **config.get('kwargs', {}))
            return model
        case 'resnet':
            from .sample.resnet import (
                resnet18, resnet34, resnet50, resnet101
            )
            variant = config.get('variant', '18').lower()
            n_classes = config.get('n_classes', 1000)
            variants = {
                '18': resnet18,
                '34': resnet34,
                '50': resnet50,
                '101': resnet101,
            }
            if variant in variants:
                model = variants[variant](num_classes=n_classes, **config.get('kwargs', {}))
            else:
                model = resnet18(num_classes=n_classes, **config.get('kwargs', {}))
            return model
        case 'vgg':
            from .sample.vgg import (
                vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
            )
            variant = config.get('variant', '16').lower()
            n_classes = config.get('n_classes', 1000)
            variants = {
                '11': vgg11,
                '11_bn': vgg11_bn,
                '13': vgg13,
                '13_bn': vgg13_bn,
                '16': vgg16,
                '16_bn': vgg16_bn,
                '19': vgg19,
                '19_bn': vgg19_bn,
            }
            if variant in variants:
                model = variants[variant](num_classes=n_classes, **config.get('kwargs', {}))
            else:
                model = vgg16(num_classes=n_classes, **config.get('kwargs', {}))
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
                cache_dir=tokenizer_c.get('cache_dir') or str(get_pretrained_dir()),
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
                cache_dir=model_c.get('cache_dir') or str(get_pretrained_dir()),
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
                cache_dir=_cache_dir,
            )
            # 使用包装器以适配训练器的数据格式
            model = EMOERefCOCOModelWrapper(base_model)
            return model

    raise ValueError(f'No supported model: {model_name}')
