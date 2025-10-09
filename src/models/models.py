import torch

def get_model(model_name: str, config: dict):
    # 预先定义默认的torch数据类型
    default_dtype = torch.float16
    
    match model_name.lower():
        case "clip":
            from .llm.clip import CLIP
            model = CLIP(
                model_name=config.get('model_name', 'openai/clip-vit-base-patch32'), 
                cache_dir=config.get('cache_dir', None),
                device=config.get('device', 'cuda'), 
                dtype=config.get('dtype', default_dtype)
            )
            return model
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
