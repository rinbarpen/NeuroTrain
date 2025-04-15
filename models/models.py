
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

    return None
