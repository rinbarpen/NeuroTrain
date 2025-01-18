
def get_model(model_name: str, config: dict):
    match model_name:
        case 'unet_neck':
            from models.like.unet_neck import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model
        case 'unet':
            from models.sample.unet import UNet
            model = UNet(config['n_channels'], config['n_classes'], bilinear=False)
            return model

    return None
