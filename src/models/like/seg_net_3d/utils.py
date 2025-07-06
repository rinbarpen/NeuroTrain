from segment_anything import build_sam_vit_b, build_sam_vit_l, build_sam_vit_h

def get_model_config(model_name: str):
    if model_name == 'sam_vit_h':
        embed_dim = 1280
        build_model_fn = build_sam_vit_h
    elif model_name == 'sam_vit_l':
        embed_dim = 1024
        build_model_fn = build_sam_vit_l
    else:
        embed_dim = 768
        build_model_fn = build_sam_vit_b
    
    return {
        'embed_dim': embed_dim,
        'build_model_fn': build_model_fn,
    }

