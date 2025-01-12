from argparse import ArgumentParser

from onnxruntime.transformers.huggingface_models import MODELS
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget, SoftmaxOutputTarget, RawScoresOutputTarget, FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

# def get_cam_info(model: str):
#     model = get_model(model) # get target model
#     layer = # get target layer
#     input_tensor = # set input tensor


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='model name')
    args = parser.parse_args()

    # MODEL, LAYER, INPUT_TENSOR = get_cam_info(args.model)
    # with GradCAM(model=MODEL, target_layers=LAYER) as cam:
    #     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #     grayscale_cam = cam(input_tensor=INPUT_TENSOR, targets=targets)
    #     # In this example grayscale_cam has only one image in the batch:
    #     grayscale_cam = grayscale_cam[0, :]
    #     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #     # You can also get the model outputs without having to redo inference
    #     model_outputs = cam.outputs

    # example
    # model = resnet50(pretrained=True)
    # target_layers = [model.layer4[-1]]
    # input_tensor = # Create an input tensor image for your model..
    # # Note: input_tensor can be a batch tensor with several images!
    #
    # # We have to specify the target we want to generate the CAM for.
    # targets = [ClassifierOutputTarget(281)]
    #
    # # Construct the CAM object once, and then re-use it on many images.
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #   # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #   grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #   # In this example grayscale_cam has only one image in the batch:
    #   grayscale_cam = grayscale_cam[0, :]
    #   visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #   # You can also get the model outputs without having to redo inference
    #   model_outputs = cam.outputs
