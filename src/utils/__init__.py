from .transform import (
    get_transforms,
    build_image_transforms,
    VisionTransformersBuilder,
    get_transform_template,
    template_classification_train,
    template_classification_eval,
    template_clip,
    template_grayscale_medical,
    template_inference,
    template_segmentation_shared,
    template_monai_3d,
    IMAGE_1K_MEAN,
    IMAGE_1K_STD,
    CLIP_MEAN,
    CLIP_STD,
    MONAI_AVAILABLE,
)
from .criterion import CombineCriterion, DiceLoss, Loss, KLLoss
from .early_stopping import EarlyStopping
from .image_utils import ImageDrawer
# 移除循环导入 - 这些类应该从 src.recorder 直接导入
# from ..recorder.metric_recorder import ScoreAggregator, MetricRecorder
from .typed import (
    to_path,
    to_pil_image,
    FilePath,
    ImageInstance,
    FLOAT,
    ClassLabel,
    ClassLabelsList,
    ClassLabelOneScoreDict,
    ClassLabelManyScoreDict,
    ClassMetricManyScoreDict,
    ClassMetricOneScoreDict,
    MetricLabel,
    MetricAfterDict,
    MetricClassManyScoreDict,
    MetricClassOneScoreDict,
    MetricLabelArgmaxDict,
    MetricLabelArgminDict,
    MetricLabelManyScoreDict,
    MetricLabelOneScoreDict,
    MetricLabelsList,
)
from .timer import Timer
from .annotation import time_cost, deprecated, buildin, timer, singleton
from .criterion import dice_loss, kl_divergence_loss, get_criterion
from .see_cam import ImageHeatMapGenerator
from .training_stop import TrainingStopManager
from .progress_bar import format_progress_desc, ProgressBar
from .util import (
    prepare_logger,
    set_seed,
    get_train_tools,
    get_train_criterion,
    save_model,
    load_model,
    load_model_ext,
    save_numpy_data, load_numpy_data,
    model_info,
    model_flops,
    print_model_info_block,
    freeze_layers,
    str2dtype,
    run_async_task,
    reset_peak_memory_stats,
    log_memory_cost,
)
from .postprocess import select_postprocess_fn, get_predict_postprocess_fn
from .download import download_file, download_with_retry, download_multiple

# 延迟导入 painter 相关类以避免循环依赖
def __getattr__(name):
    if name in ['Plot', 'plt', 'CMAP', 'LINE_STYLE', 'THEME', 'Font', 'CmapPresets', 'ThemePresets']:
        from ..visualizer.painter import Plot, plt, CMAP, LINE_STYLE, THEME, Font, CmapPresets, ThemePresets
        globals().update({
            'Plot': Plot,
            'plt': plt,
            'CMAP': CMAP,
            'LINE_STYLE': LINE_STYLE,
            'THEME': THEME,
            'Font': Font,
            'CmapPresets': CmapPresets,
            'ThemePresets': ThemePresets,
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
