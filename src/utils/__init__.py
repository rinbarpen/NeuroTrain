from .transform import get_transforms, build_image_transforms, VisionTransformersBuilder
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
    freeze_layers,
    str2dtype,
)
from .postprocess import select_postprocess_fn

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
