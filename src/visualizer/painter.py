import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
from pathlib import Path
from sklearn import metrics
from PIL import Image

import toml

from src.utils.annotation import buildin
from src.utils.typed import (
    ClassLabelManyScoreDict,
    ClassLabelOneScoreDict,
    MetricLabelManyScoreDict,
    MetricLabelOneScoreDict,
    MetricClassManyScoreDict,
    MetricClassOneScoreDict,
    ClassMetricManyScoreDict,
    ClassMetricOneScoreDict,
    MetricAfterDict,
    MetricLabelArgmaxDict,
    MetricLabelArgminDict,
    FilePath,
    ImageInstance,
    FLOAT
)

# x, y, shape-like ['-o', '-D'] | font, color, label
# labelsize for tick_params
# linewidth linestyle for grid

# 定义colormap的Literal类型
CMAP = Literal[
    # 顺序色彩映射
    "viridis", "plasma", "inferno", "magma", "cividis",
    # 发散色彩映射
    "coolwarm", "bwr", "seismic", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
    # 定性色彩映射
    "tab10", "tab20", "Set1", "Set2", "Set3", "Pastel1", "Pastel2", "Accent",
    # 单色色彩映射
    "Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "Oranges", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd",
    # 其他常用
    "hot", "cool", "spring", "summer", "autumn", "winter", "bone", "copper",
    "pink", "gray", "binary", "gist_gray", "gist_yarg", "jet", "rainbow", "hsv",
    # 科学可视化常用
    "turbo", "twilight", "twilight_shifted", "terrain", "ocean", "gist_earth",
    # 感知均匀色彩映射
    "cubehelix", "gnuplot", "gnuplot2", "CMRmap", "brg", "gist_rainbow",
    # 反转版本（添加_r后缀的常用版本）
    "viridis_r", "plasma_r", "inferno_r", "magma_r", "Blues_r", "Reds_r", "Greens_r",
    "coolwarm_r", "RdBu_r", "RdYlBu_r", "Spectral_r", "hot_r", "cool_r", "gray_r"
]
# 定义theme的Literal类型
THEME = Literal[
    # matplotlib内置样式
    "default", "classic", "seaborn", "ggplot", "bmh", "fivethirtyeight",
    "grayscale", "dark_background", "tableau-colorblind10",
    # seaborn样式
    "seaborn-v0_8", "seaborn-v0_8-bright", "seaborn-v0_8-colorblind", 
    "seaborn-v0_8-dark", "seaborn-v0_8-dark-palette", "seaborn-v0_8-darkgrid",
    "seaborn-v0_8-deep", "seaborn-v0_8-muted", "seaborn-v0_8-notebook",
    "seaborn-v0_8-paper", "seaborn-v0_8-pastel", "seaborn-v0_8-poster",
    "seaborn-v0_8-talk", "seaborn-v0_8-ticks", "seaborn-v0_8-white",
    "seaborn-v0_8-whitegrid",
    # 自定义科学主题
    "scientific", "medical", "presentation", "publication", "minimal",
    "high_contrast", "colorblind_friendly", "monochrome"
]

# 定义轴的Literal类型
AXIS = Literal["x", "y", "xy"]

# 定义标记的Literal类型
MARKER = Literal[
    ".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "d", "D", "|", "_"
]

# 定义线型的Literal类型
LINE_STYLE = Literal["-", "--", "-.", ":", "", " "]

from .presets import ThemePresets, CmapPresets

class BasePainter:
    def __init__(self, pa):
        self.pa = pa

    def complete(self):
        return self.pa

class Font:
    _dict = {"family": "Times New Roman", "weight": "normal", "size": 16}

    def family(self, x: str):
        self._dict["family"] = x
        return self

    def weight(self, x: str):
        self._dict["weight"] = x
        return self

    def size(self, x: int):
        self._dict["size"] = x
        return self

    def arg(self, name: str, value):
        self._dict[name] = value
        return self

    def build(self):
        return self._dict

    @staticmethod
    def from_fontdict(fontdict: dict):
        font = Font()
        font._dict = fontdict
        return font
    
    # 预设字体配置
    @staticmethod
    def default():
        """默认字体：Times New Roman, 16pt"""
        return Font()
    
    @staticmethod
    def title_font():
        """标题字体：Times New Roman, Bold, 18pt"""
        return Font().family("Times New Roman").weight("bold").size(18)
    
    @staticmethod
    def label_font():
        """标签字体：Times New Roman, Normal, 14pt"""
        return Font().family("Times New Roman").weight("normal").size(14)
    
    @staticmethod
    def small_font():
        """小字体：Times New Roman, Normal, 12pt"""
        return Font().family("Times New Roman").weight("normal").size(12)
    
    @staticmethod
    def large_font():
        """大字体：Times New Roman, Bold, 20pt"""
        return Font().family("Times New Roman").weight("bold").size(20)
    
    @staticmethod
    def serif_font():
        """衬线字体：serif, Normal, 16pt"""
        return Font().family("serif").weight("normal").size(16)
    
    @staticmethod
    def sans_serif_font():
        """无衬线字体：sans-serif, Normal, 16pt"""
        return Font().family("sans-serif").weight("normal").size(16)
    
    @staticmethod
    def monospace_font():
        """等宽字体：monospace, Normal, 14pt"""
        return Font().family("monospace").weight("normal").size(14)
    
    @staticmethod
    def arial_font():
        """Arial字体：Arial, Normal, 16pt"""
        return Font().family("Arial").weight("normal").size(16)
    
    @staticmethod
    def helvetica_font():
        """Helvetica字体：Helvetica, Normal, 16pt"""
        return Font().family("Helvetica").weight("normal").size(16)
    
    @staticmethod
    def calibri_font():
        """Calibri字体：Calibri, Normal, 16pt"""
        return Font().family("Calibri").weight("normal").size(16)
    
    @staticmethod
    def scientific_font():
        """科学论文常用字体：Computer Modern, Normal, 16pt"""
        return Font().family("Computer Modern").weight("normal").size(16)
    
    @staticmethod
    def presentation_font():
        """演示文稿字体：Arial, Bold, 18pt"""
        return Font().family("Arial").weight("bold").size(18)

class Bar(BasePainter):
    def __init__(self, ax, bars):
        super().__init__(ax)  # 调用父类构造函数
        self._bars = bars

    def with_patterns(self, patterns=["/", "\\", "|", "-"]):
        for bar, pattern in zip(self._bars, patterns):
            bar.set_hatch(pattern)  # 修复：set_vatch -> set_hatch
        return self

    def with_text(self, text: str, font: Font = Font()):
        for bar in self._bars:
            height = bar.get_height()
            self.pa.text(
                bar.get_x() + bar.get_width() / 2,  # 修复：位置计算错误
                height + 0.01,  # 修复：位置偏移
                text,
                **font.build(),
                ha="center",
                va="bottom",  # 修复：va="center" -> va="bottom"
            )
        return self

class BarH(BasePainter):
    def __init__(self, ax, bars):
        super().__init__(ax)  # 调用父类构造函数
        self._bars = bars

    def with_patterns(self, patterns=["/", "\\", "|", "-"]):
        for bar, pattern in zip(self._bars, patterns):
            bar.set_hatch(pattern)  # 设置填充模式
        return self

    def with_text(self, text: str, font: Font = Font()):
        for bar in self._bars:
            width = bar.get_width()
            self.pa.text(
                width + 0.01,  # 修复：位置偏移
                bar.get_y() + bar.get_height() / 2,
                text,
                **font.build(),
                ha="left",
                va="center",
            )
        return self

class Line(BasePainter):
    # '.'：点
    # ','：像素
    # 'o'：圆圈
    # 'v'：倒三角形
    # '^'：正三角形
    # '<'：左三角形
    # '>'：右三角形
    # 's'：正方形
    # 'p'：五边形
    # '*'：星号
    # 'h'：六边形1
    # 'H'：六边形2
    # '+'：加号
    # 'x'：叉号
    # 'd'：菱形
    # 'D'：粗菱形
    # '|'：垂直线
    # '_'：水平线
    MarkerLiteral = MARKER
    LineLiteral = LINE_STYLE

    def __init__(self, ax, x, y):
        super().__init__(ax)  # 调用父类构造函数
        self._x = x
        self._y = y

    def with_pattern(self, marker: MarkerLiteral, line: LineLiteral, **kwargs):
        self.pa.plot(self._x, self._y, marker + line, **kwargs)
        return self

class Scatter(BasePainter):
    def __init__(self, ax, x, y):
        super().__init__(ax)  # 调用父类构造函数
        self._x = x
        self._y = y

    def with_color(self, c=None, **kwargs):
        """设置散点图颜色"""
        self.pa.scatter(self._x, self._y, c=c, **kwargs)
        return self
    
    def with_size(self, s=None, **kwargs):
        """设置散点图大小"""
        self.pa.scatter(self._x, self._y, s=s, **kwargs)
        return self
    
    def with_marker(self, marker='o', **kwargs):
        """设置散点图标记"""
        self.pa.scatter(self._x, self._y, marker=marker, **kwargs)
        return self

class Subplot:
    def __init__(self, ax, parent):
        self._ax = ax
        self._parent = parent

    def plot(self, x, y, *args, **kwargs):
        self._ax.plot(x, y, *args, **kwargs)
        return self
        # return Line(self._ax, x, y)

    def bar(self, x, height, width=0.35, *args, **kwargs):
        self._ax.bar(x, height, width, *args, **kwargs)
        return self
        # bars = self._ax.bar(x, height, width, *args, **kwargs)
        # return Bar(self._ax, bars)

    def barh(self, y, width, height=0.35, *args, **kwargs):
        self._ax.barh(y, width, height, *args, **kwargs)
        return self
        # barhs = self._ax.barh(y, width, height, *args, **kwargs)
        # return BarH(self._ax, barhs)

    def scatter(self, x, y, *args, **kwargs):
        self._ax.scatter(x, y, *args, **kwargs)
        return self
        # return Scatter(self._ax, x, y)

    def hist(self, x, bins=None, *args, **kwargs):
        self._ax.hist(x, bins=bins, *args, **kwargs)
        return self
        # return Hist(self._ax, x, bins)

    def grid(self, visible: bool | None = None, **kwargs):
        self._ax.grid(visible, **kwargs)
        return self

    def colorbar(self, **kwargs):
        self._ax.colorbar(**kwargs)
        return self

    # prop=Font().build()
    def legend(self, **kwargs):
        self._ax.legend(**kwargs)
        return self

    def figsize(self, figsize: tuple[float, float]):
        fig = self._ax.get_figure()
        fig.set_size_inches(figsize)
        return self

    def xlabel(self, xlabel: str, font: Font = None, *args, **kwargs):
        """设置x轴标签
        
        Args:
            xlabel: 标签文本
            font: 字体设置，如果为None则使用默认字体
            *args, **kwargs: 其他matplotlib参数
        """
        if font is not None:
            kwargs.update(font.build())
        self._ax.set_xlabel(xlabel, *args, **kwargs)
        return self

    def ylabel(self, ylabel: str, font: Font = None, *args, **kwargs):
        """设置y轴标签
        
        Args:
            ylabel: 标签文本
            font: 字体设置，如果为None则使用默认字体
            *args, **kwargs: 其他matplotlib参数
        """
        if font is not None:
            kwargs.update(font.build())
        self._ax.set_ylabel(ylabel, *args, **kwargs)
        return self

    def label(self, axis: AXIS, label: str, font: Font = None, *args, **kwargs):
        """设置轴标签
        
        Args:
            axis: 轴类型 ('x', 'y', 'xy')
            label: 标签文本
            font: 字体设置，如果为None则使用默认字体
            *args, **kwargs: 其他matplotlib参数
        """
        if axis == "x":
            self.xlabel(label, font, *args, **kwargs)
        elif axis == "y":
            self.ylabel(label, font, *args, **kwargs)
        else:
            self.xlabel(label, font, *args, **kwargs)
            self.ylabel(label, font, *args, **kwargs)
        return self
    
    def lim(self, axis: AXIS, start: float, end: float):
        if axis == "x":
            self._ax.set_xlim(start, end)
        elif axis == "y":
            self._ax.set_ylim(start, end)
        else:
            self._ax.set_xlim(start, end)
            self._ax.set_ylim(start, end)
        return self
    
    def xlim(self, start: float, end: float):
        self._ax.set_xlim(start, end)
        return self
    
    def ylim(self, start: float, end: float):
        self._ax.set_ylim(start, end)
        return self
    
    def axis(self, axis: AXIS, visible: bool = False):
        if axis == "x":
            self._ax.xaxis.set_visible(visible)
        elif axis == "y":
            self._ax.yaxis.set_visible(visible)
        else:
            self._ax.xaxis.set_visible(visible)
            self._ax.yaxis.set_visible(visible)
        return self

    def title(self, title: str, font: Font = None, **kwargs):
        """设置子图标题
        
        Args:
            title: 标题文本
            font: 字体设置，如果为None则使用默认字体
            **kwargs: 其他matplotlib参数
        """
        if font is not None:
            kwargs.update(font.build())
        self._ax.set_title(title, **kwargs)
        return self

    def complete(self):
        return self._parent

    @buildin(desc="loss by epoch")
    def epoch_loss(
        self,
        epoch: int,
        num_epochs: int,
        losses: np.ndarray,
        label: str = "Loss",
        title="Epoch-Loss",
    ):
        # 确保epochs数组长度与losses数组长度匹配
        actual_epochs = len(losses)
        epochs = np.arange(1, actual_epochs + 1, dtype=np.int32)
        self._ax.plot(epochs, losses, label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        self._ax.grid(True, alpha=0.3)
        return self

    @buildin(desc="loss by step")
    def step_loss(
        self,
        step_losses: np.ndarray,
        label: str = "Loss",
        title: str = "Step-Loss",
    ):
        """绘制step级别的损失图"""
        steps = np.arange(1, len(step_losses) + 1, dtype=np.int32)
        self._ax.plot(steps, step_losses, label=label)
        
        self._ax.set_title(title)
        self._ax.set_xlabel("Step")
        self._ax.set_ylabel("Loss")
        self._ax.legend()
        self._ax.grid(True, alpha=0.3)
        return self

    @buildin(desc="learning rate by step")
    def step_lr(
        self,
        step_lrs: np.ndarray,
        label: str = "Learning Rate",
        title: str = "Step-LR",
    ):
        """绘制step级别的学习率图"""
        steps = np.arange(1, len(step_lrs) + 1, dtype=np.int32)
        self._ax.plot(steps, step_lrs, label=label)
        
        self._ax.set_title(title)
        self._ax.set_xlabel("Step")
        self._ax.set_ylabel("Learning Rate")
        self._ax.legend()
        self._ax.grid(True, alpha=0.3)
        return self

    @buildin(desc="learning rate by epoch")
    def epoch_lr(
        self,
        epoch: int,
        num_epochs: int,
        epoch_lrs: np.ndarray,
        label: str = "Learning Rate",
        title: str = "Epoch-LR",
    ):
        """绘制epoch级别的学习率图"""
        epochs = np.arange(1, len(epoch_lrs) + 1, dtype=np.int32)
        self._ax.plot(epochs, epoch_lrs, label=label)
        
        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Learning Rate")
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        self._ax.grid(True, alpha=0.3)
        return self

    @buildin(desc="loss by epoch with tasks")
    def many_epoch_loss(
        self,
        epoch: int,
        num_epochs: int,
        losses: list[np.ndarray],
        labels: list[str] = ["Loss"],
        title="Epoch-Loss",
    ):
        # 使用第一个损失数组的长度来确定epochs数组长度
        if losses and len(losses) > 0:
            actual_epochs = len(losses[0])
            epochs = np.arange(1, actual_epochs + 1, dtype=np.int32)
        else:
            epochs = np.arange(1, epoch + 1, dtype=np.int32)

        for i, label in enumerate(labels):
            if i < len(losses):
                self._ax.plot(epochs[:len(losses[i])], losses[i], label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        return self

    @buildin(desc="metrics by epoch with one class")
    def epoch_metrics(
        self,
        epoch: int,
        num_epochs: int,
        metrics: np.ndarray,
        class_label: str,
        title: str = "Epoch-Label-Metric",
    ):
        # 确保epochs数组长度与metrics数组长度匹配
        actual_epochs = len(metrics)
        epochs = np.arange(1, actual_epochs + 1, dtype=np.int32)
        self._ax.plot(epochs, metrics, label=class_label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Metric Score")
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        return self

    @buildin(desc="metrics by epoch with many classes")
    def many_epoch_metrics_by_class(
        self,
        epoch: int,
        num_epochs: int,
        class_metrics: ClassLabelManyScoreDict,
        class_labels: list[str],
        title: str = "Epoch-Label-Metric",
    ):
        for label in class_labels:
            metrics = class_metrics[label]
            if isinstance(metrics, list) or isinstance(metrics, tuple):
                metrics = np.array(metrics, dtype=np.float64)

            # 确保epochs数组长度与metrics数组长度匹配
            actual_epochs = len(metrics)
            epochs = np.arange(1, actual_epochs + 1, dtype=np.int32)
            self._ax.plot(epochs, metrics, label=label)

        self._ax.set_title(title)
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        return self

    @buildin(desc="metrics by epoch with many classes")
    def many_epoch_metrics(
        self,
        epoch: int,
        num_epochs: int,
        metrics_scores: MetricLabelManyScoreDict,
        metric_labels: list[str],
        title: str = "Epoch-Metrics",
        std_scores: MetricLabelManyScoreDict = None,
    ):
        for metric in metric_labels:
            metrics = metrics_scores[metric]
            if isinstance(metrics, list) or isinstance(metrics, tuple):
                metrics = np.array(metrics, dtype=np.float64)
            
            # 确保epochs数组长度与metrics数组长度匹配
            actual_epochs = len(metrics)
            epochs = np.arange(1, actual_epochs + 1, dtype=np.int32)
            
            # 绘制均值曲线
            self._ax.plot(epochs, metrics, label=metric, marker='o', markersize=4)
            
            # 如果有标准差数据，添加误差带
            if std_scores and metric in std_scores:
                std = std_scores[metric]
                if isinstance(std, (list, tuple)):
                    std = np.array(std, dtype=np.float64)
                if len(std) == len(metrics):
                    self._ax.fill_between(
                        epochs, 
                        metrics - std, 
                        metrics + std, 
                        alpha=0.2
                    )

        self._ax.set_title(title)
        self._ax.set_xlim(1, num_epochs)
        self._ax.legend()
        self._ax.grid(True, alpha=0.3)
        return self

    @buildin(desc="confusion matrix")
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels=None,
        xlabel="True",
        ylabel="Prediction",
        title="Confusion Matrix",
        cmap: CMAP = "Blues",
    ):
        cm = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, xticklabels=labels, yticklabels=labels, cmap=cmap, annot=True, fmt="d"
        )
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.legend()
        return self

    @buildin(desc="metrics")
    def metrics(
        self,
        metric_score: MetricLabelOneScoreDict,
        title: str | None = None,
        patterns=["/", "\\", "|", "-"],
        text=False,
        *,
        height=0.35,
        width=0.35,
        tick_threshold=0.2,
        use_barh=True,
    ):
        metric_labels, scores = list(metric_score.keys()), list(metric_score.values())
        if use_barh:
            bars = self._ax.barh(metric_labels, scores, height)
            self._ax.set_xlim(0, 1)
            self._ax.set_xticks(np.arange(0, 1.1, tick_threshold))
            # self._ax.set_yticklabels(metric_labels)
            # for bar, pattern in zip(bars, patterns):
            #     bar.set_hatch(pattern)
            if text:
                for bar, score in zip(bars, scores):
                    width = bar.get_width()
                    self._ax.text(
                        width + 0.01, 
                        bar.get_y() + bar.get_height() / 2,
                        f"{score:.3f}",
                        ha="left",
                        va="center",
                    )
        else:
            bars = self._ax.bar(metric_labels, scores, width)
            self._ax.set_ylim(0, 1)
            self._ax.set_yticks(np.arange(0, 1.1, tick_threshold))
            # self._ax.set_xticklabels(metric_labels)
            # for bar, pattern in zip(bars, patterns):
            #     bar.set_hatch(pattern)  # 修复：set_vbatch -> set_hatch
            if text:
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    self._ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f"{score:.3f}",
                        ha="center",
                        va="bottom", 
                    )

        if title:
            self._ax.set_title(title)
        return self

    @buildin(desc="many metrics")
    def many_metrics(
        self,
        label_metric_score: ClassMetricOneScoreDict,
        title: str | None = None,
        patterns=["/", "\\", "|", "-"],
        text=False,
        *,
        height=0.35,
        width=0.35,
        tick_threshold=0.2,
        use_barh=True,
        use_label=True,
    ):
        colors = sns.color_palette("husl", len(label_metric_score))
        for color, (class_label, metric_score) in zip(
            colors, label_metric_score.items()
        ):
            metric_labels, scores = list(metric_score.keys()), list(metric_score.values())
            if use_barh:
                bars = self._ax.barh(
                    metric_labels, scores, height, color=color, label=class_label if use_label else None
                )
                self._ax.set_xlim(0, 1)
                self._ax.set_xticks(np.arange(0, 1.1, tick_threshold))
                # self._ax.set_yticklabels(metric_labels)
                # for bar, pattern in zip(bars, patterns):
                #     bar.set_hatch(pattern)
                if text:
                    for bar, score in zip(bars, scores):
                        width = bar.get_width()
                        self._ax.text(
                            width + 0.01,  # 修复：位置偏移
                            bar.get_y() + bar.get_height() / 2,
                            f"{score:.3f}",
                            ha="left",
                            va="center",
                        )
            else:
                bars = self._ax.bar(
                    metric_labels, scores, width, color=color, label=class_label if use_label else None
                )
                self._ax.set_ylim(0, 1)
                self._ax.set_yticks(np.arange(0, 1.1, tick_threshold))
                # self._ax.set_xticklabels(metric_labels)
                # for bar, pattern in zip(bars, patterns):
                #     bar.set_hatch(pattern)  # 修复：set_vbatch -> set_hatch
                if text:
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        self._ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.01,  # 修复：位置偏移
                            f"{score:.3f}",
                            ha="center",
                            va="bottom",  # 修复：va="center" -> va="bottom"
                        )

        if title:
            self._ax.set_title(title)
        if use_label:
            self._ax.legend()
        return self

    def with_autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            self._ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        return self

    @buildin(desc="ROC Curve")
    def roc(
        self,
        y_trues: list[np.ndarray],
        y_preds: list[np.ndarray],
        labels: list[str],
        title="ROC Curve",
    ):
        for y_true, y_pred, label in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            self._ax.plot(fpr, tpr, label=label)
        
        self._ax.set_title(title)
        self._ax.set_xlabel("False Positive Rate")
        self._ax.set_ylabel("True Positive Rate")
        self._ax.legend()
        return self

    @buildin(desc="ROC Curve with AUC")
    def auc(
        self,
        y_trues: list[np.ndarray],
        y_preds: list[np.ndarray],
        labels: list[str],
        title: str = "ROC Curve with AUC",
    ):
        for y_true, y_pred, label in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            self._ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        self._ax.set_title(title)
        self._ax.set_xlabel("False Positive Rate")
        self._ax.set_ylabel("True Positive Rate")
        self._ax.legend(loc="lower right")
        return self

    @buildin(desc="PR Curve")
    def pr_curve(
        self,
        precisions: list[np.ndarray]|np.ndarray,
        recalls: list[np.ndarray]|np.ndarray,
        labels: list[str],
        title: str = "PR Curve",
    ):
        colors = sns.color_palette("husl", len(labels))
        for precision, recall, label, color in zip(precisions, recalls, labels, colors):
            self._ax.plot(recall, precision, label=label, color=color)
        self._ax.set_title(title)
        self._ax.set_xlabel("Recall")
        self._ax.set_ylabel("Precision")
        self._ax.legend()
        return self

    # cmap bug
    @buildin(desc="image show")
    def image(self, image: Image.Image | cv2.Mat, cmap: CMAP | None = None):
        self._ax.axis("off")
        self._ax.imshow(image, cmap=cmap)
        return self
    
    def no_title(self):
        self.title('')
        return self
    
    def no_xylabel(self):
        self.label('xy', '')
        return self
    

    def instance(self):
        return self._ax

class Plot:
    def __init__(self, nrows: int=1, ncols: int=1, figsize: tuple[float, float] = (8, 6)):
        self.nrows = nrows
        self.ncols = ncols
        self._theme = None
        self._fig, self._axs = plt.subplots(nrows, ncols, figsize=figsize)
        self._index = 0

    def subplot(self, index: int = -1) -> Subplot:
        if index < 0:
            index = self._index

        if self.nrows == 1 and self.ncols == 1:
            ax = self._axs
        elif self.nrows == 1:
            if index >= self.ncols:
                index = self.ncols - 1
            ax = self._axs[index]
        elif self.ncols == 1:
            if index >= self.nrows:
                index = self.nrows - 1
            ax = self._axs[index]
        else:
            row, col = index // self.ncols, index % self.ncols
            ax = self._axs[row, col]

        self._index = index + 1
        return Subplot(ax, self)
    
    def theme(self, theme: str):
        """设置主题，支持预设主题、自定义主题和配置文件重载"""
        self._theme = theme
        
        # 应用主题（支持配置文件重载）
        ThemePresets.apply_theme_with_config(theme)
        
        return self
    
    def set_theme_preset(self, category: str, theme_type: str = None) -> 'Plot':
        """设置预设主题类别，支持配置文件重载"""
        theme = ThemePresets.get_theme(category, theme_type)
        return self.theme(theme)
    
    def reload_theme_config(self):
        """重新加载主题配置文件"""
        _config_loader.reload_config()
        return self

    def show(self):
        """显示图形"""
        if self._fig is not None:
            plt.show()
        return self
    
    def save(self, filename: str|Path, **kwargs):
        """保存图形"""
        if self._fig is not None:
            self._fig.savefig(filename, **kwargs)
        return self
    
    def close(self):
        """关闭图形"""
        if self._fig is not None:
            plt.close(self._fig)
        return self
