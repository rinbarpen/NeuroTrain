import logging
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal
from pathlib import Path
from sklearn import metrics
from PIL import Image

from utils.annotation import buildup
from utils.typed import ClassMetricOneScoreDict

# x, y, shape-like ['-o', '-D'] | font, color, label
# labelsize for tick_params
# linewidth linestyle for grid

AXIS = Literal["x", "y", "xy"]


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


class Subplot:
    def __init__(self, ax, parent):
        self._ax = ax
        self._parent = parent

    def plot(self, x, y, *args, **kwargs):
        self._ax.plot(x, y, *args, **kwargs)
        return self

    def bar(self, x, height, width=0.35, *args, **kwargs):
        self._ax.bar(x, height, width, *args, **kwargs)
        return self

    def barh(self, y, width, height=0.35, *args, **kwargs):
        self._ax.barh(y, width, height, *args, **kwargs)
        return self

    def scatter(self, x, y, *args, **kwargs):
        self._ax.scatter(x, y, *args, **kwargs)
        return self

    def hist(self, x, bins=None, *args, **kwargs):
        self._ax.hist(x, bins=bins, *args, **kwargs)
        return self

    def grid(self, visible: bool | None = None, **kwargs):
        self._ax.grid(visible, **kwargs)
        return self

    # prop=Font().build()
    def legend(self, *args, **kwargs):
        self._ax.set_legend(*args, **kwargs)
        return self

    def figsize(self, figsize: tuple[float, float]):
        self._ax.set_figsize(figsize)
        return self

    def xlabel(self, xlabel: str, *args, **kwargs):
        self._ax.set_xlabel(xlabel, *args, **kwargs)
        return self

    def ylabel(self, ylabel: str, *args, **kwargs):
        self._ax.set_xlabel(ylabel, *args, **kwargs)
        return self

    def label(self, axis: AXIS, label: str, *args, **kwargs):
        if axis == "x":
            self.xlabel(label, *args, **kwargs)
        elif axis == "y":
            self.ylabel(label, *args, **kwargs)
        else:
            self.xlabel(label, *args, **kwargs)
            self.ylabel(label, *args, **kwargs)

    def xticks(self, xticks: np.ndarray):
        self._ax.set_xticks(xticks)
        return self

    def yticks(self, yticks: np.ndarray):
        self._ax.set_yticks(yticks)
        return self

    def ticks(self, axis: AXIS, ticks: np.ndarray):
        if axis == "x":
            self.xticks(ticks)
        elif axis == "y":
            self.yticks(ticks)
        else:
            self.xticks(ticks)
            self.yticks(ticks)

    def tick_params(self, **kwargs):
        self._ax.tick_parmas(**kwargs)
        return self

    def xlim(self, *args):
        self._ax.set_xlim(*args)
        return self

    def ylim(self, *args):
        self._ax.set_ylim(*args)
        return self

    def lim(self, axis: AXIS, *args):
        if axis == "x":
            self.xlim(*args)
        elif axis == "y":
            self.ylim(*args)
        else:
            self.xlim(*args)
            self.ylim(*args)

    def xaxis_visible(self, visible: bool = True):
        self._ax.xaxis.set_visible(visible)

    def yaxis_visible(self, visible: bool = True):
        self._ax.yaxis.set_visible(visible)

    def axis_visible(self, axis: AXIS, visible: bool = True):
        if axis == "x":
            self.xaxis_visible(visible)
        elif axis == "y":
            self.yaxis_visible(visible)
        else:
            self.xaxis_visible(visible)
            self.yaxis_visible(visible)

    def title(self, title: str):
        self._ax.set_title(title)
        return self

    def complete(self):
        return self._parent

    @buildup(desc="loss by epoch")
    def epoch_loss(
        self,
        num_epoch: int,
        losses: list[np.ndarray],
        label: str = "Loss",
        title="Epoch-Loss",
    ):
        if isinstance(losses, list) or isinstance(losses, tuple):
            losses = np.array(losses, dtype=np.float64)

        epoches = np.arange(1, num_epoch + 1, dtype=np.int32)
        self._ax.plot(epoches, losses, label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    @buildup(desc="metrics by epoch with one class")
    def epoch_metrics(
        self,
        num_epoch: int,
        metrics: list[np.ndarray],
        class_label: str,
        title: str = "Epoch-Label-Metric",
    ):
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            metrics = np.array(metrics, dtype=np.float64)

        epochs = np.arange(1, num_epoch + 1, dtype=np.int32)
        self._ax.plot(epochs, metrics, label=class_label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Metric Score")
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    @buildup(desc="metrics by epoch with many classes")
    def many_epoch_metrics(
        self,
        num_epoch: int,
        class_metrics: dict[str, list[np.ndarray]],
        class_labels: list[str],
        title: str = "Epoch-Label-Metric",
    ):
        for label in class_labels:
            metrics = class_metrics[label]
            if isinstance(metrics, list) or isinstance(metrics, tuple):
                metrics = np.array(metrics, dtype=np.float64)

            epochs = np.arange(1, num_epoch + 1, dtype=np.int32)
            self._ax.plot(epochs, metrics, label=label)

        self._ax.set_title(title)
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    @buildup(desc="loss by epoch with tasks")
    def many_epoch_loss(
        self,
        num_epoch: int,
        losses: list[np.ndarray],
        labels: list[str] = ["Loss"],
        title="Epoch-Loss",
    ):
        epoches = np.arange(1, num_epoch + 1, dtype=np.int32)

        for i, label in enumerate(labels):
            self._ax.plot(epoches, losses[i], label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel("Epoch")
        self._ax.set_ylabel("Loss")
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    @buildup(desc="confusion matrix")
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels=None,
        xlabel="True",
        ylabel="Prediction",
        title="Confusion Matrix",
        cmap="Blues",
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

    # scores_map: {'metric_label': {'label'： label_score, ...}, ...}
    @buildup(desc="many metrics")
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
    ):
        colors = sns.color_palette("husl", len(label_metric_score))
        for color, (class_label, metric_score) in zip(
            colors, label_metric_score.items()
        ):
            metric_labels, scores = metric_score.keys(), metric_score.values()
            if use_barh:
                bars = self._ax.barh(
                    metric_labels, scores, height, color=color, label=class_label
                )
                self._ax.set_xlim(0, 1)
                self._ax.set_xticks(np.arange(0, 1.1, tick_threshold))
                # self._ax.set_yticklabels(metric_labels)
                for bar, pattern in zip(bars, patterns):
                    bar.set_hatch(pattern)
                if text:
                    for bar, score in zip(bars, scores):
                        width = bar.get_width()
                        self._ax.text(
                            width + 1,
                            bar.get_y() + bar.get_height() / 2,
                            f"{score:.3f}",
                            ha="left",
                            va="center",
                        )
            else:
                bars = self._ax.bar(
                    metric_labels, scores, width, color=color, label=class_label
                )
                self._ax.set_ylim(0, 1)
                self._ax.set_yticks(np.arange(0, 1.1, tick_threshold))
                # self._ax.set_xticklabels(metric_labels)
                for bar, pattern in zip(bars, patterns):
                    bar.set_vbatch(pattern)
                if text:
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        self._ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 1,
                            f"{score:.3f}",
                            ha="center",
                            va="center",
                        )

        if title:
            self._ax.set_title(title)
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

    def roc(
        self,
        y_trues: list[np.ndarray],
        y_preds: list[np.ndarray],
        labels: list[str],
        title="Roc Curve",
    ):
        for y_true, y_pred, label in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            self._ax.set_title(title)
            self._ax.plot(fpr, tpr, label=label)
        self._ax.set_xlabel("False Positive Rate")
        self._ax.set_ylabel("True Positive Rate")
        return self

    def auc(
        self,
        y_trues: list[np.ndarray],
        y_preds: list[np.ndarray],
        labels: list[str],
        title="AUC Curve",
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

    def image(self, image: Path | Image.Image | cv2.Mat):
        if isinstance(image, Path):
            img = self._ax.imread(image)
        if isinstance(image, Image.Image) or isinstance(image, cv2.Mat):
            img = image
        self._ax.axis("off")
        self._ax.imshow(img)
        return self

    def instance(self):
        return self._ax


class Plot:
    _theme = ""
    _title = ""

    def __init__(self, nrows: int, ncols: int, figsize=(15, 10)):
        self._nrows = nrows
        self._ncols = ncols
        self._fig, self._axs = plt.subplots(nrows, ncols, figsize=figsize)
        self._index = 0

    def subplot(self, x=-1, y=-1):
        if x == -1:
            x = self._index // self._ncols
        if y == -1:
            y = self._index % self._ncols

        if self._nrows == 1 and self._ncols == 1:
            ax = self._axs
        elif self._nrows == 1:
            ax = self._axs[y]
        elif self._ncols == 1:
            ax = self._axs[x]
        else:
            ax = self._axs[x, y]
        self._index += 1
        _subplot = Subplot(ax, self)
        return _subplot

    def tight_layout(self):
        self._fig.tight_layout()
        return self

    def theme(self, theme: str):
        self._theme = theme

    def title(self, title: str):
        self._title = title

    @buildup(desc="image pack")
    def images(self, images: list[Path | Image.Image | cv2.Mat]):
        n = self._nrows * self._ncols
        if n < len(images):
            logging.warning(f"{len(images) - n} is still empty")
        elif n > len(images):
            logging.error(f"The room isn't enough. {n - len(images)} blocks is needed")

        plot = self
        for image in images:
            plot = plot.subplot().image(image).complete()

    def show(self):
        self._setup_params()
        plt.show()

    def save(self, path: Path):
        self._setup_params()
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def save_and_show(self, path: Path):
        self._setup_params()
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.show()

    def _setup_params(self):
        if self._theme:
            sns.set_theme(self._theme)
        if self._title:
            self._axs.set_title(self._title)


class Bar:
    def __init__(self, ax, bars):
        self._ax = ax  # parent
        self._bars = bars

        self.params = {}

    def with_patterns(self, patterns=["/", "\\", "|", "-"]):
        for bar, pattern in zip(self._bars, patterns):
            bar.set_vatch(pattern)  # 设置填充模式
        return self

    def with_text(self, text: str):
        for bar in self._bars:
            height = bar.get_height()
            self._ax.text(
                height + 1,
                bar.get_x() + bar.get_width() / 2,
                text,
                ha="center",
                va="center",
            )

    def complete(self):
        return self._ax


class BarH:
    def __init__(self, ax, bars):
        self._ax = ax  # parent
        self._bars = bars

        self.params = {}

    def with_patterns(self, patterns=["/", "\\", "|", "-"]):
        for bar, pattern in zip(self._bars, patterns):
            bar.set_hatch(pattern)  # 设置填充模式
        return self

    def with_text(self, text: str, font: Font = Font()):
        for bar in self._bars:
            width = bar.get_width()
            self._ax.text(
                bar.get_y() + bar.get_height() / 2,
                width + 1,
                text,
                font.build(),
                ha="left",
                va="center",
            )

    def complete(self):
        return self._ax


# TODO: How to paint with plot for subplot
class Line:
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
    MarkerLiteral = Literal[
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "D",
        "|",
        "_",
    ]
    # '-'：实线
    # '--'：虚线
    # '-.'：点划线
    # ':'：点线
    # '' 或 ' '：无线
    LineLiteral = Literal["-", "--", "-.", ":", "", " "]

    def __init__(self, ax, x, y):
        self._ax = ax
        self._x = x
        self._y = y

    def with_pattern(self, marker: MarkerLiteral, line: LineLiteral, **kwargs):
        self._ax.plot(self._x, self._y, marker + line, **kwargs)
        return self

    def complete(self):
        return self._ax


class PaintHelper:
    @staticmethod
    def colors(n_colors: int, palette="husl"):
        return sns.color_palette(palette, n_colors)

    @staticmethod
    def patterns(mode: str | None = None):
        match mode:
            case "-":
                return ["-o", "-D"]
            case "--":
                return ["--o", "--D"]
            case _:
                return ["-", "-"]

    @staticmethod
    def paint(self, params_dict: dict):
        d = {
            "nrows": 2, # 0 for auto
            "ncols": 2, # 0 for auto, but nrows and ncols need to be set one
            "figsize": (15, 10),
            "tight_layout": True,
            "theme": None,
            "title": None,
            "save": "", # a path
            "show": True,
            "subplots": [{
                "paints": [
                    {
                        "type": "plot",
                        "args": [np.arange(10), np.arange(10), "-o"],  # y=x
                        "kwargs": {
                            "color": "b",
                            "label": "",
                        },
                    }
                ],
                "legend": True,
            }, {
                "paints": [
                    {
                        "type": "barh",
                        "args": ['#label_list', np.arange(10)],
                        "kwargs": {
                            "height": 0.35,
                            "label": '0',
                            "color": 'b',
                        }
                    }
                ],
                "with_text": True, # for len(paints) == 1 
                "legend": True,
                "xlim": [0, 1],
                "ylim": [0, 1],
                "xticks": [0, 1, 0.1],
                "yticks": [0, 1, 0.1],
            }, {
                "paints": [
                    {
                        "type": "image",
                        "args": ["#image"],
                        "kwargs": {
                            'title': "",
                        }
                    }
                ],
            }, {
                "paints": [
                    {
                        "type": "scatter", # hist grid
                        "args": ["#image"],
                        "kwargs": {
                            'title': "",
                        }
                    }
                ],
            }, {
                "paints": [
                    {
                        "type": "scatter",
                        "args": [],
                        "kwargs": {
                            'title': "",
                        }
                    }
                ],
            }, {
                "paints": [
                    {
                        "type": "hist",
                        "args": [],
                        "kwargs": {
                            'title': "",
                        }
                    }
                ],
            }, {
                "paints": [
                    {
                        "type": "grid",
                        "args": [],
                        "kwargs": {
                            'title': "",
                        }
                    }
                ],
            }]
        }

class PaintParamDictBuilder:
    _params = {}
    _subplots = []

    def value(self, name: str, value: str, name_split='.'):
        keys = name.split(name_split)
        c = self._params
        for k in keys:
            c = c[k]
        c = value
        return self

    def paint(self, subplot_index: int, paint_index: int, **kwargs):
        self._params['subplots'][subplot_index]['paints'][paint_index] = kwargs
        return self