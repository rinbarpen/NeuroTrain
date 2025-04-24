import logging
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn import metrics
from typing import Literal
from PIL import Image
import cv2

from utils.annotation import buildup
from utils.typed import ClassMetricOneScoreDict

# x, y, shape-like ['-o', '-D'] | font, color, label
# labelsize for tick_params
# linewidth linestyle for grid

AXIS = Literal['x', 'y', 'xy']

class Font:
    _family = 'Times New Roman'
    _weight = 'normal'
    _size = 16
    def __init__(self):
        pass

    def family(self, x: str):
        self._family = x
        return self
    
    def weight(self, x: str):
        self._weight = x
        return self
    
    def size(self, x: int):
        self._size = x
        return self

    def build(self):
        return {
            'family': self._family,
            'weight': self._weight,
            'size': self._size,
        }

class Subplot:
    def __init__(self, ax, parent):
        self._ax = ax
        self._parent = parent

    def plot(self, x, y, *args, **kwargs):
        self._ax.plot(x, y, *args, **kwargs)
        return self
    def bar(self, x, height, *args, **kwargs):
        self._ax.bar(x, height, *args, **kwargs)
        return self
    def barh(self, y, width, *args, **kwargs):
        self._ax.barh(y, width, *args, **kwargs)
        return self
    def scatter(self, x, y, *args, **kwargs):
        self._ax.scatter(x, y, *args, **kwargs)
        return self
    def hist(self, x, bins=None, *args, **kwargs):
        self._ax.hist(x, bins=bins, *args, **kwargs)
        return self
    def grid(self, visible: bool|None=None, **kwargs):
        self._ax.grid(visible, **kwargs)
        return self
    # prop=Font().build()
    def legend(self, *args, **kwargs):
        self._ax.set_legend(*args, **kwargs)

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
        if axis == 'x':
            self.xlabel(label, *args, **kwargs)
        elif axis == 'y':
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
        if axis == 'x':
            self.xticks(ticks)
        elif axis == 'y':
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
        if axis == 'x':
            self.xlim(*args)
        elif axis == 'y':
            self.ylim(*args)
        else:
            self.xlim(*args)
            self.ylim(*args)

    def xaxis_visible(self, visible: bool=True):
        self._ax.xaxis.set_visible(visible)
    def yaxis_visible(self, visible: bool=True):
        self._ax.yaxis.set_visible(visible)
    def axis_visible(self, axis: AXIS, visible: bool=True):
        if axis == 'x':
            self.xaxis_visible(visible)
        elif axis == 'y':
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
    def epoch_loss(self, 
                   num_epoch: int, 
                   losses: list[np.ndarray], 
                   label: str='Loss', 
                   title='Epoch-Loss'):
        if isinstance(losses, list) or isinstance(losses, tuple): 
            losses = np.array(losses, dtype=np.float64)

        epoches = np.arange(1, num_epoch+1, dtype=np.int32)
        self._ax.plot(epoches, losses, label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Loss')
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self
    @buildup(desc="metrics by epoch with one class")
    def epoch_metrics(self, 
                   num_epoch: int, 
                   metrics: list[np.ndarray], 
                   class_label: str, 
                   title: str='Epoch-Label-Metric'):
        if isinstance(metrics, list) or isinstance(metrics, tuple): 
            metrics = np.array(metrics, dtype=np.float64)

        epochs = np.arange(1, num_epoch+1, dtype=np.int32)
        self._ax.plot(epochs, metrics, label=class_label)

        self._ax.set_title(title)
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Metric Score')
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self
    @buildup(desc="metrics by epoch with many classes")
    def many_epoch_metrics(self, 
                   num_epoch: int, 
                   class_metrics: dict[str, list[np.ndarray]], 
                   class_labels: list[str], 
                   title: str='Epoch-Label-Metric'):
        for label in class_labels:
            metrics = class_metrics[label]
            if isinstance(metrics, list) or isinstance(metrics, tuple): 
                metrics = np.array(metrics, dtype=np.float64)

            epochs = np.arange(1, num_epoch+1, dtype=np.int32)
            self._ax.plot(epochs, metrics, label=label)

        self._ax.set_title(title)
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self
    @buildup(desc="loss by epoch with tasks")
    def many_epoch_loss(self, 
                   num_epoch: int, 
                   losses: list[np.ndarray], 
                   labels: list[str]=['Loss'], 
                   title='Epoch-Loss'):        
        epoches = np.arange(1, num_epoch+1, dtype=np.int32)

        for i, label in enumerate(labels):
            self._ax.plot(epoches, losses[i], label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Loss')
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    @buildup(desc="confusion matrix")
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels=None, xlabel='True', ylabel='Prediction', title='Confusion Matrix', cmap='Blues'):
        cm = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap=cmap, annot=True, fmt='d')
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.legend()
        return self

    # scores_map: {'metric_label': {'label'： label_score, ...}, ...}
    @buildup(desc="many metrics")
    def many_metrics(self, label_metric_score: ClassMetricOneScoreDict, title: str|None=None, *, width=0.35):
        colors = sns.color_palette("husl", len(label_metric_score))
        for color, (class_label, metric_score) in zip(colors, label_metric_score.items()):
            metric_labels, scores = metric_score.keys(), metric_score.values()
            self._ax.barh(metric_labels, scores, width, color=color, label=class_label)

        self._ax.set_title(title)
        self._ax.set_ylim(0, 1)
        self._ax.set_xticklabels(metric_labels)
        self._ax.legend()
        return self

    def with_autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            self._ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def roc(self, y_trues: list[np.ndarray], y_preds: list[np.ndarray],
            labels: list[str], title='Roc Curve'):
        for (y_true, y_pred, label) in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            self._ax.set_title(title)
            self._ax.plot(fpr, tpr, label=label)
        self._ax.set_xlabel('False Positive Rate')
        self._ax.set_ylabel('True Positive Rate')
        return self

    def auc(self, y_trues: list[np.ndarray], y_preds: list[np.ndarray],
            labels: list[str], title='AUC Curve'):
        for (y_true, y_pred, label) in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            self._ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        self._ax.set_title(title)
        self._ax.set_xlabel('False Positive Rate')
        self._ax.set_ylabel('True Positive Rate')
        self._ax.legend(loc='lower right')
        return self

    def image(self, image: Path|Image|cv2.Mat):
        if isinstance(image, Path):
            img = self._ax.imread(image)
        if isinstance(image, Image) or isinstance(image, cv2.Mat):
            img = image
        self._ax.axis('off')
        self._ax.imshow(img)
        return self

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
    def images(self, images: list[Path|Image|cv2.Mat]):
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
        if not path.parent.exists():
            os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def save_and_show(self, path: Path):
        self._setup_params()
        if not path.parent.exists():
            os.makedirs(path.parent, exist_ok=True)
        plt.savefig(path)
        plt.show()

    def _setup_params(self):
        if self._theme:
            sns.set_theme(self._theme)
        if self._title:
            self._axs.set_title(self._title)
