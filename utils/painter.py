import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn import metrics


class Subplot:
    def __init__(self, ax, parent):
        self._ax = ax
        self._parent = parent

    def plot(self, x, y):
        self._ax.plot(x, y)
        return self
    def bar(self, x, height, **kwargs):
        self._ax.bar(x, height, **kwargs)
        return self
    def barh(self, y, width, **kwargs):
        self._ax.barh(y, width, **kwargs)
        return self
    def scatter(self, x, y, **kwargs):
        self._ax.scatter(x, y, **kwargs)
        return self
    def hist(self, x, bins=None, **kwargs):
        self._ax.hist(x, bins=bins, **kwargs)
        return self

    def figsize(self, figsize: tuple[int, int]):
        self._ax.set_figsize(figsize)
        return self
    def xlabel(self, xlabel: str):
        self._ax.set_xlabel(xlabel)
        return self
    def ylabel(self, ylabel: str):
        self._ax.set_xlabel(ylabel)
        return self
    def label(self, label: str):
        self._ax.set_label(label)
        return self
    def title(self, title: str):
        self._ax.set_title(title)
        return self

    def complete(self):
        return self._parent

    def epoch_loss(self, 
                   num_epoch: int, 
                   losses: np.ndarray|list[np.ndarray]|list[float]|list[list[float]], 
                   labels: str|tuple[str, ...]='Loss', 
                   title='Epoch-Loss'):
        if isinstance(losses, list[float]):
            losses = np.array([loss for loss in losses], dtype=np.float64)
        elif isinstance(losses, list[list[float]]):
            losses = np.array([[loss for loss in loss_a] for loss_a in losses], dtype=np.float64)

        epoches = np.arange(1, num_epoch+1, dtype=np.int32)
        if isinstance(labels, str):
            self._ax.plot(epoches, losses, label=labels)
        if isinstance(labels, tuple):
            for (loss, label) in zip(losses, labels):
                self._ax.plot(epoches, loss, label=label)

        self._ax.set_title(title)
        self._ax.set_xlabel('Epoch')
        self._ax.set_ylabel('Loss')
        self._ax.set_xlim(1, num_epoch)
        self._ax.legend()
        return self

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels=None, xlabel='True', ylabel='Prediction', title='Confusion Matrix', cmap='Blues'):
        cm = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap=cmap, annot=True, fmt='d')
        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.legend()
        return self

    # scores_map: the inputs: {'metric_label': score, ...}
    def metric(self, scores_map: dict[str, np.float64], title='XX Metric'):
        labels, scores = scores_map.keys(), scores_map.values()
        colors = sns.color_palette("husl", len(labels))
        self._ax.set_title(title)
        self._ax.bar(labels, scores, color=colors)
        self._ax.set_ylim(0, 1)
        return self

    def roc(self, y_trues: list[np.ndarray], y_preds: list[np.ndarray],
            labels: list[str], title='Roc Curve'):
        for (y_true, y_pred, label) in zip(y_trues, y_preds, labels):
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            self._ax.set_title(title)
            self._ax.plot(tpr, fpr, label=label)
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

    def image(self, image_path: Path):
        img = self._ax.imread(image_path)
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

    def theme(self, theme: str):
        self._theme = theme

    def title(self, title: str):
        self._title = title

    # input: {'f1': {'label_name': score, ...}, 'accuracy': {...}, ...}
    def metrics(self, scores_map: dict[str, dict[str, np.float64]]):
        for metric_name, metric in scores_map.items():
            self.subplot().metric(metric, title=metric_name).complete()
        return self

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


if __name__ == '__main__':
    # y_true = [0, 1, 2, 2, 0, 1, 0, 2, 1, 0]
    # y_pred = [0, 2, 1, 2, 0, 0, 0, 2, 1, 1]
    # labels = ['c0', 'c1', 'c2']

    # plot = Plot(1, 2)
    # plot.subplot().confusion_matrix(y_true, y_pred, labels=labels).complete()
    # plot.show()

    # c = {
    #     'f1': {
    #         'c0': 0.56,
    #         'c1': 0.67,
    #         'c2': 0.99,
    #     },
    #     'recall': {
    #         'c0': 0.56,
    #         'c1': 0.67,
    #         'c2': 0.99,
    #     }
    # }
    # # plot.subplot().metric(c).complete()
    # plot.metrics(c)
    # plot.show()

    # losses = [np.random.rand(20), np.random.rand(20), np.random.rand(20)]
    # plot = Plot(1, 1)
    # plot.subplot().epoch_loss(20, losses, labels=('Loss1', 'Loss2', 'Loss3'))
    # plot.save_and_show(Path('epoch_loss.png'))

    # y_true = np.array([0, 0, 1, 1])
    # y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    # plot = Plot(1, 2)
    # plot.subplot().roc([y_true], [y_pred], labels=['test']).complete()
    # plot.subplot().auc([y_true], [y_pred], labels=['test']).complete()
    # plot.show()

    img = plt.imread('epoch_loss.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
