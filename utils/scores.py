import numpy as np
from sklearn import metrics

# y_true, y_pred: (B, C, X)
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.f1_score(y_true, y_pred, average=average)
        result[label] = score

    return result

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.recall_score(y_true, y_pred, average=average)
        result[label] = score

    return result

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.precision_score(y_true, y_pred, average=average)
        result[label] = score

    return result

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.accuracy_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = 2 * intersection / (union + 2 * intersection)
        result[label] = score

    return result

def iou_score(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], *, class_axis: int=1, average: str='binary'):
    n_labels = len(labels)

    y_true_flatten = [yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)]
    y_pred_flatten = [yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)]

    result = dict()
    for (label, y_true, y_pred) in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = intersection / union if union > 0 else 0.0
        result[label] = score

    return result

def scores(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]|tuple[str], 
           metric_labels=['iou', 'accuracy', 'precision', 'recall', 'f1', 'dice'], 
           *, class_axis: int=1, average: str='binary'):
    result = {'mean': {}, 'argmax': {}, 'argmin': {}}
    if 'iou' in metric_labels:
        result['iou'] = iou_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['iou'] = np.array([score for score in result['iou'].values()]).mean()
        result['argmax']['iou'] = labels[np.array([score for score in result['iou'].values()]).argmax()]
        result['argmin']['iou'] = labels[np.array([score for score in result['iou'].values()]).argmin()]
    if 'accuracy' in metric_labels:
        result['accuracy'] = accuracy_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['accuracy'] = np.array([score for score in result['accuracy'].values()]).mean()
        result['argmax']['accuracy'] = labels[np.array([score for score in result['accuracy'].values()]).argmax()]
        result['argmin']['accuracy'] = labels[np.array([score for score in result['accuracy'].values()]).argmin()]
    if 'precision' in metric_labels:
        result['precision'] = precision_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['precision'] = np.array([score for score in result['precision'].values()]).mean()
        result['argmax']['precision'] = labels[np.array([score for score in result['precision'].values()]).argmax()]
        result['argmin']['precision'] = labels[np.array([score for score in result['precision'].values()]).argmin()]
    if 'recall' in metric_labels:
        result['recall'] = recall_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['recall'] = np.array([score for score in result['recall'].values()]).mean()
        result['argmax']['recall'] = labels[np.array([score for score in result['recall'].values()]).argmax()]
        result['argmin']['recall'] = labels[np.array([score for score in result['recall'].values()]).argmin()]
    if 'f1' in metric_labels:
        result['f1'] = f1_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['f1'] = np.array([score for score in result['f1'].values()]).mean()
        result['argmax']['f1'] = labels[np.array([score for score in result['f1'].values()]).argmax()]
        result['argmin']['f1'] = labels[np.array([score for score in result['f1'].values()]).argmin()]
    if 'dice' in metric_labels:
        result['dice'] = dice_score(y_true, y_pred, labels, class_axis=class_axis, average=average)
        result['mean']['dice'] = np.array([score for score in result['dice'].values()]).mean()
        result['argmax']['dice'] = labels[np.array([score for score in result['dice'].values()]).argmax()]
        result['argmin']['dice'] = labels[np.array([score for score in result['dice'].values()]).argmin()]
    return result

if __name__ == '__main__':
    n_batches = 2
    n_classes = 3
    labels = [str(i) for i in range(n_classes)]
    y_true = np.random.rand(n_batches, n_classes, 512, 512)
    y_pred = np.random.rand(n_batches, n_classes, 512, 512)
    y_true[y_true >= 0.5] = 1
    y_true[y_true < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    result = scores(y_true, y_pred, labels)
    from pprint import pp
    pp(result)

# result template:
# labels = ['0', '1', '2']
# {'mean': {'iou': np.float64(0.33355472378823864),
#           'accuracy': np.float64(0.5003592173258463),
#           'precision': np.float64(0.5003786162537677),
#           'recall': np.float64(0.5001200469265391),
#           'f1': np.float64(0.5002488934190708),
#           'dice': np.float64(0.4001592262278031)},
#  'argmax': {'iou': '0',
#             'accuracy': '1',
#             'precision': '0',
#             'recall': '1',
#             'f1': '0',
#             'dice': '0'},
#  'argmin': {'iou': '2',
#             'accuracy': '2',
#             'precision': '1',
#             'recall': '2',
#             'f1': '2',
#             'dice': '2'},
#  'iou': {'0': np.float64(0.3340311765483979),
#          '1': np.float64(0.33356158804182917),
#          '2': np.float64(0.3330714067744889)},
#  'accuracy': {'0': 0.5006504058837891,
#               '1': 0.5006542205810547,
#               '2': 0.4997730255126953},
#  'precision': {'0': np.float64(0.5012290920750281),
#                '1': np.float64(0.49975019164686635),
#                '2': np.float64(0.5001565650394085)},
#  'recall': {'0': np.float64(0.5003410212347635),
#             '1': np.float64(0.5007643214736118),
#             '2': np.float64(0.49925479807124207)},
#  'f1': {'0': np.float64(0.500784662938167),
#         '1': np.float64(0.5002567425950282),
#         '2': np.float64(0.499705274724017)},
#  'dice': {'0': np.float64(0.400502026711725),
#           '1': np.float64(0.4001642983878602),
#           '2': np.float64(0.399811353583824)}}