import logging
import numpy as np
import scipy
from sklearn import metrics

from utils.typed import *
from utils.painter import Plot
from utils.data_saver import DataSaver
from utils.typed import ScoreAggregator, FLOAT
from utils.defines import neighbour_code_to_normals

# y_true, y_pred: (B, C, X)
def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.f1_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.recall_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.precision_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        score = metrics.accuracy_score(y_true, y_pred)
        result[label] = np.float64(score)

    return result

def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]


    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        score = 2 * intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result

# Dice Similarity Coefficient (DSC)
dsc_score = dice_score

class SurfaceDistanceDict(TypedDict):
    distances_gt_to_pred: np.ndarray 
    distances_pred_to_gt: np.ndarray 
    surfel_areas_gt:      np.ndarray 
    surfel_areas_pred:    np.ndarray 

# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code 
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes) 
# it contains the surface normals of the triangles (called "surfel" for 
# "surface element" in the following). The length of the normal 
# vector encodes the surfel area.
#
# created by compute_surface_area_lookup_table.ipynb using the 
# marching_cube algorithm, see e.g. https://en.wikipedia.org/wiki/Marching_cubes

# Normalized Surface Dice (NSD)
def normalized_surface_dice_score(y_true: np.ndarray, y_pred: np.ndarray, spacing_mm: tuple[int, int, int], labels: ClassLabelsList, *, class_axis: int=1, **kwargs):
    """normalized_surface_dice_score

    Args:
        y_true (np.ndarray): (B, CLASS, H, W, D)
        y_pred (np.ndarray): (B, CLASS, H, W, D)
        spacing_mm (int): 
        class_axis (int): Defaults to 1.

        kwargs[percent] (list[int], optional): Default is None
    Results:
        average_surface_distance (tuple[float64, float64]): 
        hausdorff ({percent}%) (float64): 
        surface_overlap (tuple[float64, float64]): 
        surface_dice (float64): 
    """
    n_labels = len(labels)

    y_true_flatten = [
        yt for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    results = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        results[label] = dict()
        surface_distances = compute_surface_distances(y_true, y_pred, spacing_mm)
        
        average_surface_distance = compute_average_surface_distance(surface_distances)
        results["average_surface_distance"] = average_surface_distance # mm
        
        if 'percent' in kwargs:
            for percent in kwargs['percent']:
                results[label][f"hausdorff ({percent}%)"] = compute_robust_hausdorff(surface_distances, percent=percent) # mm

        if 'overlap_tolerance_mm' in kwargs:
            results[label]["surface_overlap"] = compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm=kwargs['overlap_tolerance_mm'])
        if 'dice_tolerance_mm' in kwargs:
            results[label]["surface_dice"] = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=kwargs['dice_tolerance_mm'])
    
    return results


def compute_surface_distances(mask_gt: np.ndarray, mask_pred: np.ndarray, spacing_mm: tuple[int, int, int]) -> SurfaceDistanceDict:
    """Compute closest distances from all surface points to the other surface.

    Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
    the predicted mask `mask_pred`, computes their area in mm^2 and the distance
    to the closest point on the other surface. It returns two sorted lists of
    distances together with the corresponding surfel areas. If one of the masks
    is empty, the corresponding lists are empty and all distances in the other
    list are `inf` 

    Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.
        spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
            direction 

    Returns:
        A dict with 
        "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
            from all ground truth surface elements to the predicted surface, 
            sorted from smallest to largest
        "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
            from all predicted surface elements to the ground truth surface, 
            sorted from smallest to largest 
        "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of 
            the ground truth surface elements in the same order as 
            distances_gt_to_pred
        "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of 
            the predicted surface elements in the same order as 
            distances_pred_to_gt
        
    """

    # compute the area for all 256 possible surface elements 
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx,0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx,1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx,2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
            neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return {"distances_gt_to_pred":  np.array([]), 
                "distances_pred_to_gt":  np.array([]), 
                "surfel_areas_gt":       np.array([]), 
                "surfel_areas_pred":     np.array([])}
        
    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    # print("bounding box min = {}".format(bbox_min))
    # print("bounding box max = {}".format(bbox_max))

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower, 
    # the right and the back side. This is required to obtain the "full" 
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                            bbox_min[1]:bbox_max[1]+1,
                                            bbox_min[2]:bbox_max[2]+1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                                bbox_min[1]:bbox_max[1]+1,
                                                bbox_min[2]:bbox_max[2]+1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128,64],
                        [32,16]],
                        [[8,4],
                        [2,1]]])
    neighbour_code_map_gt = scipy.ndimage.filters.correlate(cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0) 
    neighbour_code_map_pred = scipy.ndimage.filters.correlate(cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0) 

    # create masks with the surface voxels
    borders_gt   = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the surface voxels)
    if borders_gt.any():
        distmap_gt = scipy.ndimage.morphology.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():  
        distmap_pred = scipy.ndimage.morphology.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt   = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:,0]
        surfel_areas_gt      = sorted_surfels_gt[:,1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:,0]
        surfel_areas_pred    = sorted_surfels_pred[:,1]


    return {"distances_gt_to_pred":  distances_gt_to_pred, 
            "distances_pred_to_gt":  distances_pred_to_gt, 
            "surfel_areas_gt":       surfel_areas_gt, 
            "surfel_areas_pred":     surfel_areas_pred}

def compute_average_surface_distance(surface_distances: SurfaceDistanceDict) -> tuple[FLOAT, FLOAT]:
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt      = surface_distances["surfel_areas_gt"]
    surfel_areas_pred    = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = np.sum( distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
    average_distance_pred_to_gt = np.sum( distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)

def compute_robust_hausdorff(surface_distances: SurfaceDistanceDict, percent: int) -> FLOAT:
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt      = surface_distances["surfel_areas_gt"]
    surfel_areas_pred    = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:
        surfel_areas_cum_gt   = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred)-1)]
    else:
        perc_distance_gt_to_pred = np.inf

    if len(distances_pred_to_gt) > 0:
        surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
        idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt)-1)]
    else:
        perc_distance_pred_to_gt = np.inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)

def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm: float) -> tuple[FLOAT, FLOAT]:
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt      = surface_distances["surfel_areas_gt"]
    surfel_areas_pred    = surface_distances["surfel_areas_pred"]
    rel_overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
    rel_overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
    return (rel_overlap_gt, rel_overlap_pred)

def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm: float) -> FLOAT:
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt      = surface_distances["surfel_areas_gt"]
    surfel_areas_pred    = surface_distances["surfel_areas_pred"]
    overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
        np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice

def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    *,
    class_axis: int = 1,
    average: str = "binary",
):
    n_labels = len(labels)

    y_true_flatten = [
        yt.flatten() for yt in np.split(y_true, n_labels, axis=class_axis)
    ]
    y_pred_flatten = [
        yp.flatten() for yp in np.split(y_pred, n_labels, axis=class_axis)
    ]

    result = dict()
    for label, y_true, y_pred in zip(labels, y_true_flatten, y_pred_flatten):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        score = intersection / union if union > 0 else 0.0
        result[label] = np.float64(score)

    return result

def scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: ClassLabelsList,
    metric_labels: MetricLabelsList,
    *,
    class_axis: int = 1,
):
    result = {metric: {label: FLOAT() for label in labels} for metric in metric_labels}

    MAP = {
        "iou": iou_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "dice": dice_score,
    }

    for metric in metric_labels:
        result[metric] = MAP[metric](
            y_true, y_pred, labels, 
            class_axis=class_axis
        )

    return result

# result template:
# labels = ['0', '1', '2']
# result:
# {'iou': {'0': np.float64(0.3340311765483979),
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
# result_after:
# {'mean': {'iou': np.float64(0.33355472378823864),
#          'accuracy': np.float64(0.5003592173258463),
#          'precision': np.float64(0.5003786162537677),
#          'recall': np.float64(0.5001200469265391),
#          'f1': np.float64(0.5002488934190708),
#          'dice': np.float64(0.4001592262278031)},
# 'argmax': {'iou': '0',
#            'accuracy': '1',
#            'precision': '0',
#            'recall': '1',
#            'f1': '0',
#            'dice': '0'},
# 'argmin': {'iou': '2',
#            'accuracy': '2',
#            'precision': '1',
#            'recall': '2',
#            'f1': '2',
#            'dice': '2'}}

class ScoreCalculator:
    def __init__(
        self,
        output_dir: FilePath,
        class_labels: ClassLabelsList,
        metric_labels: MetricLabelsList,
        *,
        logger=None,
        saver: DataSaver,
    ):
        self.logger = logger or logging.getLogger()
        self.saver = saver
        self.output_dir = Path(output_dir)
        self.class_labels = class_labels
        self.metric_labels = metric_labels

        # {'recall': {
        #   '0': [], '1': []},
        #  'precision: {
        #   '0': [], '1': []}}
        self.all_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }
        # but epoch_metric_label_scores is saved by epoch
        self.epoch_metric_label_scores: MetricClassManyScoreDict = {
            metric: {label: [] for label in class_labels} for metric in metric_labels
        }

        for class_label in class_labels:
            class_dir = self.output_dir / class_label
            class_dir.mkdir(parents=True, exist_ok=True)

    def finish_one_batch(self, targets: np.ndarray, outputs: np.ndarray):
        metrics = scores(
            targets, outputs, labels=self.class_labels, metric_labels=self.metric_labels
        )

        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = metrics[metric_label][class_label]
                self.all_metric_label_scores[metric_label][class_label].append(score)

        return metrics

    def finish_one_epoch(self):
        for metric_label in self.metric_labels:
            for class_label in self.class_labels:
                score = np.mean(self.all_metric_label_scores[metric_label][class_label])
                self.all_metric_label_scores[metric_label][class_label] = []
                self.epoch_metric_label_scores[metric_label][class_label].append(score)
        return self.epoch_metric_label_scores

    def record_batches(self):
        self._record(self.all_metric_label_scores)

    def record_epochs(self, n_epochs: int):
        self._record(self.epoch_metric_label_scores)

        epoch_metrics_image = self.output_dir / "epoch_metrics_curve_per_classes.png"
        # paint metrics curve for all classes in one figure
        n = len(self.metric_labels)
        if n > 4:
            nrows, ncols = (n + 2) // 3, 3
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 1, n

        scores = ScoreAggregator(self.epoch_metric_label_scores)
        # for global, plot all classes in one figure with all metrics
        plot = Plot(nrows, ncols)
        for metric in self.metric_labels:
            plot.subplot().many_epoch_metrics_by_class(
                n_epochs,
                self.epoch_metric_label_scores[metric],
                self.class_labels,
                title=metric,
            ).complete()
        plot.save(epoch_metrics_image)
        # for global, plot all metrics in one figure
        epoch_metrics_image = self.output_dir / "epoch_metrics_curve.png"
        plot = Plot(1, 1)
        plot.subplot().many_epoch_metrics(
            n_epochs, scores.m2_mean, metric_labels=self.metric_labels, title="All Metrics"
        ).complete()
        plot.save(epoch_metrics_image)
        # for every classes
        for label in self.class_labels:
            epoch_metric_image = self.output_dir / label / "metrics.png"
            plot = Plot(1, 1)
            plot.subplot().many_epoch_metrics(n_epochs, scores.cmm[label], metric_labels=self.metric_labels, title=label).complete()
            plot.save(epoch_metric_image)
        # for every classes with sole metric
        for metric in self.metric_labels:
            for label in self.class_labels:
                m_scores = np.array(
                    self.epoch_metric_label_scores[metric][label], dtype=np.float64
                )
                epoch_metric_image = self.output_dir / label / f"{metric}.png"
                plot = Plot(1, 1)
                plot = (
                    plot.subplot()
                    .epoch_metrics(
                        n_epochs, m_scores, label, title=f"Epoch-{label}-{metric}"
                    )
                    .complete()
                )
                plot.save(epoch_metric_image)

    def _record(self, mc_scores: MetricClassManyScoreDict):
        scores = ScoreAggregator(mc_scores)

        self.saver.save_all_metric_by_class(scores.cmm)
        self.saver.save_mean_metric_by_class(scores.cm1_mean)
        self.saver.save_std_metric_by_class(scores.cm1_std)
        self.saver.save_mean_metric(scores.ml1_mean)
        self.saver.save_std_metric(scores.ml1_std)

        mean_metrics_image = self.output_dir / "mean_metrics_per_classes.png"
        # paint mean metrics for all classes
        Plot(1, 1).subplot().many_metrics(scores.cm1_mean).complete().save(
            mean_metrics_image
        )
        for label in self.class_labels:
            mean_image = self.output_dir / label / "mean_metric.png"
            Plot(1, 1).subplot().metrics(scores.cm1_mean[label], label).complete().save(
                mean_image
            )
        mean_metrics_image = self.output_dir / "mean_metrics.png"
        # metric mean
        Plot(1, 1).subplot().metrics(scores.ml1_mean).complete().save(
            mean_metrics_image
        )
