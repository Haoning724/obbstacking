from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import math
from scipy.optimize import fmin_bfgs


def clip_to_valid(x):
    return np.clip(x, a_min=0.01, a_max=1 - 1e-5)


def iou_calculation(obj_array, center_obj_id, other_obj_id):
    def polygon_from_obj_array(obj_array, obj_id):
        return Polygon([[obj_array[obj_id, i], obj_array[obj_id, i + 1]] for i in range(0, 8, 2)])

    pgn1 = polygon_from_obj_array(obj_array, center_obj_id)
    pgn2 = polygon_from_obj_array(obj_array, other_obj_id)
    return pgn1.intersection(pgn2).area / unary_union([pgn1, pgn2]).area


def cluster_objs(obj_array, iou_thresh, one_off=False):
    ret = []

    minx = np.min(obj_array[:, :-2:2], axis=1)
    miny = np.min(obj_array[:, 1:-2:2], axis=1)
    maxx = np.max(obj_array[:, :-2:2], axis=1)
    maxy = np.max(obj_array[:, 1:-2:2], axis=1)
    model_id = obj_array[:, -1]

    available_flag = np.ones((obj_array.shape[0]), dtype=bool)
    for center_obj_id in range(obj_array.shape[0]):
        if not available_flag[center_obj_id]:
            continue

        available_flag[center_obj_id] = False
        ret.append([center_obj_id])

        potential_objs = np.logical_and.reduce([available_flag,
                                                model_id != model_id[center_obj_id],
                                                minx < maxx[center_obj_id], maxx > minx[center_obj_id],
                                                miny < maxy[center_obj_id], maxy > miny[center_obj_id]
                                                ])
        for other_obj_id in np.nonzero(potential_objs)[0]:
            iou = iou_calculation(obj_array, center_obj_id, other_obj_id)

            if iou > iou_thresh:
                available_flag[other_obj_id] = False
                ret[-1].append(other_obj_id)

        if one_off:
            break

    return ret


def bbox_gather_fast(obj_array, cluster_list, weights):
    # cluster_flag = np.zeros((obj_array.shape[0]), dtype=bool)

    s = obj_array[:, -2]
    s = clip_to_valid(s)
    w = np.take(weights, obj_array[:, -1].astype(int))
    scores = 1 / (1 + np.exp(np.log(s / (1 - s)) * w) + weights[-1])

    # N by 2 (sides) by 2 (x y)
    dual_sides = np.reshape(obj_array[:, :4] - obj_array[:, 2:6], (-1, 2, 2))

    # N by 2 (x,y)
    main_sides = np.copy(dual_sides[:, 0, :])
    for cluster in cluster_list:
        for ci, c in enumerate(cluster):
            if ci == 0:
                continue
            else:
                main_sides[c, :] = main_sides[cluster[0], :]

    # N by 1
    angles = np.arctan2(main_sides[:, 1], main_sides[:, 0])

    # N by 2
    dual_relative_angles = np.arccos(
        np.clip(np.abs(np.sum(dual_sides * main_sides[:, None, :], axis=-1)
                       / np.linalg.norm(dual_sides, axis=-1)
                       / np.linalg.norm(main_sides, axis=-1)[:, None]), a_min=0, a_max=1)
    )

    # N
    relative_angles = np.min(dual_relative_angles, axis=1)
    width_side_id = np.argmin(dual_relative_angles, axis=1)
    widths = np.linalg.norm(np.take_along_axis(dual_sides, width_side_id[:, None, None], axis=1),
                            axis=-1)[:, 0]
    heights = np.linalg.norm(np.take_along_axis(dual_sides, 1 - width_side_id[:, None, None], axis=1),
                             axis=-1)[:, 0]
    center_xs = np.mean(obj_array[:, :8:2], axis=-1)
    center_ys = np.mean(obj_array[:, 1:8:2], axis=-1)

    return scores, angles, relative_angles, center_xs, center_ys, widths, heights


def fuse_bbox_fast(obj_array, cluster_list, weights):
    def rotate_points(points, centers, angle):
        M = np.array([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
        return np.matmul(M, (points - centers)[:, :, None])[:, :, 0] + centers

    # scores, angles, relative_angles, center_xs, center_ys, widths, heights
    gatherd_bbox_param = bbox_gather_fast(obj_array, cluster_list, weights)

    for cluster in cluster_list:
        if len(cluster) == 1:
            continue

        # relative_angles, center_xs, center_ys, widths, heights
        ensembled_params = []
        scores = gatherd_bbox_param[0][cluster]
        for param_id in range(2, 7):
            ensembled_params.append(
                np.dot(gatherd_bbox_param[param_id][cluster], scores) / np.sum(scores)
            )

        points = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                points.append(ensembled_params[3] * 0.5 * x + ensembled_params[1])
                points.append(ensembled_params[4] * 0.5 * y * x + ensembled_params[2])

        points = rotate_points(np.reshape(np.array(points), (4, 2)),
                               np.stack(ensembled_params[1:3], axis=-1),
                               gatherd_bbox_param[1][cluster[0]] + ensembled_params[0])

        points = np.reshape(points, (-1,))
        points = np.append(points, points[:2])

        for i in range(10):
            obj_array[cluster[0], i] = points[i]


def fit_ensemble_model(f, y):
    # adapted from https://github.com/neal-o-r/platt
    eps = np.finfo(np.float).tiny  # to avoid division by 0 warning
    # Bayes priors
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 1. / (prior0 + 2.)
    T1 = 1. - T
    model_num = f.shape[1]

    def objective(theta):
        A, B = theta[:model_num], theta[model_num]
        E = np.exp(np.matmul(f, A) + B)
        P = 1. / (1. + E)
        l = -(T * np.log(P + eps) + T1 * np.log(1. - P + eps))
        return l.sum()

    def grad(theta):
        A, B = theta[:model_num], theta[model_num]
        E = np.exp(np.matmul(f, A) + B)
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        dA = np.matmul(TEP_minus_T1P, f)
        dB = np.sum(TEP_minus_T1P)
        return np.append(dA, dB)

    AB0 = np.ones((model_num,)) / model_num
    AB0 = np.append(AB0, 0)
    params = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return params


def assign_score_and_labels(obj_array, gt_obj_array, iou_thresh, merge_bbox_flag, num_models):
    # cluster the prediction result
    cluster_list = cluster_objs(obj_array, iou_thresh=iou_thresh)

    # get scores
    scores = np.zeros((len(cluster_list), num_models))
    for cluster_id, cluster in enumerate(cluster_list):
        for c in cluster:
            scores[cluster_id, int(obj_array[c, -1])] = obj_array[c, -2]

    # get tps
    tp_flags = np.zeros((len(cluster_list),), dtype=bool)
    # assign labels to clusters
    if gt_obj_array.shape[0] > 0:
        for cluster_id, cluster in enumerate(cluster_list):
            if merge_bbox_flag:
                raise NotImplemented

            tp_list = cluster_objs(
                np.concatenate((obj_array[[cluster[0]], :], gt_obj_array), axis=0)
                , iou_thresh=iou_thresh, one_off=True)

            if len(tp_list[0]) > 1:
                tp_flags[cluster_id] = True

    return scores, tp_flags, cluster_list, obj_array


def combine_results_and_train(result_list):
    # combine ...
    scores = []
    tp_flags = []
    for img_result_dict in result_list:
        for k, v in img_result_dict.items():
            if k == '__meta__':
                continue
            scores_, tp_flags_, _, _ = v
            scores.append(scores_)
            tp_flags.append(tp_flags_)

    # train ...
    r_scores = clip_to_valid(np.concatenate(scores, axis=0))
    r_scores = np.log(r_scores / (1 - r_scores))
    theta = fit_ensemble_model(r_scores, np.concatenate(tp_flags, axis=0))
    print(np.mean(np.concatenate(tp_flags, axis=0)))
    return theta


def stacking(obj_array, cluster_list, number_of_models, weights, merge_bbox_flag=True):
    # apply bbox fusion
    if merge_bbox_flag:
        fuse_bbox_fast(obj_array, cluster_list, weights)

    # apply meta-learner
    center_ids = [a[0] for a in cluster_list]  # only pick the 1st one as the fused bbox.
    ret_array = np.copy(obj_array[center_ids, :])
    for id, cluster in enumerate(cluster_list):
        scores = np.zeros((number_of_models,))
        for i in cluster:
            scores[int(obj_array[i, -1])] = obj_array[i, -2]
        scores = clip_to_valid(scores)
        r_scores = np.log(scores / (1 - scores))
        ret_array[id, -2] = 1. / (1. + np.exp(np.dot(r_scores, weights[:number_of_models]) + weights[-1]))
    return ret_array