from glob import glob
import numpy as np
import os
from functools import partial
from tqdm.contrib.concurrent import process_map
from xml_processer import XmlProcessor
import argparse
from utils import pickle_load, create_zip_file, pickle_dump

from methods import cluster_objs, fit_ensemble_model, fuse_bbox_fast


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
        scores = np.clip(scores, a_min=0.02, a_max=0.98)
        r_scores = np.log(scores / (1 - scores))
        ret_array[id, -2] = 1. / (1. + np.exp(np.dot(r_scores, weights[:number_of_models]) + weights[number_of_models]))
    return ret_array


def xml_test_single(image_name, weights, iou_thresh, xml_folders, xml_output_dir):
    result_dict = {}
    obj_array_dict = XmlProcessor.read_xmls(xml_folders, image_name + ".xml", sort_by_score=True)

    for category_name, obj_array in obj_array_dict.items():
        cluster_list = cluster_objs(obj_array, iou_thresh=iou_thresh)
        result_dict[category_name] = stacking(obj_array, cluster_list, number_of_models=len(xml_folders),
                                              weights=weights, merge_bbox_flag=True)

    XmlProcessor.write_xml(result_dict, xml_output_dir, image_name)


def xml_train_single(image_name, iou_thresh, xml_folders, gt_xml_folder, merge_bbox_flag):
    obj_array_dict = XmlProcessor.read_xmls(xml_folders, image_name + ".xml", sort_by_score=True)
    gt_obj_dict = XmlProcessor.read_xmls([gt_xml_folder], image_name + ".xml", sort_by_score=False,
                                         model_id_diff=len(xml_folders))

    for category_name, obj_array in obj_array_dict.items():
        # cluster the prediction result
        cluster_list = cluster_objs(obj_array, iou_thresh=iou_thresh)

        # get scores
        scores = np.zeros((len(cluster_list), len(xml_folders)))
        for cluster_id, cluster in enumerate(cluster_list):
            for c in cluster:
                scores[cluster_id, int(obj_array[c, -1])] = obj_array[c, -2]

        # get tps
        tp_flags = np.zeros((len(cluster_list),), dtype=bool)
        if category_name in gt_obj_dict:
            # assign labels to clusters
            for cluster_id, cluster in enumerate(cluster_list):
                if merge_bbox_flag:
                    raise NotImplemented

                tp_list = cluster_objs(
                    np.concatenate((obj_array[[cluster[0]], :], gt_obj_dict[category_name]), axis=0)
                    , iou_thresh=iou_thresh, one_off=True)

                if len(tp_list[0]) > 1:
                    tp_flags[cluster_id] = True

        obj_array_dict[category_name] = (scores, tp_flags)

    return obj_array_dict


def xml_train(images_glob_url, xml_folders, gt_xml_folder, weights_url, iou_thresh):
    image_names = [os.path.basename(x).split(".")[0] for x in glob(images_glob_url)]

    runner_func = partial(xml_train_single,
                          xml_folders=xml_folders,
                          gt_xml_folder=gt_xml_folder,
                          iou_thresh=iou_thresh,
                          merge_bbox_flag=False)
    result_list = process_map(runner_func, image_names, chunksize=10)

    # combine ...
    scores = []
    tp_flags = []
    for img_result_dict in result_list:
        for _, (scores_, tp_flags_) in img_result_dict.items():
            scores.append(scores_)
            tp_flags.append(tp_flags_)

    # train ...
    r_scores = np.clip(np.concatenate(scores, axis=0), a_min=0.02, a_max=0.98)
    r_scores = np.log(r_scores / (1 - r_scores))
    theta = fit_ensemble_model(r_scores, np.concatenate(tp_flags, axis=0))

    print(f"Learned Theta is")
    print(theta)
    pickle_dump(theta, weights_url)


def xml_test(images_glob_url, xml_folders, weights_url, iou_thresh, output_dir):
    image_names = [os.path.basename(x).split(".")[0] for x in glob(images_glob_url)]

    xml_output_dir = os.path.join(output_dir, "test")
    os.makedirs(xml_output_dir, exist_ok=True)

    runner_func = partial(xml_test_single,
                          weights=pickle_load(weights_url),
                          # weights=np.array([-1, -1, -1, 0]),
                          iou_thresh=iou_thresh, xml_folders=xml_folders, xml_output_dir=xml_output_dir)
    process_map(runner_func, image_names, chunksize=10)

    create_zip_file(output_dir, xml_output_dir, file_name="test.zip")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('image_glob_url', help='image_glob_url')
    parser.add_argument('--root_dir', help='result_root')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='iou threshold for the clustering')
    parser.add_argument('--input_names', nargs='+', help='input_names')
    parser.add_argument('--output_name', help='output_name for test or '
                                              'folder for ground truth labels.')
    parser.add_argument('--format', help='DOTA or FAIR')
    parser.add_argument('--mode', help='train or test')
    parser.add_argument('--weights_url', help='weights_url')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_format = args.format.lower()
    mode = args.mode.lower()
    assert data_format in ["fair", "dota"]
    assert mode in ["train", "test"]
    if data_format == "fair":
        if mode == "test":
            xml_test(args.image_glob_url,
                     [os.path.join(args.root_dir, n, "test") for n in args.input_names],
                     weights_url=args.weights_url,
                     iou_thresh=args.iou_thresh,
                     output_dir=os.path.join(args.root_dir, args.output_name))
        else:
            xml_train(args.image_glob_url,
                      [os.path.join(args.root_dir, n, "test") for n in args.input_names],
                      weights_url=args.weights_url,
                      iou_thresh=args.iou_thresh,
                      gt_xml_folder=args.output_name)
    elif data_format == "dota":
        raise NotImplemented
