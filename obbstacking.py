from glob import glob
import numpy as np
import os
from functools import partial
from tqdm.contrib.concurrent import process_map
from xml_processer import XmlProcessor
from txt_processor import TxtProcessor
import argparse
from utils import pickle_load, create_zip_file, pickle_dump
from methods import assign_score_and_labels, cluster_objs, stacking, combine_results_and_train


def train_single(dict_pair, iou_thresh, merge_bbox_flag, num_models):
    result_dict, gt_dict=  dict_pair
    result = {}
    for cat_name, obj_array in result_dict.items():
        try:
            gt_obj_array = gt_dict[cat_name]
        except:
            gt_obj_array = np.zeros((0, 12))

        result[cat_name] = assign_score_and_labels(obj_array, gt_obj_array, iou_thresh=iou_thresh,
                                                   merge_bbox_flag=merge_bbox_flag,
                                                   num_models=num_models)

    return result


def test_single(result_dict, iou_thresh, weights, merge_bbox_flag, num_models):
    result = {}
    for cat_name, obj_array in result_dict.items():
        cluster_list = cluster_objs(obj_array, iou_thresh=iou_thresh)
        result[cat_name] = stacking(obj_array, cluster_list, number_of_models=num_models,
                                    weights=weights, merge_bbox_flag=merge_bbox_flag)

    return result


def train(file_processor, weights_url, iou_thresh):
    print("Reading data from the disk ..")
    result_dict = file_processor.read_result_style_full()
    gt_dict = file_processor.read_gt_style_folder()

    print("Processing ...")
    runner = partial(train_single, iou_thresh=iou_thresh,
                     merge_bbox_flag=False, num_models=file_processor.num_models)

    dict_pairs = [(result_dict[img_name], gt_dict[img_name] if img_name in gt_dict else {}) 
        for img_name in list(result_dict.keys())]
    results = process_map(runner, dict_pairs, chunksize=10)

    theta = combine_results_and_train(results)

    print(f"Learned Theta is")
    print(theta)
    pickle_dump(theta, weights_url)


def test(file_processor, weights_url, iou_thresh):
    print("Reading data from the disk ..")
    result_dict = file_processor.read_result_style_full()

    print("Processing ...")
    runner = partial(test_single, iou_thresh=iou_thresh,
                     weights=pickle_load(weights_url),
                     merge_bbox_flag=True, num_models=file_processor.num_models)

    img_names = list(result_dict.keys())
    input_dict_list = [result_dict[img_name] for img_name in img_names]
    results = process_map(runner, input_dict_list, chunksize=10)

    print(f"Writing results to {file_processor.output_folder_for_test} ..")
    file_processor.write_result_style_full(results, img_names)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
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
    if data_format == 'dota':
        file_processor = TxtProcessor(
            [os.path.join(args.root_dir, n, "test") for n in args.input_names], args.output_name, 
            os.path.join(args.root_dir, args.output_name))
    elif data_format == 'fair':
        file_processor = XmlProcessor([os.path.join(args.root_dir, n, "test") for n in args.input_names], args.output_name,
        os.path.join(args.root_dir, args.output_name))
    else:
        raise NotImplemented

    if mode == "train":
        train(file_processor, args.weights_url, args.iou_thresh, )
    else:
        test(file_processor, args.weights_url, args.iou_thresh, )
