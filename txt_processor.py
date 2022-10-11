import os.path
from glob import glob
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map
import re
from utils import create_zip_file


def read_result_style_single(txt_url, model_id, skiprows=0):
    if not os.path.exists(txt_url):
        print(f"{txt_url} not found.")
        data = np.zeros((0, 12))
        img_names = np.zeros((0,))
    else:

        data = np.loadtxt(txt_url, skiprows=skiprows, usecols=range(1, 10))
        img_names = np.loadtxt(txt_url, skiprows=skiprows, usecols=0, dtype=str)

        data = np.concatenate([
            data[:, 1:], data[:, 1:3], data[:, [0]],  # obj * 10 and probability
            np.full((data.shape[0], 1), fill_value=model_id)  # model id
        ], axis=1)
    return data, img_names


def read_gt_style_single(txt_url, model_id, skiprows=2):
    # single thread read per image..
    data = np.loadtxt(txt_url, skiprows=skiprows, usecols=range(8))
    labels = np.loadtxt(txt_url, dtype=str, skiprows=skiprows, usecols=[8])

    if len(data.shape) == 1:
        data = np.reshape(data, (1, -1))
    elif data.shape[0] == 0:
        data = np.zeros((0, 8))

    data = np.concatenate([data, data[:, :2],
                           np.full((data.shape[0], 1), fill_value=0),
                           np.full((data.shape[0], 1), fill_value=model_id),
                           ], axis=1)
    return os.path.basename(txt_url.split(".")[0]), data, labels


def read_result_style_folders(txt_folders, txt_name, sort_by_score=False):
    obj_array_dict = [read_result_style_single(
        os.path.join(f, txt_name + ".txt"), model_id=i,
    ) for i, f in enumerate(txt_folders)]  # read per category

    image_name_list = [d[1] for d in obj_array_dict]
    obj_array_list = [d[0] for d in obj_array_dict]
    # find maps in the category
    unique_img_names = np.unique(np.concatenate([np.unique(n) for n in image_name_list]))

    result_dict = {}
    for img_name in unique_img_names:
        obj_array = [
            l[image_name_list[i] == img_name, :] for i, l in enumerate(obj_array_list)
        ]
        obj_array = [a if len(a.shape) == 2 else a[0, :] for a in obj_array if a.shape[0] > 0]
        if len(obj_array) > 0:
            obj_array = np.concatenate(obj_array)
            if sort_by_score:
                obj_array = obj_array[np.argsort(obj_array[:, -2])[::-1], :]
            result_dict[img_name] = obj_array
    return result_dict


def write_result_style(obj_array_dict, output_dir, file_name):
    with open(os.path.join(output_dir, F"{file_name}.txt"), 'w') as f:
        for img_name, obj_array in obj_array_dict.items():
            for obj in obj_array:
                f.write(f"{img_name} {obj[-2]:.6f} " + " ".join([f"{obj[i]:.1f}" for i in range(8)]))
                f.write("\n")


class TxtProcessor:
    def __init__(self, result_folders, gt_folder, output_folder_for_test):
        self.result_folders = result_folders
        self.gt_folder = gt_folder
        self.num_models = len(result_folders)
        self.output_folder_for_test = output_folder_for_test

    def read_result_style_full(self):
        txt_names = [os.path.basename(x).split(".")[0] for x in glob(os.path.join(self.result_folders[0], "*.txt"))]
        results = {}
        for txt_name in txt_names:
            category_name = re.findall("Task1_(.*)", txt_name)[0]
            result_array_dict = read_result_style_folders(self.result_folders, txt_name, sort_by_score=True)
            for img_name, obj_array in result_array_dict.items():
                if img_name not in results:
                    results[img_name] = {}

                results[img_name][category_name] = obj_array

        return results

    def read_gt_style_folder(self):
        txts = glob(os.path.join(self.gt_folder, "*.txt"))
        runner = partial(read_gt_style_single, model_id=self.num_models)
        results = list(map(runner, txts))
        result_dict = {}
        for img_results in results:
            img_name = img_results[0]
            result_dict[img_name] = {}

            cats = np.unique(img_results[2])
            for cat in cats:
                a = img_results[1][img_results[2] == cat, :]
                if len(a.shape) == 3:
                    a = a[0, :, :]
                result_dict[img_name][cat] = a

        return result_dict

    def write_result_style_full(self, result_list, img_name_list):
        raw_output_dir = os.path.join(self.output_folder_for_test, "test")
        os.makedirs(raw_output_dir, exist_ok=True)

        result_dict = {}
        for i, img_name in enumerate(img_name_list):
            for cat_name, obj_array in result_list[i].items():
                if cat_name not in result_dict:
                    result_dict[cat_name] = {}

                result_dict[cat_name][img_name] = obj_array

        for cat_name, obj_array_dict in result_dict.items():
            write_result_style(obj_array_dict, raw_output_dir, "Task1_"+cat_name)

        create_zip_file(self.output_folder_for_test, raw_output_dir, file_name="test.zip")
