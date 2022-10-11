import xml.etree.ElementTree as ET
import numpy as np
import os
import shutil
from copy import deepcopy
from functools import partial
from tqdm.contrib.concurrent import process_map
from glob import glob
from utils import create_zip_file


xml_str = \
    """<?xml version="1.0" encoding="utf-8"?>
<annotation>
    <source>
       <filename>xxx.tif</filename>
       <origin>GF2/GF3</origin>
    </source>
    <research>
        <version>1.0</version>
        <provider>NULL</provider>
        <author>NULL</author>
        <pluginname>FAIR1M</pluginname>
        <pluginclass>object detection</pluginclass>
        <time>2021-03</time>
</research>
<objects>
</objects>
</annotation>"""

object_str = \
    """<object>
    <coordinate>pixel</coordinate>
    <type>rectangle</type>
    <description>None</description>
    <possibleresult>
        <name>category</name>                
        <probability>0.6</probability>
    </possibleresult>
    <points>
    </points>
</object>"""

xml_tree_template = ET.ElementTree(ET.fromstring(xml_str))
xml_object_template = ET.fromstring(object_str)


def read_xmls(folders, xml_name, sort_by_score, model_id_diff=0):
    obj_list_dict = {}

    for model_id, xml_folder in enumerate(folders):
        file_url = os.path.join(xml_folder, xml_name)
        if not os.path.exists(file_url):
            continue
        xml_root = ET.parse(file_url).getroot()

        for line in xml_root.find("objects").findall("object"):
            points = [list(map(float, element.text.split(",")))
                      for element in line.find("points").findall("point")]

            category_name = line.find("possibleresult").find("name").text
            try:
                prob = float(line.find("possibleresult").find("probability").text)
            except:
                prob = 0  # a label perhaps.

            if category_name not in obj_list_dict:
                obj_list_dict[category_name] = []

            obj_list_dict[category_name].append([p for pa in points for p in pa] + [prob, model_id + model_id_diff])

    for k, v in obj_list_dict.items():
        # N by [point x 10, prob, model_id]
        obj_array = np.array(v)
        if sort_by_score:
            score_ord = np.argsort(obj_array[:, -2])
            obj_array = obj_array[score_ord[::-1], :]

        obj_list_dict[k] = obj_array
    return obj_list_dict


def write_xml(obj_array_dict, output_dir, file_name):
    tree = deepcopy(xml_tree_template)
    tree.getroot().find("source/filename").text = file_name + ".tif"
    objects_root = tree.getroot().find("objects")
    format_str = "{:.6f}, {:.6f}"

    for category_name, obj_array in obj_array_dict.items():
        for obj in obj_array:
            node = deepcopy(xml_object_template)
            node.find("possibleresult/name").text = category_name
            node.find("possibleresult/probability").text = "{:.3f}".format(obj[-2])
            points_node = node.find("points")
            for point_id in range(5):
                ET.SubElement(points_node, "point").text = \
                    format_str.format(obj[2 * point_id], obj[2 * point_id + 1])

            objects_root.insert(-1, node)

    with open(os.path.join(output_dir, F"{file_name}.xml"), 'wb') as f:
        tree.write(f)


class XmlProcessor:
    def __init__(self, result_folders, gt_folder, output_folder_for_test):
        self.result_folders = result_folders
        self.gt_folder = gt_folder
        self.num_models = len(result_folders)
        self.output_folder_for_test = output_folder_for_test


    def read_result_style_full(self):
        image_names = [os.path.basename(x).split(".")[0] for x in glob(os.path.join(self.result_folders[0], "*.xml"))]

        results = {}
        for image_name in image_names:
            results[image_name] = read_xmls(self.result_folders, image_name + ".xml", sort_by_score=True,
                                            model_id_diff=0)

        return results

    def read_gt_style_folder(self):
        image_names = [os.path.basename(x).split(".")[0] for x in glob(os.path.join(self.gt_folder, "*.xml"))]

        results = {}
        for image_name in image_names:
            results[image_name] = read_xmls([self.gt_folder], image_name + ".xml", sort_by_score=False,
                                            model_id_diff=self.num_models)

        return results

    def write_result_style_full(self, result_list, img_name_list):
        raw_output_dir = os.path.join(self.output_folder_for_test, "test")
        os.makedirs(raw_output_dir, exist_ok=True)

        for i, img_name in enumerate(img_name_list):
            write_xml(result_list[i], raw_output_dir, img_name)

        create_zip_file(self.output_folder_for_test, raw_output_dir, file_name="test.zip")
