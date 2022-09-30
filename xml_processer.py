import xml.etree.ElementTree as ET
import numpy as np
import os
import shutil
from copy import deepcopy
from functools import partial
from tqdm.contrib.concurrent import process_map

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


class XmlProcessor:
    @staticmethod
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

    @staticmethod
    def generate_xml_single(file_template, object_template, output_dir, info):
        image_name, objects = info
        tree = deepcopy(file_template)
        tree.getroot().find("source/filename").text = image_name + ".tif"
        objects_root = tree.getroot().find("objects")

        for obj in objects:
            node = deepcopy(object_template)
            node.find("possibleresult/name").text = obj.cat
            node.find("possibleresult/probability").text = "{:.3f}".format(float(obj.score))
            points_node = node.find("points")
            for p in obj.get_xml_points():
                ET.SubElement(points_node, "point").text = p

            objects_root.insert(-1, node)

        with open(os.path.join(output_dir, "{}.xml".format(image_name)), 'wb') as f:
            tree.write(f)

    @staticmethod
    def generate_xmls(output_dir, images, output_score_list=False):
        """
        ##########################
        # create xmls
        ##########################
        :param output_dir:
        :param images: dict of image_name: [objects]
        :return:
        """
        # check test folder.
        output_dir_new = os.path.join(output_dir, "test")
        if os.path.isdir(output_dir_new):
            shutil.rmtree(output_dir_new)

        os.makedirs(output_dir_new, exist_ok=False)

        single_func = partial(XmlProcessor.generate_xml_single, xml_tree_template, xml_object_template,
                              output_dir_new,
                              output_score_list=output_score_list)
        # for image_name, objects in images.items():
        #     generate_xmls_single(output_xml_tree, object_xml, )
        process_map(single_func, [(k, v) for k, v in images.items()], max_workers=4, chunksize=100)

        return output_dir_new

    @staticmethod
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
