import ast
import json
import time
import random
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import processpcd as pr

def look_json():
    print(f"opening json")
    with open("/Users/nilanlovelace/Desktop/Capstone/pointcloud_detection_classification/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/annotation3Dlayout/index.json") as f:
        annotations = json.load(f)

    keys = annotations.keys()

    #print(type(annotations))

    key_names = ['date', 'name', 'frames', 'objects', 'extrinsics', 'conflictList', 'fileList']

    y_min = annotations['objects'][0]['polygon'][0]['Ymin']
    y_max = annotations['objects'][0]['polygon'][0]['Ymax']

    return y_min, y_max

def get_groundtruth():
    path = 'groundtruth.xlsx'

    # print("Getting paths from trainsplit")
    try:
        df = pd.read_excel(path)

        return df

    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def parse_annontations(annotations):
    for box_data in annotations:
        position = box_data["geometry"]["position"]  # Example: Accessing position data
        orientation = box_data["geometry"]["orientation"]
        dimensions = box_data["geometry"]["dimensions"]
        class_name = box_data["className"]

def clean_list(original_list):

    cleaned_list = []
    spl = original_list.replace("[", "").replace("]", "").split()

    if len(spl) == 3:
        return [eval(x) for x in spl]
    else:
        cleaned = []
        for i in range(0, len(spl), 3):
            split = spl[i:i+3]
            cleaned.append([eval(x) for x in split])
        return cleaned

def pcd_from_groundtruth(groundtruth, num):
    pcd = pr.load_specific(groundtruth['sequenceName'][num])

    return pcd

def main():
    groundtruth = get_groundtruth()

    selection = random.randint(0, len(groundtruth))
    example = pcd_from_groundtruth(groundtruth, 0)

    center = clean_list(groundtruth['centroid'][0])
    extent = clean_list(groundtruth['orientation'][0])
    R = clean_list(groundtruth['basis'][0])
    rotation = np.eye(3)

    obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)

    o3d.visualization.draw_geometries([example, obb])

    #look_json()

if __name__ == "__main__":
    main()