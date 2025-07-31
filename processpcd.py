import subprocess
import sys
import os
import shutil
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from PIL import Image
from scipy.io import loadmat

def get_groundtruth():
    path = 'groundtruth.xlsx'

    # print("Getting paths from trainsplit")
    train_paths = []
    try:
        df = pd.read_excel(path)
        ''' with open(f"{path}", "r") as file:
            content = file.readlines()

            for line in content:
                # con = line.split()
                train_paths.append(str(line[14:-3]))'''

        return df

    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_trainsplit():
    path = 'trainsplit.csv'

    #print("Getting paths from trainsplit")
    train_paths = []
    try:
        with open(f"{path}", "r") as file:
            content = file.readlines()

            for line in content:
                #con = line.split()
                train_paths.append(str(line[14:-3]))

    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    #print(f"Trainsplit retrieved. {len(train_paths)} paths extracted")
    return train_paths#[:2]

def get_valsplit():
    path = 'valsplit.csv'
    #print(f"Extracting paths from valsplit")
    val_paths = []

    try:
        with open(f"{path}", "r") as file:
            content = file.readlines()

            for line in content:
                # con = line.split()
                val_paths.append(str(line[14:-3]))
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    #print(f"Valsplit retrieved. {len(val_paths)} paths extracted")
    return val_paths#[:2]

def get_extrinsics(dirName):
    extDest = f"{dirName}/extrinsics"
    dest = list(os.listdir(extDest))

    #print(f"Extracting extrinsics from {dirName}")
    if len(dest) != 0:
        extrinFile = dest[0]
    else:
        return "Extrinsics not found"

    if ".txt" in extrinFile:
        camera_extrinsics = []
        try:
            with open(f"{extDest}/{extrinFile}", "r") as file:
                content = file.readlines()

                for line in content:
                    con = line.split()
                    camera_extrinsics.append([eval(x) for x in con])

        except FileNotFoundError:
            print(f"Error: The file {extrinFile} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        #print(f"Camera extrinsics: {camera_extrinsics}")
        #camera_extrinsics = [eval(x) for x in camera_extrinsics]

        return camera_extrinsics

def get_intrinsics(dirName):
    intrinsFile = f"{dirName}/intrinsics.txt"

    #print(f"Extracting intrinsics from {dirName}")
    camera_intrinsics = []
    try:
        with open(f"{intrinsFile}", "r") as file:
            content = file.readlines()

            for line in content:
                con = line.split()
                camera_intrinsics += con

    except FileNotFoundError:
        print(f"Error: The file {intrinsFile} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    camera_intrinsics = [eval(x) for x in camera_intrinsics]

    return camera_intrinsics

def get_annotations(dirName):
    matFile = f"{dirName}/seg.mat"
    data = loadmat(matFile)

    names = 'names'
    seglabel = 'seglabel'

    key_rmv = {'__header__', '__version__', '__globals__', }
    data = {k: v for k, v in data.items() if k not in key_rmv}
    #print(f"Extracting intrinsics from {dirName}")

    #current available labels in specific image
    labelList = data[names][0]
    #label for each pixel in image
    pixelLabels = data[seglabel]

    return labelList, pixelLabels

def apply_annotations(color, depth, intrinsics, extrinsics, pcd, labelList, pixelLabels):

    print("Start annotation")

    #print(f"pixelLabels: {len(pixelLabels)}, {len(pixelLabels[0])}")

    #m
    annotated_pixels_2d = []

    for u in range(len(pixelLabels)):
        for v in range(len(pixelLabels[0])):
            annotated_pixels_2d.append((u, v))

    #print(f"annotated_pixels_2d: {len(annotated_pixels_2d)}\n"
    #      f"pcd: {len(pcd.points)}")

    #depthmap data
    depth_map = np.asarray(depth)
    annotated_depths = [depth_map[u, v] for u, v in annotated_pixels_2d]

    #ex_depth = o3d.geometry.Image(depth_map)

    #ex_pcd = o3d.geometry.PointCloud.create_from_depth_image(ex_depth, intrinsics, extrinsics)

    #ex_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    #check if map is correct
    #should show overlapping image, with ex_pcd on top and pcd not visible
    #o3d.visualization.draw_geometries([pcd, ex_pcd], zoom=0.5)

    #print(f"ex_pcd: {len(ex_pcd.points)}")

    #convert 2d coordinates into 3d coordinatos to prepare for labelling
    annotated_points_3d = []
    for i, (u, v) in enumerate(annotated_pixels_2d):
        z = annotated_depths[i] / 1000.0  # Assuming depth is in mm, convert to meters
        x = (u - intrinsics.get_principal_point()[0]) * z / intrinsics.get_focal_length()[0]
        y = (v - intrinsics.get_principal_point()[1]) * z / intrinsics.get_focal_length()[1]
        annotated_points_3d.append(np.round([x, y, z], decimals=4, out=None))

    #round xyz coordinates to 4 decimals out
    #produces collection of points sharing coordinates w
    #contains all points in point cloud plus duplicates
    xyz_coordinates = np.round(np.asarray(pcd.points), decimals=4, out=None)
    #print(f"3d: {annotated_points_3d[:3]}\n"
    #      f"pcd points: {xyz_coordinates[:3]}")

    #find shared points
    present_count = [tuple(item) for item in annotated_points_3d if item in xyz_coordinates]
    #get rid of duplicate points
    common = set(present_count)

    #print(f"Number of items from annotated_points_3 present in xyz_coordinates: {len(common)}")

    #get labels for each pixel
    labels = [item for sublist in pixelLabels for item in sublist]

    labeled_coords = []

    final_labels = []

    for i in range(len(present_count)):
        curr = present_count[i]
        if curr in common and not curr in labeled_coords:
            labeled_coords.append(curr)
            final_labels.append(labels[i])

    '''
    anot = []
    for point in annotated_points_3d:
        temp = []
        for coord in point:
            temp.append(coord.astype(int).item())
        anot.append(temp)
    '''

    #confirm xyz format
    #print(f"labeled_coords: {labeled_coords[:3]} & {len(labeled_coords)}\n"
    #      f"final labels: {final_labels[:3]} & {len(final_labels)}\n")

    labeled_pcd = o3d.geometry.PointCloud()

    labeled_pcd.points = o3d.utility.Vector3dVector(labeled_coords)

    #o3d.visualization.draw_geometries([sample_pcd], zoom=0.5)

    labeled_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return labeled_pcd, labeled_coords, final_labels

def get_rgb(dirName):
    rgbDir = f"{dirName}/image"
    #print(f" paths: {list(os.listdir(rgbDir))}")
    rgbFile = list(os.listdir(rgbDir))[-1]

    #print(f"Extracting rgb ({rgbFile}) from {dirName}")
    try:
        color_path = f"{rgbDir}/{rgbFile}"
        color = Image.open(color_path)
        #print(color)

        color_image = o3d.io.read_image(color_path)

        #color_fin = o3d.geometry.Image(np.array(np.asarray(color_image)[:, :]).astype('uint8'))

    except FileNotFoundError:
        print(f"Error: The file {rgbFile} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return color_image, color.size

def get_depth(dirName):
    depthDir = f"{dirName}/depth"
    depthFile = list(os.listdir(depthDir))[-1]

    #print(f"Extracting depth ({depthFile}) from {dirName}")

    try:
        depth_path = f"{depthDir}/{depthFile}"
        depth = Image.open(depth_path)

        #print(depth)

        depth_image = o3d.io.read_image(depth_path)

        depth_fin = o3d.geometry.Image(np.array(np.asarray(depth_image)[:, :]).astype('uint16'))

    except FileNotFoundError:
        print(f"Error: The file {depthFile} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return depth_image, depth.size

def print_color_depth(color, depth):
    plt.subplot(1, 2, 1)
    plt.title('SUN Format: Color Image')
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.title('SUN Format: Depth Image')
    plt.imshow(depth)
    plt.show()

def generate_pc(path):

    #files = clean_filename(path)
    #print(f"Path being passed to Instrinsics and extrinsics: {path}\n")
    intrins_data = get_intrinsics(path)
    #print(f"Intrinsics extracted")
    extrins_data = get_extrinsics(path)
    #print(f"Extrinsics extracted")

    color_image, color_size = get_rgb(path)
    depth_image, depth_size = get_depth(path)

    #print("Generating RGBD image")
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
        color_image, depth_image)

    #print_color_depth(rgbd_image.color, rgbd_image.depth)
    #print(f"{color_size == depth_size}")

    if color_size != depth_size:
        return f"Color dimensions {color_size} do not match Depth dimensions {depth_size}"

    #print(f"Path name:{path}")

    width = color_size[0]
    height = color_size[1]
    fx = intrins_data[0]
    fy = intrins_data[4]
    cx = intrins_data[2]
    cy = intrins_data[5]

    #print(f"Width: {width}\nHeight: {height}\nfx,fy: {fx} -- {fy}\ncx, cy: {cx} -- {cy}\n")

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrinsics = np.eye(4)
    extrinsics[:3, :] = extrins_data

    #print("Generating Point Cloud")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics,
        extrinsics
    )

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # uncomment below to preview point cloud
    #o3d.visualization.draw_geometries([pcd], zoom=0.5)
    #print(pcd)
    #print(f"\npoint lentgh {len(pcd.points)}\n")

    #uncomment below to preview point cloud
    #o3d.visualization.draw_geometries([pcd])

    # o3d.visualization.draw_geometries([pcd])
    #print("Overwriting normals")
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    pcd.estimate_normals()

    #o3d.io.write_point_cloud(f"{path}/pointcloud/{filename}.ply", pcd, write_ascii=True)
    # o3d.visualization.draw_plotly([pcd], width=1000, height=1000)
    #print("Point Cloud generated")
    return pcd

def generate_pcd_annotation(path):

    #files = clean_filename(path)
    #print(f"Path being passed to Instrinsics and extrinsics: {path}\n")
    intrins_data = get_intrinsics(path)
    #print(f"Intrinsics extracted")
    extrins_data = get_extrinsics(path)
    #print(f"Extrinsics extracted")

    color_image, color_size = get_rgb(path)
    depth_image, depth_size = get_depth(path)

    #print("Generating RGBD image")
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
        color_image, depth_image)

    #print_color_depth(rgbd_image.color, rgbd_image.depth)
    #print(f"{color_size == depth_size}")

    if color_size != depth_size:
        return f"Color dimensions {color_size} do not match Depth dimensions {depth_size}"

    #print(f"Path name:{path}")

    width = color_size[0]
    height = color_size[1]
    fx = intrins_data[0]
    fy = intrins_data[4]
    cx = intrins_data[2]
    cy = intrins_data[5]

    #print(f"Width: {width}\nHeight: {height}\nfx,fy: {fx} -- {fy}\ncx, cy: {cx} -- {cy}\n")

    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    extrinsics = np.eye(4)
    extrinsics[:3, :] = extrins_data

    #print("Generating Point Cloud")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics,
        extrinsics
    )

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # uncomment below to preview point cloud
    #o3d.visualization.draw_geometries([pcd], zoom=0.5)
    #print(pcd)

    labelList, pixelLabels = get_annotations(path)

    labeled_pcd,labeled_points, labels = apply_annotations(color_image, depth_image, intrinsics, extrinsics, pcd, labelList, pixelLabels)

    #print(f"\npoint lentgh {len(pcd.points)}\n")

    #uncomment below to preview point cloud
    #o3d.visualization.draw_geometries([pcd])

    # o3d.visualization.draw_geometries([pcd])
    #print("Overwriting normals")
    labeled_pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    labeled_pcd.estimate_normals()

    #o3d.io.write_point_cloud(f"{path}/pointcloud/{filename}.ply", pcd, write_ascii=True)
    # o3d.visualization.draw_plotly([pcd], width=1000, height=1000)
    #print("Point Cloud generated")
    return labeled_pcd, labeled_points, labels

def generate_bb():
    with open("annotations.json") as f:
        annotations = json.load(f)

    for box_data in annotations:
        position = box_data["geometry"]["position"]  # Example: Accessing position data
        orientation = box_data["geometry"]["orientation"]
        dimensions = box_data["geometry"]["dimensions"]
        class_name = box_data["className"]

def get_rand_idx(n, limit):

    idxList = []

    while len(idxList) < n:
        idx = random.randint(0, limit)
        if not idx in idxList:
            idxList.append(idx)

    return idxList


def load_data(n, annot = False):
    #print("Loading data")
    trains_split = get_trainsplit()
    val_split = get_valsplit()

    train_idc = get_rand_idx(n, len(trains_split))
    val_idc = get_rand_idx(n, len(val_split))

    print(f"train_icd: {train_idc}")
    print(f"val_icd: {val_idc}")

    train_paths = [trains_split[i] for i in train_idc]
    val_paths = [val_split[i] for i in val_idc]
    train_pc = []
    train_annotated = []
    val_pc = []
    val_annotated = []


    start_time = time.time()
    #print("Generating Training Point Clouds")

    if annot:
        for path in train_paths:
            pcd, annotated, labels = generate_pcd_annotation(path)
            train_pc.append(pcd)
            train_annotated.append([annotated, labels])

        #print("Generating Validation Point Clouds")
        for path in val_paths:
            pcd, annotated, labels = generate_pcd_annotation(path)
            val_pc.append(pcd)
            val_annotated.append([annotated, labels])

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        return train_pc, train_annotated, val_pc, val_annotated

    else:
        for path in train_paths:
            pcd = generate_pc(path)
            train_pc.append(pcd)

        # print("Generating Validation Point Clouds")
        for path in val_paths:
            pcd = generate_pc(path)
            val_pc.append(pcd)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time

        return train_pc, val_pc

    #print("Point Clouds generated")

    #print("Load Data Completed")


def load_specific(path, annotation = False):
    prefix = "data/"
    #print("Loading data")
    start_time = time.time()
    print("Generating Training Point Clouds")

    if annotation:
        pcd, labeled_points, labels = generate_pcd_annotation(prefix + path)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        return pcd, labeled_points, labels

    else:
        pcd = generate_pc(prefix + path)

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")

        return pcd


def main():
    pass
    #gt = get_groundtruth()
    #print(gt)
    train_pc, val_pc = load_data(3)

    print(f"train_pc: {train_pc}\n"
          f"val_pc: {val_pc}\n")

    pcd, annot, labels = load_data(3, True)

    print(f"pcd: {pcd}\n"
          f"annot: {annot}\n"
          f"labels: {labels}")

    #points = list(pcd.points)


if __name__ == "__main__":
    main()