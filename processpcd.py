import subprocess
import sys
import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
import pandas as pd

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
    return train_paths

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
    return val_paths

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
    #print(rgbd_image)

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

    '''
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(color_image)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(depth_image)
    plt.show()
    '''

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

def generate_bb():
    with open("annotations.json") as f:
        annotations = json.load(f)

    for box_data in annotations:
        position = box_data["geometry"]["position"]  # Example: Accessing position data
        orientation = box_data["geometry"]["orientation"]
        dimensions = box_data["geometry"]["dimensions"]
        class_name = box_data["className"]

def load_data():
    #print("Loading data")
    trains_paths = get_trainsplit()
    val_paths = get_valsplit()

    train_pc = []
    val_pc = []
    start_time = time.time()
    #print("Generating Training Point Clouds")
    for path in trains_paths:
        pcd = generate_pc(path)
        train_pc.append(pcd)

    #print("Generating Validation Point Clouds")
    for path in val_paths:
        pcd = generate_pc(path)
        val_pc.append(pcd)

    #print("Point Clouds generated")

    #print("Load Data Completed")

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    return train_pc, val_pc

def load_specific(path):
    prefix = "data/"
    #print("Loading data")
    start_time = time.time()
    #print("Generating Training Point Clouds")
    pcd = generate_pc(prefix + path)

    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    return pcd

def main():

    gt = get_groundtruth()
    print(gt)
    #load_data()

if __name__ == "__main__":
    main()