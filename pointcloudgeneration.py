import subprocess
import sys
import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image


#TRAIN_PATHS = ["../data/train/7-scenes-heads", "../data/train/sun3d-brown_cogsci_1-brown_cogsci_1",
#         "../data/train/sun3d-harvard_c11-hv_c11_2"]

TRAIN_PATHS = ["../data/train/7-scenes-heads", "../data/train/sun3d-brown_cogsci_1-brown_cogsci_1",
         "../data/train/sun3d-harvard_c11-hv_c11_2"]

PROC_PATHS = ["../data/a100_dome", "../data/a100_dome_vo", "../data/a200_met", "../data/box_met", "../data/p2at_met"]

TEST_PATHS = ["../data/test/sun3d-hotel_umd-maryland_hotel3"]

SUBPATHS = ["seq-01", "seq-02"]

INTRINSICS = "camera-intrinsics.txt"

FORMATS = [".color.png", ".depth.png"]

DEST = "pointclouds"

def get_xyz(paths):
    for path in paths:
        entries = os.listdir(path)
        #print(f"path: {path}\n"
        #      f"entries: {entries}")

        destination = f"{path}/{DEST}/"

        print(f"Destination: {destination}")
        print(os.listdir(destination))

        for entry in entries:
            if entry == "pointclouds":
                continue
            src = f"{path}/{entry}"
            #print(f"file entry: {entry}")
            if not os.path.isdir(src):
                continue
            files = list(os.listdir(src))

            xyz = files[[files.index(file) for file in files if ".xyz" in file][0]]

            print(f"adding {xyz} to {destination}")
            shutil.copy(f"{src}/{xyz}", destination)


def get_train_pcd(paths):

    for path in paths:
        entries = os.listdir(path)

        camera_intrinsics = []

        if INTRINSICS in entries:
            #print("True")
            try:
                with open(f"{path}/{INTRINSICS}", "r") as file:
                    content = file.readlines()

                    for line in content:
                        con = line.split()
                        camera_intrinsics += con

            except FileNotFoundError:
                print(f"Error: The file {INTRINSICS} was not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

        camera_intrinsics = [eval(x) for x in camera_intrinsics]

        #print(f"intrinsics: {camera_intrinsics}")

        for entry in entries:
            #print(f"entry: {entry}")
            if entry in SUBPATHS:
                generate_pc(f"{path}/{entry}", camera_intrinsics)

def get_test_pcd():

    for path in PATHS:
        entries = os.listdir(path)

        camera_intrinsics = []

        if INTRINSICS in entries:
            #print("True")
            try:
                with open(f"{path}/{INTRINSICS}", "r") as file:
                    content = file.readlines()

                    for line in content:
                        con = line.split()
                        camera_intrinsics += con

            except FileNotFoundError:
                print(f"Error: The file {INTRINSICS} was not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

        camera_intrinsics = [eval(x) for x in camera_intrinsics]

        #print(f"intrinsics: {camera_intrinsics}")

        for entry in entries:
            #print(f"entry: {entry}")
            if entry in SUBPATHS:
                generate_pc(f"{path}/{entry}", camera_intrinsics)

def generate_pc(path, intrin):

    files = clean_filename(path)

    #print(f"Path name:{path}")
    allowed_ext = [".png", ".txt"]

    files = [file for file in files if file[-4:] in allowed_ext]

    for i in range(0, len(files), 3):

        filename = files[i][:12]

        if i + 1 >= len(files):
            return
        color_path = f"{path}/{files[i]}"
        depth_path = f"{path}/{files[i + 1]}"
        color = Image.open(color_path)
        depth = Image.open(depth_path)

        #print(f"Color Image dimensions: {color.size}")
        #print(f"Depth Image dimensions: {depth.size}")


        if color.size == depth.size:
            width = color.size[0]
            height = color.size[1]
            fx = intrin[0]
            fy = intrin[4]
            cx = intrin[2]
            cy = intrin[5]

            intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            color_image = o3d.io.read_image(color_path)
            depth_image = o3d.io.read_image(depth_path)

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsics
            )

            #o3d.visualization.draw_geometries([pcd])

            pcd.normals = o3d.utility.Vector3dVector(np.zeros(
                (1, 3)))  # invalidate existing normals

            pcd.estimate_normals()

            o3d.io.write_point_cloud(f"{path}/{DEST}/{filename}.ply", pcd, write_ascii=True)
            #o3d.visualization.draw_plotly([pcd], width=1000, height=1000)

def clean_filename(path):
    #print(f"Cleaning")
    files = sorted(list(os.listdir(path)))

    for filename in files:

        old_file_path = os.path.join(path, filename)

        # Implement your renaming logic here.
        # For example, to add a prefix:
        if ".color" in filename:
            new_filename = filename.replace(".color", "-color")
        elif ".depth" in filename:
            new_filename = filename.replace(".depth", "-depth")
        elif ".pose" in filename:
            new_filename = filename.replace(".pose", "-pose")
        elif "-png" in filename:
            new_filename = filename.replace("-png", ".png")
            #print(f"new filename: {new_filename}")
        elif "-txt" in filename:
            new_filename = filename.replace("-txt", ".txt")
        else:
            continue

        new_file_path = os.path.join(path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        #print(f"Renamed '{filename}' to '{new_filename}'")

    return sorted(list(os.listdir(path)))

def del_xyzi(paths):
    #print(f"Cleaning")
    #print(path)
    for path in paths:
        files = os.listdir(path)
        print(files)

        for filename in files:
            if ".xyzi" in filename:
                os.remove(f"{path}/pointclouds/{filename}")

if __name__ == "__main__":
    #get_train_pcd(PATHS)
    #get_xyz(PROC_PATHS)
    #del_xyzi(PROC_PATHS)