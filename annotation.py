import open3d as o3d
import numpy as np
import  bb
import processpcd as pr
from scipy.io import loadmat

# Load RGBD images (replace with your actual image paths)
color_raw = o3d.io.read_image("path/to/your/color_image.png")
depth_raw = o3d.io.read_image("path/to/your/depth_image.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, convert_rgb_to_intensity=False
)

# Define camera intrinsics (replace with your camera's parameters)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=640, height=480, fx=525.0, fy=525.0, cx=319.5, cy=239.5
)

# Create point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# --- Apply 2D annotations to 3D point cloud ---
# Assuming you have a list of annotated 2D pixel coordinates (e.g., from a segmentation mask)
annotated_pixels_2d = [(100, 150), (101, 150), ...] # Example: list of (u,v) tuples

# Get depth values for annotated pixels
depth_map = np.asarray(depth_raw)
annotated_depths = [depth_map[v, u] for u, v in annotated_pixels_2d]

# Project 2D annotated pixels to 3D points
annotated_points_3d = []
for i, (u, v) in enumerate(annotated_pixels_2d):
    z = annotated_depths[i] / 1000.0  # Assuming depth is in mm, convert to meters
    x = (u - intrinsic.get_principal_point()[0]) * z / intrinsic.get_focal_length()[0]
    y = (v - intrinsic.get_principal_point()[1]) * z / intrinsic.get_focal_length()[1]
    annotated_points_3d.append([x, y, z])

# Now you have the 3D points corresponding to your 2D annotations.
# You can use these 3D points to:
# - Create a separate point cloud for the annotated region.
# - Color the corresponding points in the original point cloud.
# - Generate 3D bounding boxes or segmentation masks in 3D space.