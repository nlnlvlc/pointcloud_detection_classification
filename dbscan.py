import open3d as o3d
import numpy as np
import  bb
import processpcd as pr
import matplotlib.pyplot as plt

# Create a sample point cloud (or load from a file)

groundtruth = bb.get_groundtruth()
pcd = pr.load_specific(groundtruth['sequenceName'][500])

# DBSCAN parameters
eps = 0.05
min_points = 75

# Apply DBSCAN
labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

# Visualize the clusters
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Black for noise
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd])