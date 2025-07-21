import open3d as o3d
import numpy as np

# Create a sample point cloud (or load from a file)
pcd = o3d.geometry.PointCloud()
points = np.random.rand(1000, 3) * 10
pcd.points = o3d.utility.Vector3dVector(points)

# DBSCAN parameters
eps = 0.5
min_points = 10

# Apply DBSCAN
labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

# Visualize the clusters
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Black for noise
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcd])