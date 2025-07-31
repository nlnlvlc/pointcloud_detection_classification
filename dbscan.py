import open3d as o3d
import numpy as np
import bb
import random
import processpcd as pr
import matplotlib.pyplot as plt

# Create a sample point cloud (or load from a file)

def run_dbscan(pcd_idx = 100, eps = .1, min_points = 10, print_progress = False, visualize = False):
    groundtruth = pr.get_groundtruth()
    pcd = pr.load_specific(groundtruth['sequenceName'][pcd_idx])

    # Apply DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

    max_label = labels.max()
    num_clusters = max_label + 1  # Add 1 because labels are 0-indexed
    #print(f"The point cloud has {num_clusters} clusters (excluding noise).\n")

    if visualize:
        visualized_dbscan(labels, pcd)

    return num_clusters

def visualized_dbscan(labels, pcd):
    # Visualize the clusters
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Black for noise
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])

def get_rand_idx():

    idx = random.randint(0, 64793)

    return idx

def run_experiment():
    epsList = [.1, .01, .05]
    mpList = [10, 50, 100]

    idxList = []

    while len(idxList) < 3:
        idx = get_rand_idx()

        if idx not in idxList:
            idxList.append(idx)

    idx_dict = {}

    for idx in idxList:
        for eps in epsList:
            for mp in mpList:
                idx_dict.update({(idx, eps, mp): None})

    #print(f"idx_dict: {idx_dict}")

    for key in idx_dict.keys():
        idx = key[0]
        eps = key[1]
        mp = key[2]

        clusters = run_dbscan(idx, eps, mp)

        idx_dict[key] = clusters

    return idx_dict

def main():
    #run_dbscan(799, .04, 60)
    experiment = run_experiment()

    print(experiment)

if __name__ == "__main__":
    main()