from operator import invert

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_pcd(filename):
    pcd = o3d.io.read_point_cloud("../data/" + filename)
    return pcd

def preprocess(pcd, nn, std_multiplier):
    pcd_center = pcd.get_center()
    pcd.translate(-pcd_center)


    #filter outliers
    filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)
    outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
    outliers.paint_uniform_colour([1, 0, 1])
    filtered_pcd = filtered_pcd[0]
    o3d.visualization.draw_geometries([filtered_pcd, outliers])

    #downsample
    size = 0.01
    pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size=size)
    o3d.visualization.draw_geometries([pcd_downsampled])

    #normal estimation
    nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
    radius_normals = nn_distance*4
    pcd_downsampled.estimate_normals(
        search_param = o3d.geometry.KDTreesSearchParamHybrid(radius = radius_normals, max_nn = 16),
        fast_normal_computation = True)
    pcd_downsampled.paint_uniform.color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([pcd_downsampled, outliers])

    #get from pose data in dataset
    front = []
    lookat = []
    up = []
    zoom = []

    pcd = pcd_downsampled
    o3d.visualization.draw_geometries([pcd], zoom=zoom,
      front=front,
      lookat=lookat,
      up=up)

    #RANSAC Planar Segmentation
    pt_to_plane_dist = 0.5 #adjust if visualization shows the wrong threshold
    plane_model, inliers = pcd.segment_plame(distance_threshold=pt_to_plane_dist,
     ransac_n=3,
     num_iterations=1000)
    [a,b,c,d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:2f} = 0")

    inliear_cloud = pcd.select_by_index(inliers)
    outliear_cloud = pcd.select_by_index(inliers, invert=True)
    inliear_cloud.paint_uniform_color([1.0, 0, 0])
    outliear_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    o3d.visualization.draw_geometries([inliear_cloud, outliear_cloud],
      zoom=zoom,
      front=front,
      lookat=lookat,
      up=up)

    #multi order ransac
    max_plane_idx = 6 #number of planes in pointcloud/intended image, eg. number of walls, floors, etc...
    pt_to_plane_dist = .02 #experiment

    segment_models = ()
    segments = ()
    rest = pcd

    for i in range(max_plane_idx):
        colors = plt.get_cmap("tab20")(i)
        segment_models[i]. inliers = rest.segment_plane(distance_threshold = pt_to_plane_dist,
            ransac_n = 3,
            num_iterations = 100)
        segments[i] = rest.select_by_index(inliers)
        segments[i].paint_uniform_color(list(colors[:3]))
        rest - reset.select_by_index(inliers, invert=True)
        print("pass",i,"/",max_plane_idx,"done.")

        o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest],
          zoom=zoom,
          front=front,
          lookat=lookat,
          up=up)

    #dbscan

    labels = np.array(rest.cluster_dbscan(eps=0.05, min_points = 5))
    max_label = labels.max()
    print(f"point cloud had {max_label + 1} clusters")
    colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:,:3])

    o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest],
                                      zoom=zoom,
                                      front=front,
                                      lookat=lookat,
                                      up=up)