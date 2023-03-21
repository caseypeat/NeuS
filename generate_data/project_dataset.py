import cv2
import numpy as np
import os
import json
import open3d as o3d

from tqdm import tqdm

from models.dataset import load_K_Rt_from_P



def show_cameras_and_pointclouds(cameras, pointclouds):
    """
    Show camera positions and toggle projected point clouds using Open3D.

    Parameters:
        - cameras (list of numpy arrays): list of camera positions (3x1 numpy arrays)
        - pointclouds (list of numpy arrays): list of point cloud data (Nx3 numpy arrays)
    """

    o3d.visualization.draw(pointclouds)

    # Create an Open3D visualizer object
    # vis = o3d.visualization.Visualizer()

    # # Add cameras to the visualizer
    # for camera in cameras:
    #     vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    #     vis.add_geometry(o3d.geometry.TriangleMesh.create_camera_marker(camera=camera))

    # Add point clouds to the visualizer
    # for i, pointcloud in enumerate(pointclouds):
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(pointcloud)
    #     pc.paint_uniform_color([i/len(pointclouds), 0, 1-i/len(pointclouds)])
    #     vis.add_geometry(pc)

    # Create a key callback function to toggle point clouds
    # def toggle_pointcloud_callback(vis, key):
    #     if key == ord(" "):
    #         for i in range(len(pointclouds)):
    #             vis.get_geometry_list()[len(cameras) + i].visible = not vis.get_geometry_list()[len(cameras) + i].visible
    #         vis.update_renderer()

    # Set the key callback function
    # vis.register_key_callback(ord(" "), toggle_pointcloud_callback)

    # Run the visualizer
    # vis.run()
    # vis.destroy_window()


def project_image_to_pointcloud(image, P, distance):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)

    c = np.stack([xx.flatten(), yy.flatten(), zz.flatten(), zz.flatten()], axis=0)  # [x, y, 3]

    # P_inv = np.linalg.inv(P)[]

    C = np.linalg.inv(P) @ c

    color = image.reshape(-1, 3).astype(np.float32) / 255
    color = color[:, ::-1]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(C[:3, :].T)
    pcl.colors = o3d.utility.Vector3dVector(color)

    return pcl
    


if __name__ == "__main__":

    # filepath = "../../data/plant_and_food/tonemapped/2022-08-31/jpeg_north/refined_5cm.json"
    # filepath = os.path.join("generate_data", filepath)

    # with open(filepath, 'r') as file:
    #     data = json.load(file)

    # dirpath = "../NeRF-Reconstruction/outputs/processed_data/2"
    dirpath = "./public_data/thin_cube"

    # cameras_path = os.path.join(dirpath, "cameras.npz")
    cameras_path = os.path.join(dirpath, "cameras_sphere.npz")

    output_path = os.path.join(dirpath, "camera.json")

    camera_dict = np.load(cameras_path)

    image_filenames = sorted(os.listdir(os.path.join(dirpath, "image")))
    n_images = len(image_filenames)

    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics = []
    poses = []

    images = []
    pointclouds = []

    for i, (scale_mat, world_mat, image_filename) in tqdm(enumerate(zip(scale_mats_np, world_mats_np, image_filenames))):
        P_ = world_mat @ scale_mat
        P = P_[:3, :4]
        intrinsic, pose = load_K_Rt_from_P(None, P)

        image = cv2.imread(os.path.join(dirpath, "image", image_filenames[i]))
        
        intrinsics.append(intrinsic)
        poses.append(pose)

        images.append(image)

        pointcloud = project_image_to_pointcloud(image, P_, 0.25)
        pointclouds.append(pointcloud)

        if i > 2:
            break


    
    # images = []
    # pointclouds = []
    # for image_filename in image_filenames:
    #     image = cv2.imread(os.path.join(dirpath, "image", image_filenames[0]))
    #     images.append(image)

    #     pointcloud = project_image_to_pointcloud(image)
    #     pointclouds.append(pointcloud)

    

    show_cameras_and_pointclouds(poses, pointclouds)

    # cameras = {}
    # cameras['image_sets'] = {}
    # cameras['image_sets']['rgb'] = []

    # cameras['pose_sets'] = {}
    # cameras['pose_sets']['rig_pose'] = []
    # cameras['pose_sets']['ref_pose'] = []

    # cameras['camera_poses'] = {}
    # cameras['camera_poses']['camera'] = {}
    # cameras['camera_poses']['camera']['R'] = np.eye(3).tolist()
    # cameras['camera_poses']['camera']['T'] = np.array([0, 0, 0]).tolist()

    # cameras['cameras'] = {}
    # cameras['cameras']['camera'] = {}
    # cameras['cameras']['camera']['K'] = intrinsics[0][:3, :3].tolist()
    # cameras['cameras']['camera']['dist'] = [0, 0, 0, 0, 0]
    # cameras['cameras']['camera']['image_size'] = [image.shape[1], image.shape[0]]


    # for intrinsic, pose, image_filename in zip(intrinsics, poses, image_filenames):
    #     image_filepath = os.path.join('image', image_filename)

    #     cameras['pose_sets']["rig_pose"].append(pose.tolist())
    #     cameras['pose_sets']["ref_pose"].append(np.eye(4).tolist())

    #     cameras['image_sets']['rgb'].append({'camera': image_filepath})

    # with open(output_path, 'w') as file:
    #     json.dump(cameras, file)
