""" This module contains utility functions for key components of MAGiC-SLAM pipeline. """
import random

import numpy as np
import open3d as o3d
import roma
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.entities.gaussian_model import GaussianModel
from src.entities.losses import l1_loss, ssim
from src.utils import utils
from src.utils.utils import find_submap, np2ptcloud, coarse_registration

from argparse import ArgumentParser
from src.entities.arguments import OptimizationParams

import copy
import time

class Registration(object):
    """ A class to store the registration information between two submaps. """
    def __init__(self, source_agent_id: int, source_frame_id: int, target_agent_id: int, target_frame_id: int) -> None:
        self.source_agent_id = source_agent_id
        self.source_frame_id = source_frame_id
        self.target_agent_id = target_agent_id
        self.target_frame_id = target_frame_id
        self.source_transformation = np.eye(4)
        self.target_transformation = np.eye(4)
        self.init_transformation = np.eye(4)
        self.transformation = np.eye(4)
        self.inlier_rmse = 100.0
        self.fitness = 0.0


def refine_map(gaussian_model, agents_datasets: dict, agents_keyframe_ids: dict, agents_c2ws: dict, iterations=3000):
    """ Refine a Gaussian model using keyframes from agents' datasets: Section 3.4
    Args:
        gaussian_model: The Gaussian model.
        agents_datasets: The agents' datasets (agent_id: str -> dataset : Dataset)
        agents_keyframe_ids: The agents' keyframe IDs (agent_id: str -> keyframe_ids : np.ndarray)
        agents_c2ws: The agents' camera-to-world matrices (agent_id: str -> c2ws : np.ndarray)
        iterations: The number of iterations to refine the map.
    Returns:
        gaussian_model: The refined Gaussian model.
    """
    print("Refining map")
    for iteration in tqdm(range(iterations)):
        agent_id = random.choice(list(agents_datasets.keys()))
        sample_id = random.randint(0, len(agents_keyframe_ids[agent_id]) - 1)
        render_data = agents_datasets[agent_id].get_render_frame(
            agents_keyframe_ids[agent_id][sample_id],
            np.linalg.inv(agents_c2ws[agent_id][sample_id]))
        gt_color, gt_depth = render_data["gt_color"], render_data["gt_depth"]

        render_dict = utils.render_gaussian_model(gaussian_model, render_data["render_settings"])
        color_loss = 0.8 * l1_loss(render_dict["color"], gt_color) + 0.2 * (1.0 - ssim(render_dict["color"], gt_color))

        depth_lss = l1_loss(render_dict["depth"], gt_depth)

        loss = color_loss + depth_lss

        loss.backward()

        with torch.no_grad():
            radii = render_dict["radii"]
            visibility_filter = radii > 0
            gaussian_model.max_radii2D[visibility_filter] = torch.max(
                gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])

            if iteration > iterations * 0.2 and iteration < iterations * 0.8:
                prune_mask = (gaussian_model.get_opacity() < 0.005).squeeze()
                gaussian_model.prune_points(prune_mask)

            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            # gaussian_model.update_learning_rate(iteration)
    return gaussian_model


def merge_submaps(agents_submaps: dict, agents_kf_ids: dict, agents_opt_kf_c2ws: dict, opt_args):
    """ Merge submaps from agents in a coarse manner: Section 3.4
    Args:
        agents_submaps: A dictionary of agent submaps.
        agents_kf_ids: A dictionary of agent keyframe IDs.
        agents_opt_kf_c2ws: A dictionary of agent optimized camera-to-world matrices.
        opt_args: The optimization arguments.
    Returns:
        merged_map: The merged Gaussian model.
    """
    merged_map = GaussianModel(0)
    merged_map.training_setup(opt_args)
    device = "cuda"

    print("Merging submaps")
    for agent_id in tqdm(sorted(agents_submaps.keys())):

        for _, submap in enumerate(agents_submaps[agent_id][::-1]):

            xyz = submap["gaussian_model_params"]["xyz"].to(device)
            rotations = submap["gaussian_model_params"]["rotation"].to(device)
            features_dc = submap["gaussian_model_params"]["features_dc"].to(device)
            features_rest = submap["gaussian_model_params"]["features_rest"].to(device)
            opacity = submap["gaussian_model_params"]["opacity"].to(device)
            scaling = submap["gaussian_model_params"]["scaling"].to(device)

            kf_mask = agents_kf_ids[agent_id] == submap["submap_start_frame_id"]
            submap_opt_c2w = agents_opt_kf_c2ws[agent_id][kf_mask][0]
            submap_c2w = submap["submap_c2ws"][0]
            delta = utils.np2torch(submap_opt_c2w @ np.linalg.inv(submap_c2w), device=device)

            opt_xyz = xyz @ delta[:3, :3].T + delta[:3, 3]

            rotation_matrices = roma.unitquat_to_rotmat(rotations)
            opt_rotations = delta[:3, :3][None] @ rotation_matrices
            opt_rotations = roma.rotmat_to_unitquat(opt_rotations)

            merged_map.densification_postfix(
                opt_xyz,
                features_dc,
                features_rest,
                opacity,
                scaling,
                opt_rotations)

    return merged_map


def apply_pose_correction(submap_optimized_poses: dict, agents_submaps: dict) -> dict:
    """ Applies the pose correction to the keyframes of the agents.
        The pose graph consists of the first poses of each sub-map.
        After optimization, the correction between the optimized pose of a sub-map
        is applied to all the keyframe poses within that sub-map.
    Args:
        submap_optimized_poses: The optimized poses of the submaps.
        agents_submaps: The submaps of the agents.
    Returns:
        agents_corrected_kf_poses: A dictionary containing the corrected keyframe poses of the agents
    """
    agents_corrected_kf_poses = {}
    for agent_id in sorted(agents_submaps.keys()):
        opt_kf_poses = []
        for i, submap in enumerate(agents_submaps[agent_id]):
            submap_kf_ids = submap["keyframe_ids"] - submap["submap_start_frame_id"]
            delta = submap_optimized_poses[agent_id][i] @ np.linalg.inv(submap["submap_c2ws"][0])
            for pose in submap["submap_c2ws"][submap_kf_ids]:
                opt_kf_poses.append(delta @ pose)
        agents_corrected_kf_poses[agent_id] = np.array(opt_kf_poses)
    return agents_corrected_kf_poses


def register_submaps(agents_submaps: dict, registration: Registration):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    source_cloud = agents_submaps[registration.source_agent_id]
    source_submap = find_submap(registration.source_frame_id, agents_submaps[registration.source_agent_id])
    target_submap = find_submap(registration.target_frame_id, agents_submaps[registration.target_agent_id])

    source_cloud = np2ptcloud(source_submap["point_cloud"][:, :3], source_submap["point_cloud"][:, 3:] / 255.0)
    target_cloud = np2ptcloud(target_submap["point_cloud"][:, :3], target_submap["point_cloud"][:, 3:] / 255.0)

    source_cloud.estimate_normals()
    target_cloud.estimate_normals()

    distance_threshold = 0.005
    fine_alignment = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, registration.init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500,
                                                          relative_fitness=1e-9, relative_rmse=1e-9))

    registration.transformation = fine_alignment.transformation
    registration.fitness = fine_alignment.fitness
    registration.inlier_rmse = fine_alignment.inlier_rmse

    return registration


def register_agents_submaps(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 5) -> list:
    """ Register the submaps of the agents in parallel.
    Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        max_threads: The maximum number of threads to use.
    """
    registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
        agents_submaps, registration) for registration in registrations)
    return registrations

def register_submaps_depth(agents_submaps: dict, registration: Registration):
    """ Register two submaps using ICP.
    Args:
        agents_submaps: A dictionary of agent submaps.
        registration: The registration object.
    Returns:
        registration: The registration object with the transformation and fitness updated.
    """

    source_submap = find_submap(registration.source_frame_id, agents_submaps[registration.source_agent_id])
    target_submap = find_submap(registration.target_frame_id, agents_submaps[registration.target_agent_id])

    source_cloud = utils.get_pcd_from_rgbd(source_submap)
    target_cloud = utils.get_pcd_from_rgbd(target_submap)

    if source_submap['agent_id'] != target_submap['agent_id']:
        registration.init_transformation = coarse_registration(source_cloud, target_cloud)

    source_cloud.estimate_normals()
    target_cloud.estimate_normals()

    distance_threshold = 0.005
    fine_alignment = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, registration.init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500,
                                                          relative_fitness=1e-9, relative_rmse=1e-9))

    # draw_registration_result(source_cloud, target_cloud, fine_alignment.transformation)

    registration.transformation = fine_alignment.transformation
    registration.fitness = fine_alignment.fitness
    registration.inlier_rmse = fine_alignment.inlier_rmse
    # registration.info_matrix = get_information_matrix(
    #    source_cloud, target_cloud, distance_threshold, registration.transformation, fine_alignment.fitness, threshold=0.5)

    return registration

def move_to_device(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    elif isinstance(obj, set):
        return {move_to_device(v, device) for v in obj}
    else:
        return obj # Leave non-tensor types unchanged

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def get_information_matrix(source_cloud, target_cloud, distance_threshold, transformation, fitness, threshold=0.5):
    """ Compute the information matrix for the registration.
    Args:
        source_cloud: The source point cloud.
        target_cloud: The target point cloud.
        transformation: The transformation matrix.
    Returns:
        information_matrix: The information matrix.
    """
    if not isinstance(source_cloud, o3d.t.geometry.PointCloud):
        source_cloud_t = o3d.t.geometry.PointCloud.from_legacy(source_cloud)
    else:
        source_cloud_t = source_cloud

    if not isinstance(target_cloud, o3d.t.geometry.PointCloud):
        target_cloud_t = o3d.t.geometry.PointCloud.from_legacy(target_cloud)
    else:
        target_cloud_t = target_cloud
    # Now get_information_matrix will work
    if fitness > threshold:
        info_matrix = o3d.t.pipelines.registration.get_information_matrix(
            source_cloud_t, target_cloud_t, distance_threshold, transformation)
    else:
        info_matrix = o3d.core.Tensor.zeros((6, 6), dtype=o3d.core.float64)
    
    # Reorder from Open3D [tx,ty,tz,rx,ry,rz] to GTSAM [rx,ry,rz,tx,ty,tz]
    reorder_indices = [3, 4, 5, 0, 1, 2]
    info_matrix = info_matrix[np.ix_(reorder_indices, reorder_indices)].numpy()

    return info_matrix

def register_agents_submaps_depth(agents_submaps: dict, registrations: list,
                            registration_method, max_threads: int = 1)-> list:
    """ Args:
        agents_submaps: A dictionary of agent submaps with (agent_id: int -> submaps: list)
        registrations: A list of Registration objects.
        registration_method: The registration method to use.
        intrinsics: The camera intrinsics to use for depth registration.
    """
    if max_threads < 2:
        registrations = [registration_method(agents_submaps, registration) for registration in registrations]
    else:
       registrations = Parallel(n_jobs=max_threads)(delayed(registration_method)(
           agents_submaps, registration) for registration in registrations)
    return registrations