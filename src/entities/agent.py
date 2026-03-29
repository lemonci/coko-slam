""" This module includes the agent entity, which is responsible for controlling Mapper and Tracker.
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.loop_detection.feature_extractors import get_feature_extractor
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.utils import (clone_obj, get_render_settings, np2torch,
                             ptcloud2numpy, render_gaussian_model, setup_seed,
                             torch2np, tensor_to_jpeg_bytes_cv2, depth_to_compressed_bytes)
from src.utils.vis_utils import COLORS_ANSI
import faiss
import open3d as o3d
import torch.nn.functional as F
import copy


class Agent(object):

    def __init__(self, agent_id: int, run_info: dict, config: dict) -> None:

        self.device = "cuda"
        self.config = deepcopy(config)
        self.agent_id = agent_id
        self.run_info = run_info

        self.output_path = Path(config["data"]["output_path"]) / f"agent_{self.agent_id}"
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        agent_input_path = sorted(Path(config["data"]["input_path"]).glob("*"))[self.agent_id]
        self.config["data"]["input_path"] = str(agent_input_path)
        self.dataset = get_dataset(config["dataset_name"])({**self.config["data"], **self.config["cam"], **self.config["submap"]})

        # Add keyframing configurations
        self.active_keyframing = self.config['keyframing']['active_keyframing']
        self.keyframing_threshold = self.config['keyframing']['keyframing_threshold']
        # Add submapping configurations
        self.active_submapping = self.config['submapping']['active_submapping']
        self.submapping_threshold = self.config['submapping']['submapping_threshold']

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.gt_c2ws = np2torch(np.array(self.dataset.poses))
        self.keyframe_ids = []
        self.submap_features = {}

        save_dict_to_yaml(self.config, "config.yaml", directory=self.output_path)

        self.submap_id = 0
        self.submap_start_frame_ids = [0]

        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.log_color = COLORS_ANSI["blue"] if agent_id == 0 else COLORS_ANSI["purple"]

        self.logger = Logger(self.output_path, self.agent_id, self.config["use_wandb"])
        self.mapper = Mapper(self.config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(self.config["tracking"], self.dataset, self.logger)
        self.feature_extractor = get_feature_extractor(self.config["loop_detection"])
        self.current_keyframe_feature = None
        self.current_submap_feature = None

    def set_pipe(self, pipe) -> None:
        """ Sets the pipe for communication with the server (main process). """
        self.pipe = pipe

    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap, saving the current submap's checkpoint and resetting the Gaussian model.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        ckpt_gaussian_model = clone_obj(gaussian_model)
        reference_w2c = torch2np(torch.inverse(self.estimated_c2ws[frame_id]))
        render_settings = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics, reference_w2c)

        with torch.no_grad():
            render_dict = render_gaussian_model(gaussian_model, render_settings)
            visibile_gaussians_mask = render_dict["radii"] > 0

            gaussian_model = GaussianModel(0)
            gaussian_model.training_setup(self.opt)

            if self.config["submap"]["anchor_data"] == "pcd":
                ckpt_gaussian_model.prune_points(visibile_gaussians_mask)
                submap_ckpt_name = str(self.submap_id).zfill(6)
                submap_data = self.get_submap_data(frame_id, ckpt_gaussian_model)
                save_dict_to_ckpt(submap_data, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
            elif self.config["submap"]["anchor_data"] == "depth":
                ckpt_gaussian_model.prune_points(visibile_gaussians_mask)
                self.save_current_submap_depth(frame_id, ckpt_gaussian_model)
            elif self.config["submap"]["anchor_data"] == "render_depth":
                self.save_current_submap_render(frame_id, ckpt_gaussian_model)
            else:
                raise ValueError(f"Unknown anchor data type: {self.config['submap']['anchor_data']}")

        self.mapper.keyframes = []
        self.submap_start_frame_ids.append(frame_id)
        self.submap_id += 1
        return gaussian_model

    def _init_wandb(self) -> None:
        """ Initializes the Weights & Biases logging for the agent. """
        wandb.init(project=self.run_info["project_name"], entity=self.run_info["entity"],
                   id=self.run_info["run_id"], name=self.run_info["run_name"],
                   resume="must", save_code=True, config=self.config, group=self.run_info["group"])

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if not self.active_submapping:
            # In case we use old initialization scheme
            return frame_id % self.config["mapping"]["new_submap_every"] == 0
        else:
            cosine_similarity = F.cosine_similarity(self.current_frame_feature, self.current_submap_feature)
            if (1 - cosine_similarity > self.submapping_threshold) and (self.submap_faiss_index.ntotal >= self.config["submapping"]["keyframe_num"]):
                # Update the current submap feature to the current frame feature
                self.current_submap_feature = self.current_frame_feature
                # If the cosine similarity is below the threshold, we start a new submap
                return True
            return False

    def should_start_mapping(self, frame_id: int) -> bool:
        """ Determines whether mapping should be started based on the current frame ID.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start mapping.
        """
        # In case we use old initialization scheme
        if not self.active_keyframing:
            # In case we use old initialization scheme
            if frame_id % self.config["mapping"]["map_every"] == 0:
                return True
            return False
        else:
            # Search the submap's Faiss index for the most similar keyframe
            similarities, indices = self.submap_faiss_index.search(self.current_frame_feature, k=1)
            highest_similarity = similarities[0][0]
            if 1 - highest_similarity > self.keyframing_threshold:
                # Add the current frame feature to the submap index
                self.submap_faiss_index.add(self.current_frame_feature)
                return True
            return False

    def save_current_submap_depth(self, frame_id: int, gaussian_model: GaussianModel, after_pgo: bool = False):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        # Pruning the Gaussian model if active compaction is enabled and before PGO
        # if self.config["mapping"]["active_compaction"] and not after_pgo:
        #     gaussian_model.prune_points(gaussian_model.spa_mask)
        submap_start_frame_id = self.submap_start_frame_ids[-1]
        submap_end_frame_id = frame_id

        submap_ckpt_name = str(self.submap_id).zfill(6)

        submap_ckpt = {
            "agent_id": self.agent_id,
            "submap_id": self.submap_id,
            "gaussian_model_params": gaussian_model.capture_dict(),
            "submap_start_frame_id": submap_start_frame_id,
            "submap_end_frame_id": submap_end_frame_id - 1,  # we do [] incusive indexing
            "submap_c2ws": torch2np(self.estimated_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "submap_gt_c2ws": torch2np(self.gt_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "keyframe_ids": np.array([kf_id for kf_id, _ in self.mapper.keyframes]),
            "submap_features": self.submap_features[self.submap_id],
            "start_rgb": self.dataset[submap_start_frame_id][1],
            "start_depth": self.dataset[submap_start_frame_id][2],
            "intrinsics": self.dataset.intrinsics
        }

        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")

    def save_current_submap_render(self, frame_id: int, gaussian_model: GaussianModel, after_pgo: bool = False):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        # Pruning the Gaussian model if active compaction is enabled and before PGO
        if self.config["mapping"]["active_compaction"] and not after_pgo:
            gaussian_model.prune_points(gaussian_model.spa_mask)
        submap_start_frame_id = self.submap_start_frame_ids[-1]
        submap_end_frame_id = frame_id

        submap_ckpt_name = str(self.submap_id).zfill(6)

        submap_ckpt = {
            "agent_id": self.agent_id,
            "submap_id": self.submap_id,
            "gaussian_model_params": gaussian_model.capture_dict(),
            "submap_start_frame_id": submap_start_frame_id,
            "submap_end_frame_id": submap_end_frame_id - 1,  # we do [] incusive indexing
            "submap_c2ws": torch2np(self.estimated_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "submap_gt_c2ws": torch2np(self.gt_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "keyframe_ids": np.array([kf_id for kf_id, _ in self.mapper.keyframes]),
            "submap_features": self.submap_features[self.submap_id],
            "width": self.dataset.width,
            "height": self.dataset.height,
            "intrinsics": self.dataset.intrinsics
        }

        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")

    def get_submap_data(self, frame_id: int, gaussian_model: GaussianModel) -> dict:
        """ Prepares the submap data to be saved to a checkpoint file or sent to the server.
        Args:
            frame_id: The ID of the current frame being processed.
            gaussian_model: The GaussianModel instance to capture the parameters from.
        Returns:
            A dictionary containing the submap data.
        """
        submap_start_frame_id = self.submap_start_frame_ids[-1]
        submap_end_frame_id = frame_id
        # Encode RGB images as JPEG in memory
        kf_rgb_encoded = {}
        for kf_id, data in self.mapper.keyframes:
            rgb_tensor = data['color']
            kf_rgb_encoded[kf_id] = tensor_to_jpeg_bytes_cv2(rgb_tensor, quality=95)
        
        # Encode depth maps as PNG in memory
        kf_depth_encoded = {}
        for kf_id, data in self.mapper.keyframes:
            depth_tensor = data['depth']
            kf_depth_encoded[kf_id] = depth_to_compressed_bytes(depth_tensor)
        return {
            "agent_id": self.agent_id,
            "submap_id": self.submap_id,
            "gaussian_model_params": gaussian_model.capture_dict(),
            "submap_start_frame_id": submap_start_frame_id,
            "submap_end_frame_id": submap_end_frame_id - 1,  # we do [] incusive indexing
            "submap_c2ws": torch2np(self.estimated_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "submap_gt_c2ws": torch2np(self.gt_c2ws[submap_start_frame_id:submap_end_frame_id]),
            "keyframe_ids": np.array([kf_id for kf_id, _ in self.mapper.keyframes]),
            "kf_rgb_gt": kf_rgb_encoded,
            "kf_depth_gt": kf_depth_encoded,
            "submap_features": self.submap_features[self.submap_id],
            "point_cloud": ptcloud2numpy(self.dataset.get_point_cloud(submap_start_frame_id, np.eye(4)))
        }

    def init_map(self) -> GaussianModel:
        """ Initializes the mapping process by setting up the Gaussian model and mapping the first frame.
        Returns:
            The initialized GaussianModel instance.
        """
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.map(0, torch2np(self.gt_c2ws[0]), gaussian_model, True, 1000)
        self.estimated_c2ws[0] = self.gt_c2ws[0].clone()
        self.keyframe_ids.append(0)
        return gaussian_model

    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        print(self.log_color + f"Starting agent {self.agent_id}" + COLORS_ANSI["end"] + "\n")
        setup_seed(self.config["seed"])

        if self.config["use_wandb"]:
            self._init_wandb()

        gaussian_model = self.init_map()
        image = Image.fromarray(self.dataset[0][1])
        self.current_keyframe_feature = self.feature_extractor.extract_features(image)
        self.current_submap_feature = copy.deepcopy(self.current_keyframe_feature)
        self.submap_features[self.submap_id] = self.feature_extractor.extract_features(image).cpu().numpy()
        self.submap_faiss_index = faiss.IndexFlatIP(self.current_keyframe_feature.shape[1])
        self.submap_faiss_index.add(self.current_keyframe_feature)

        for frame_id in range(1, len(self.dataset)):

            if frame_id == 1:
                estimated_c2w = self.dataset[frame_id][-1]
            else:
                estimated_c2w = self.tracker.track(
                    frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Prepare gaussian model for potential mapping
            gaussian_model.training_setup(self.opt)
            image = Image.fromarray(self.dataset[frame_id][1])
            self.current_frame_feature = self.feature_extractor.extract_features(image)
            start_new_submap = self.should_start_new_submap(frame_id)

            if start_new_submap:
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)
                self.keyframe_ids.append(frame_id)
                self.submap_faiss_index.reset()
                self.submap_faiss_index.add(self.current_frame_feature)
                extensive_seeding, max_iterations = True, self.mapper.new_submap_iterations
                self.mapper.map(frame_id, estimated_c2w, gaussian_model, extensive_seeding, max_iterations)

                image = Image.fromarray(self.dataset[frame_id][1])
                self.current_keyframe_feature = self.feature_extractor.extract_features(image)
                self.current_submap_feature = copy.deepcopy(self.current_keyframe_feature)
                self.submap_features[self.submap_id] = self.feature_extractor.extract_features(image).cpu().numpy()
                self.submap_faiss_index.add(self.current_keyframe_feature)

            elif self.should_start_mapping(frame_id):
                self.keyframe_ids.append(frame_id)
                extensive_seeding, max_iterations = False, self.mapper.iterations
                self.mapper.map(frame_id, estimated_c2w, gaussian_model, extensive_seeding, max_iterations)

        if self.config["multi_gpu"]:  # Logging GPU usage makes sense only in multi-GPU mode
            # GPU:0 is used by the server
            self.logger.log_gpu_usage(self.agent_id + 1)

        if self.config["submap"]["anchor_data"] == "pcd":
            submap_ckpt_name = str(self.submap_id).zfill(6)
            submap_data = self.get_submap_data(frame_id + 1, gaussian_model)
            save_dict_to_ckpt(submap_data, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
        elif self.config["submap"]["anchor_data"] == "depth":
            self.save_current_submap_depth(frame_id + 1, gaussian_model)
        elif self.config["submap"]["anchor_data"] == "render_depth":
            self.save_current_submap_render(frame_id + 1, gaussian_model)
        else:
            raise ValueError(f"Unknown anchor data type: {self.config['submap']['anchor_data']}")

        agents_submaps = []
        for submap_path in sorted((self.output_path / "submaps").glob("*")):
            agent_submap = torch.load(submap_path, map_location="cpu", weights_only=False)
            agents_submaps.append(agent_submap)
        self.pipe.send(("pgo", agents_submaps))

        optimized_kf_poses = self.pipe.recv()

        self.tracker = None  # to avoid CUDA memory leak
        self.logger.log_tracking_results(self.estimated_c2ws[self.keyframe_ids],
                                         self.gt_c2ws[self.keyframe_ids], "kfs.png")
        self.logger.log_tracking_results(optimized_kf_poses,
                                         self.gt_c2ws[self.keyframe_ids], "optimized_kfs.png")

        save_dict_to_ckpt(np2torch(np.array(self.keyframe_ids)), "kf_ids.ckpt", directory=self.output_path)
        save_dict_to_ckpt(self.estimated_c2ws[self.keyframe_ids], "estimated_kf_c2w.ckpt", directory=self.output_path)
        save_dict_to_ckpt(np2torch(optimized_kf_poses), "optimized_kf_c2w.ckpt", directory=self.output_path)

        if self.config["submap"]["anchor_data"] == "pcd":
            submap_ckpt_name = str(self.submap_id).zfill(6)
            submap_data = self.get_submap_data(frame_id + 1, gaussian_model)
            save_dict_to_ckpt(submap_data, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
            save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
        elif self.config["submap"]["anchor_data"] == "depth":
            self.save_current_submap_depth(frame_id + 1, gaussian_model, after_pgo=True)
        elif self.config["submap"]["anchor_data"] == "render_depth":
            self.save_current_submap_render(frame_id + 1, gaussian_model, after_pgo=True)
        else:
            raise ValueError(f"Unknown anchor data type: {self.config['submap']['anchor_data']}")