from typing import Union
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from helper import *
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json


class PointCloudManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(
        self, dataset_file: str, pred_horizon: int, obs_horizon: int, action_horizon:int, load_count=-1, success_only: bool = False, normalize: bool = False, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.normalize = normalize
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.is_pointcloud = dataset_file.find("pointcloud") != -1

        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count), desc="Loading Episodes", colour="green"):
            eps = self.episodes[eps_id]
            if success_only:
                assert (
                    "success" in eps
                ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps["success"]:
                    continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])

            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if eps_id == 0:
                self.obs = obs
            else:
                self.obs = common.append_dict_array(self.obs, obs)

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])

        self.actions = np.vstack(self.actions)
        self.terminated = np.concatenate(self.terminated)
        self.truncated = np.concatenate(self.truncated)
        
        
        if self.rewards is not None:
            self.rewards = np.concatenate(self.rewards)
        if self.success is not None:
            self.success = np.concatenate(self.success)
        if self.fail is not None:
            self.fail = np.concatenate(self.fail)

        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x

        # uint16 dtype is used to conserve disk space and memory
        # you can optimize this dataset code to keep it as uint16 and process that
        # dtype of data yourself. for simplicity we simply cast to a int32 so
        # it can automatically be converted to torch tensors without complaint
        self.obs = remove_np_uint16(self.obs)
        
        if device is not None:
            self.actions = common.to_tensor(self.actions, device=device)
            self.obs = common.to_tensor(self.obs, device=device)
            self.terminated = common.to_tensor(self.terminated, device=device)
            self.truncated = common.to_tensor(self.truncated, device=device)
            if self.rewards is not None:
                self.rewards = common.to_tensor(self.rewards, device=device)
            if self.success is not None:
                self.success = common.to_tensor(self.terminated, device=device)
            if self.fail is not None:
                self.fail = common.to_tensor(self.truncated, device=device)
        


        # Added code for diffusion policy

         # Initialize index lists and stat dicts
        self.indices = create_sample_indices(
            episode_ends=self.truncated, 
            sequence_length=self.pred_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1
        )

        # normalize observations between -1 and 1
        if self.normalize:
            self.obs = normalize_data(self.obs, self.terminated)
        

    def __len__(self):
        # all possible sequenzes of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Change data to fit diffusion policy
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        if self.is_pointcloud:
            train_data = dict(
                obs_agent_qpos=self.obs["agent"]["qpos"],
                obs_agent_qvel=self.obs["agent"]["qvel"],
                obs_xyzw=self.obs["pointcloud"]["xyzw"],
                obs_rgb=self.obs["pointcloud"]["rgb"],
                obs_segmentation=self.obs["pointcloud"]["segmentation"],
                actions=self.actions,
            )
        else:
            train_data = dict(
            obs_agent_qpos=self.obs["agent"]["qpos"],
            obs_agent_qvel=self.obs["agent"]["qvel"],
            actions=self.actions,
            )
            for k in self.obs["sensor_data"].keys():
                for kk in self.obs["sensor_data"][k].keys():
                    train_data[f"obs_{k}_{kk}"] = self.obs["sensor_data"][k][kk]
                       
                  

    
        sampled = sample_sequence(
            train_data=train_data, 
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
    
        # discard unused observations in the sequence
        for k in sampled.keys():
            if k != "actions":
                # discard unused observations in the sequence
                sampled[k] = sampled[k][:self.pred_horizon,:]
        sampled[k] = common.to_tensor(sampled[k], device=self.device)
    
        return sampled