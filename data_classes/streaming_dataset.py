import math
import h5py
from torch.utils.data import IterableDataset
from tqdm import tqdm
from helper import *
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import torch


class TrajectoryDataset(IterableDataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(
        self, dataset_file: str, pred_horizon: int, obs_horizon: int, action_horizon:int, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.is_pointcloud = dataset_file.find("pointcloud") != -1
        self.current_episode = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process (just in case)
            episode_indices = range(len(self.episodes))
        else:
            # Split episodes among workers
            num_episodes = len(self.episodes)
            per_worker = int(math.ceil(num_episodes / float(worker_info.num_workers)))
            worker_id = worker_info.id
            episode_indices = range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_episodes))
        
        if self.current_episode >= len(self.episodes):
            return None
        eps = self.episodes[self.current_episode]
        self.current_episode += 1
        
        with h5py.File(self.dataset_file, "r") as data:  # Context manager
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)  # Load data
            eps_len = len(trajectory["actions"])

            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if self.current_episode > 0:
                obs = common.append_dict_array(obs, obs)

            actions = trajectory["actions"]
            terminated = trajectory["terminated"]
            truncated = trajectory["truncated"]
            rewards = trajectory.get("rewards", None)
            success = trajectory.get("success", None)
            fail = trajectory.get("fail", None)
         
            
            # uint16 dtype is used to conserve disk space and memory
            # you can optimize this dataset code to keep it as uint16 and process that
            # dtype of data yourself. for simplicity we simply cast to a int32 so
            # it can automatically be converted to torch tensors without complaint
            obs = self.remove_np_uint16(obs)
            
            if self.device is not None:
                actions = common.to_tensor(actions, device=self.device)
                obs = common.to_tensor(obs, device=self.device)
                terminated = common.to_tensor(terminated, device=self.device)
                truncated = common.to_tensor(truncated, device=self.device)
                if rewards is not None:
                    rewards = common.to_tensor(rewards, device=self.device)
                if success is not None:
                    success = common.to_tensor(terminated, device=self.device)
                if fail is not None:
                    fail = common.to_tensor(truncated, device=self.device)
            
            # Added code for diffusion policy

            # Initialize index lists and stat dicts
            indices = create_sample_indices(
                episode_ends=truncated, 
                sequence_length=self.pred_horizon,
                pad_before=self.obs_horizon - 1,
                pad_after=self.action_horizon - 1
            )

            # normalize observations between -1 and 1
            if self.normalize:
                obs = normalize_data(obs, terminated)
    
            # ... (rest of your data processing and sequence sampling logic)
            # ... (yield sampled sequences)
            
            # NOTE: Since we're loading data on the fly, you might need to slightly modify the indexing logic in create_sample_indices to work with individual trajectories instead of the entire dataset.

    def __len__(self):
        # all possible sequenzes of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Change data to fit diffusion policy
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        
        
        if self.is_pointcloud:
            # For pointcloud
            
            train_data = dict(
                obs_agent_qpos=self.obs["agent"]["qpos"],
                obs_agent_qvel=self.obs["agent"]["qvel"],
                obs_xyzw=self.obs["pointcloud"]["xyzw"],
                obs_rgb=self.obs["pointcloud"]["rgb"],
                obs_segmentation=self.obs["pointcloud"]["segmentation"],
                actions=self.actions,
            )

    
            sampled = sample_sequence(
                train_data=train_data, 
                sequence_length=self.pred_horizon,
                buffer_start_idx=buffer_start_idx,
                buffer_end_idx=buffer_end_idx,
                sample_start_idx=sample_start_idx,
                sample_end_idx=sample_end_idx
            )
            for k in sampled.keys():
                if k != "actions":
                    # discard unused observations in the sequence
                    sampled[k] = sampled[k][:self.pred_horizon,:]
                sampled[k] = common.to_tensor(sampled[k], device=self.device)
            
            
            if idx == 0:
                print("")
                print("Dataset info for sequence:",idx)
                print("Sequence Buffer start idx",buffer_start_idx)
                print("Sequence Buffer end idx",buffer_end_idx)
                print("Sequence Sample start idx",sample_start_idx)
                print("Sequence Sample end idx",sample_end_idx)
                for k in sampled.keys():
                    print("Result",k,"has shape:",sampled[k].shape, "and type:", sampled[k].dtype)
                    
            return sampled
        else:
            # For each camera in rgbd
            train_data = dict(
                obs_agent_qpos=self.obs["agent"]["qpos"],
                obs_agent_qvel=self.obs["agent"]["qvel"],
                obs_rgb=self.obs["rgbd"]["rgb"],
                obs_depth=self.obs["rgbd"]["depth"],
                obs_segmentation=self.obs["rgbd"]["segmentation"],
                actions=self.actions,
            )
            sampled = sample_sequence(
                train_data=train_data, 
                sequence_length=self.pred_horizon,
                buffer_start_idx=buffer_start_idx,
                buffer_end_idx=buffer_end_idx,
                sample_start_idx=sample_start_idx,
                sample_end_idx=sample_end_idx
            )
            for k in sampled.keys():
                if k != "actions":
                    # discard unused observations in the sequence
                    sampled[k] = sampled[k][:self.pred_horizon,:]
                sampled[k] = common.to_tensor(sampled[k], device=self.device)
                
            if idx == 0:
                print("")
                print("Dataset info for sequence:",idx)
                print("Sequence Buffer start idx",buffer_start_idx)
                print("Sequence Buffer end idx",buffer_end_idx)
                print("Sequence Sample start idx",sample_start_idx)
                print("Sequence Sample end idx",sample_end_idx)
                for k in sampled.keys():
                    print("Result",k,"has shape:",sampled[k].shape, "and type:", sampled[k].dtype)
                    
            return sampled