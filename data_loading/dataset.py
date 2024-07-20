from torch.utils.data import Dataset
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import h5py
from tqdm import tqdm
import numpy as np
# loads h5 data into memory for faster access
from torch import dtype
import numpy as np
import h5py
from collections import OrderedDict
from typing import Union
import torch


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def create_sample_indices(episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0):
    # Currently uses truncated as episode ends which is the end of the episode and not the end of the trajectory
    indices = list()
    episode_length = 0
    episode_index = 1 # Start 1 for human readability
    for i in range(len(episode_ends)):
        episode_length += 1
        if episode_ends[i]:
            start_idx = 0 if i <= 0 else i - episode_length + 1
            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after

            # Create indices for each possible sequence in the episode
            for idx in range(min_start, max_start + 1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
                start_offset = buffer_start_idx - (idx + start_idx)
                end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
            episode_length = 0
            episode_index += 1
    return np.array(indices)


def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            if isinstance(input_arr, torch.Tensor):
                data = torch.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            else:
                data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x

def convert_observation(obs, task_id):
    # adds task_id to the observation
    values = list(obs.values())
    task_id = np.full((values[0].shape[0], 1), task_id, dtype=values[0].dtype) 
    values.append(task_id)
    print("Task ID")
    print(values[0].shape)
    print(task_id.shape)
    # concatenate all the values
    return np.concatenate(values, axis=-1)

def get_observations(obs):
    #ensoure that the observations are in the correct format
    #and ordered correctly across tasks

    cleaned_obs = OrderedDict()
    cleaned_obs["qpos"] = obs["agent"]["qpos"]
    cleaned_obs["qvel"] = obs["agent"]["qvel"]
    cleaned_obs["tcp_pose"] = obs["extra"]["tcp_pose"]
    obs["extra"].pop("tcp_pose")

    #this code is not generic and only works for the specific observation spaces we have
    # Handle different goal position formats gracefully
    goal_pose_keys = ["goal_pose", "goal_pos", "box_hole_pos", "cubeB_pose"]
    for key in goal_pose_keys:
        if key in obs["extra"]:
            pos = obs["extra"][key]

            # Ensure 'pos' is 2D with the correct number of columns
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)  # Reshape to 2D if necessary
            elif pos.ndim > 2:
                raise ValueError(f"Unexpected dimensions for '{key}': {pos.shape}")

            # Pad or truncate 'pos' to have 7 columns
            print("Dtypes of before and after")
            print(pos.dtype)
            pos = np.pad(pos[:, :7], ((0, 0), (0, 7 - pos.shape[1])), mode='constant')
            if isinstance(cleaned_obs["tcp_pose"], torch.Tensor):
                pos = torch.tensor(pos, dtype=cleaned_obs["tcp_pose"].dtype)
            print(pos.dtype)
            cleaned_obs["goal_pose"] = pos
            obs["extra"].pop(key)
            break  # Stop once a valid goal pose key is found
    else:
        print("No goal pose found. Setting to zero.")
        length = len(obs["extra"]["tcp_pose"])
        cleaned_obs["goal_pose"] = np.zeros((length, 7), dtype=np.float32)  # Ensure 2D shape
        
    #is_grasped_reshaped = np.reshape(obs["extra"]["is_grasped"], (len(obs["extra"]["is_grasped"]), 1))
    
    # Filter and add other observations with 7 columns
    for key, value in obs["extra"].items():
        if value.shape[-1] == 7 and value.ndim == 2:
            cleaned_obs[key] = value
    
    for key, value in cleaned_obs.items():
        print(key)
        print(value.dtype)
    return cleaned_obs

def get_data_stats(data, obs_mask: bool = False):
    data = data.reshape(-1,data.shape[-1])

    # Create a mask to exclude the specified values from normalization
    mask = np.ones_like(data[0], dtype=bool)
    if obs_mask:
        # This values are the goal pose values
        # + the task id which is the last value
        mask[25] = False # x
        mask[26] = False # y
        mask[27] = False # z
        mask[28] = False # qx
        mask[29] = False # qy
        mask[30] = False # qz
        mask[31] = False # qw
        mask[39] = False # task id
    
    # Filter data to exclude specified values
    mask = np.repeat(mask[np.newaxis, :], data.shape[0], axis=0)
    mask = np.where(mask, 0, 1)
    masked_data = np.ma.masked_array(data, mask) # Apply mask to data

    stats = {
        'min': np.min(masked_data.data, axis=0),
        'max': np.max(masked_data.data, axis=0),
        'mask': mask,
    }

    return stats

def set_last_dimension(array, mask):
    """Sets the last dimension of an array to new_data, matching the original shape.

    Args:
        original_array: The original NumPy array.
        new_data: A 1D array containing the new data for the last dimension.

    Returns:
        A new NumPy array with the same shape as original_array, 
        but with the last dimension set to new_data.
    """
    new_shape = array.shape[:-1] + (len(mask),)
    new_data_expanded = np.expand_dims(mask, axis=0)  
    return np.broadcast_to(new_data_expanded, new_shape)


def normalize_data(data, stats):
    # Calculate the denominator for normalization
    # Avoid division by zero
    denominator = stats['max'] - stats['min']
    denominator[denominator == 0] = 1

    # nomalize to [0,1]
    ndata = (data - stats['min']) / denominator

    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    # Set masked values to original values
    mask = set_last_dimension(data, stats['mask'][0])
    ndata = np.where(mask, data, ndata)

    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']

    # Set masked values to original values
    mask = set_last_dimension(data, stats['mask'][0])
    data = np.where(mask, data, ndata)

    return data




class StateDataset(Dataset):
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
        self, dataset_file: str, pred_horizon: int, obs_horizon: int, action_horizon:int, task_id: np.float32, load_count=-1, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.task_id = task_id
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count), desc="Loading Episodes", colour="green"):
            eps = self.episodes[eps_id]
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
        
        self.truncated = np.zeros(self.actions.shape[0], dtype=bool)
        self.truncated[-1] = True
        
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
        obs_dict = get_observations(self.obs)
        train_data = dict(
                        obs=convert_observation(obs_dict, self.task_id),
                        actions=self.actions,
                        )

         # Initialize index lists and stat dicts
        self.indices = create_sample_indices(
            episode_ends=self.truncated, 
            sequence_length=self.pred_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1
        )

        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            if key == "actions":
                stats[key] = get_data_stats(data)
            else:
                stats[key] = get_data_stats(data, True)
            
            normalized_train_data[key] = normalize_data(data, stats[key])
  
        self.normalized_train_data = normalized_train_data
        self.stats = stats

    def __len__(self):
        # all possible sequenzes of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Change data to fit diffusion policy
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

    
        sampled = sample_sequence(
            train_data=self.normalized_train_data, 
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
                sampled[k] = sampled[k][:self.obs_horizon,:]
        sampled[k] = common.to_tensor(sampled[k], device=self.device)

        return sampled