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

def convert_observation(obs):
    return np.concatenate(list(obs.values()), axis=-1)

def get_observations(obs):
    #ensoure that the observations are in the correct format
    #and ordered correctly across tasks

    cleaned_obs = OrderedDict()
    cleaned_obs["qpos"] = obs["agent"]["qpos"]
    cleaned_obs["qvel"] = obs["agent"]["qvel"]
    cleaned_obs["tcp_pose"] = obs["extra"]["tcp_pose"]

    #this code is not generic and only works for the specific observation spaces we have
    if "goal_pose" in obs["extra"]:
        cleaned_obs["goal_pose"] = obs["extra"]["goal_pose"]
        obs["extra"].pop("goal_pose")
    elif "goal_pos" in obs["extra"]:
        pos = obs["extra"]["goal_pos"]
        quad = np.array([1, 0, 0, 0])
        cleaned_obs["goal_pose"] = np.concatenate([pos, quad])
        obs["extra"].pop("goal_pos")
    elif "box_hole_pos" in obs["extra"]:
        cleaned_obs["goal_pose"] = obs["extra"]["box_hole_pos"]
        obs["extra"].pop("box_hole_pos")
    elif "cubeB_pose" in obs["extra"]:
        cleaned_obs["goal_pose"] = obs["extra"]["cubeB_pose"]
        obs["extra"].pop("cubeB_pose")
    else:
        print("No goal pose found in observation")
        print("Setting goal pose to zero")
        cleaned_obs["goal_pose"] = np.zeros(7)
        
    #is_grasped_reshaped = np.reshape(obs["extra"]["is_grasped"], (len(obs["extra"]["is_grasped"]), 1))
    # Filter out observations that are no positions
    for key, value in obs["extra"].items():
        if value is dtype.is_floating_point:
            if value.size == 7:
                cleaned_obs[key] = value
    
    return cleaned_obs

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
