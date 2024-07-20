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
    print("Task ID")
    print(values.shape)
    print(task_id.shape)
    values.append(task_id)

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
            pos = np.pad(pos[:, :7], ((0, 0), (0, 7 - pos.shape[1])), mode='constant')

            if isinstance(cleaned_obs["tcp_pose"], torch.Tensor):
                pos = torch.tensor(pos, dtype=cleaned_obs["tcp_pose"].dtype)

            cleaned_obs["goal_pose"] = pos
            obs["extra"].pop(key)
            break  # Stop once a valid goal pose key is found
    else:
        print("No goal pose found. Setting to zero.")
        length = len(obs["extra"]["tcp_pose"])
        cleaned_obs["goal_pose"] = np.zeros((length, 7), dtype=torch.float64)  # Ensure 2D shape
        
    #is_grasped_reshaped = np.reshape(obs["extra"]["is_grasped"], (len(obs["extra"]["is_grasped"]), 1))
    
    # Filter and add other observations with 7 columns
    for key, value in obs["extra"].items():
        if value.shape[-1] == 7 and value.ndim == 2:
            cleaned_obs[key] = value
    
    return cleaned_obs

def get_data_stats(data, obs_mask: bool = False):
    data = data.reshape(-1,data.shape[-1])

    # Create a mask to exclude the specified values from normalization
    mask = np.ones_like(data[0], dtype=bool)
    if obs_mask:
        mask[28] = False
        mask[29] = False
        mask[30] = False
        mask[31] = False
        mask[39] = False
    
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

def normalize_data(data, stats):
    # Calculate the denominator for normalization
    denominator = stats['max'] - stats['min']
    denominator[denominator == 0] = 1

    # nomalize to [0,1]
    ndata = (data - stats['min']) / denominator

    # normalize to [-1, 1]
    ndata = ndata * 2 - 1

    # Set masked values to original values
    ndata = np.where(stats['mask'], data, ndata)
    #print("Unnormalized NEW")
    #print(data[0])
    #print("Stats NEW")
    #print(stats['max'])
    #print(stats['min'])
    #print("Normalized Data NEW")
    #print(ndata[0])
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']

    # Set masked values to original values
    mask = data[:,:, stats['mask'][0]]
    data = np.where(mask, data, ndata)

    #print("Unnormalized NEW AGAIN")
    #print(data[0][0])
    return data
