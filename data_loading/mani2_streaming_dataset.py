import math
import h5py
from torch.utils.data import IterableDataset
from tqdm import tqdm
from helper import *
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import torch
from typing import Union
import h5py
import numpy as np
import torch

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def convert_observation(observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the rgb and depth images

    # image data is not scaled here and is kept as uint16 to save space
    image_obs = observation["image"]
    rgb = image_obs["base_camera"]["rgb"]
    depth = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]

    # we provide a simple tool to flatten dictionaries with state data
    from mani_skill.utils.common import flatten_state_dict
    state = np.hstack(
        [
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"]),
        ]
    )

    # combine the RGB and depth images
    rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=-1)
    obs = dict(rgbd=rgbd, state=state)
    return obs

def rescale_rgbd(rgbd, scale_rgb_only=False):
    # rescales rgbd data and changes them to floats
    rgb1 = rgbd[..., 0:3] / 255.0
    rgb2 = rgbd[..., 4:7] / 255.0
    depth1 = rgbd[..., 3:4]
    depth2 = rgbd[..., 7:8]
    if not scale_rgb_only:
        depth1 = rgbd[..., 3:4] / (2**10)
        depth2 = rgbd[..., 7:8] / (2**10)
    return np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)

def create_sample_indices(episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0):
    # Currently uses truncated as episode ends which is the end of the episode and not the end of the trajectory
    # TODO: What to use as episode ends?
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

class StreamingTrajectoryDataset2(IterableDataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases
    Implements the IterableDataset class for PyTorch to allow for streaming data loading.
    Currently only supports PointCloud and RGBD data.

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        pred_horizon (int): the number of steps to predict into the future
        obs_horizon (int): the number of steps to observe in the past
        action_horizon (int): the number of steps to execute actions in the future
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
        self.is_pointcloud = False #dataset_file.find("pointcloud") != -1
        self.current_episode = 0

    def __iter__(self):
        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process
            episode_indices = range(len(self.episodes))
        else:
            num_episodes = len(self.episodes)
            per_worker = int(math.ceil(num_episodes / float(worker_info.num_workers)))
            worker_id = worker_info.id
            episode_indices = range(worker_id * per_worker, min((worker_id + 1) * per_worker, num_episodes))

        for eps_id in episode_indices:
            eps = self.episodes[eps_id]
            with h5py.File(self.dataset_file, "r") as data:  # Context manager
                trajectory = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(trajectory)  # Load data
                eps_len = len(trajectory["actions"])

                # exclude the final observation as most learning workflows do not use it
                obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
                if self.current_episode > 0:
                    obs = common.append_dict_array(obs, obs)

                actions = trajectory["actions"]
                #terminated = trajectory["terminated"]
                truncated = np.zeros(actions.shape[0], dtype=bool)
                truncated[-1] = True



                rewards = trajectory.get("rewards", None)
                success = trajectory.get("success", None)
                fail = trajectory.get("fail", None)


                # uint16 dtype is used to conserve disk space and memory
                # you can optimize this dataset code to keep it as uint16 and process that
                # dtype of data yourself. for simplicity we simply cast to a int32 so
                # it can automatically be converted to torch tensors without complaint
                obs = remove_np_uint16(obs)

                if self.device is not None:
                    actions = common.to_tensor(actions, device=self.device)
                    obs = common.to_tensor(obs, device=self.device)
                    # terminated = common.to_tensor(terminated, device=self.device)
                    truncated = common.to_tensor(truncated, device=self.device)
                    if rewards is not None:
                        rewards = common.to_tensor(rewards, device=self.device)
                    #if success is not None:
                    #    success = common.to_tensor(terminated, device=self.device)
                    #if fail is not None:
                    #    fail = common.to_tensor(truncated, device=self.device)

                # Added code for diffusion policy

                # Initialize index lists and stat dicts
                indices = create_sample_indices(
                    episode_ends=truncated,
                    sequence_length=self.pred_horizon,
                    pad_before=self.obs_horizon - 1,
                    pad_after=self.action_horizon - 1
                )

                for idx in indices:
                    buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = idx

                    if self.is_pointcloud:
                        train_data = dict(
                            obs_agent_qpos=obs["agent"]["qpos"],
                            obs_agent_qvel=obs["agent"]["qvel"],
                            obs_xyzw=obs["pointcloud"]["xyzw"],
                            obs_rgb=obs["pointcloud"]["rgb"],
                            obs_segmentation=obs["pointcloud"]["segmentation"],
                            actions=actions,
                        )
                    else:
                        train_data = dict(
                            obs_agent_qpos=obs["agent"]["qpos"],
                            obs_agent_qvel=obs["agent"]["qvel"],
                            rgbd = convert_observation(obs)['rgbd'],
                            actions=actions
                        )
                        print(train_data)

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
                            sampled[k] = sampled[k][:self.obs_horizon,:]
                    sampled[k] = common.to_tensor(sampled[k], device=self.device)

                    yield sampled