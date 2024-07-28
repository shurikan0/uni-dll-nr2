import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union
from collections import OrderedDict
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json


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
    #print(f"episode_ends: {episode_ends}")
    #print(f"len(episode_ends): {len(episode_ends)}")
    end_of_last_episode = False
    for i in range(len(episode_ends)):
        episode_length += 1
        if episode_ends[i] and not end_of_last_episode:
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
            #print(f"Episode {episode_index} has {episode_length} steps")
            episode_length = 0
            episode_index += 1
            end_of_last_episode = True
        elif not episode_ends[i]:
            end_of_last_episode = False
    #print(f"Created {len(indices)} samples from {episode_index - 1} episodes")
    #print(f"All indices: {indices}")
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
    example = values[0]
    if isinstance(example, torch.Tensor):
          example = example.numpy()

    # add task_id to the observation
    task_id_array = np.full((example.shape[0], 1), task_id, dtype=example.dtype) 
    values.append(task_id_array)
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
    goal_pose_keys = ["goal_pose", "goal_pos", "box_hole_pose", "cubeB_pose"]
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
        length = len(cleaned_obs["tcp_pose"])
        cleaned_obs["goal_pose"] = np.zeros((length, 7), dtype=np.float32)  # Ensure 2D shape
        
    #is_grasped_reshaped = np.reshape(obs["extra"]["is_grasped"], (len(obs["extra"]["is_grasped"]), 1))
    
    # Filter and add other observations with 7 columns
    for key, value in obs["extra"].items():
        if value.shape[-1] == 7 and value.ndim == 2:
            if key != "receptacle_pose":
                cleaned_obs[key] = value

    count = 0
    for key in cleaned_obs.keys():
        count += cleaned_obs[key].shape[-1]
    
    assert count == 39, "Observation size is not 39"

    
    return cleaned_obs

def get_min_max_values(dataloader):
    # do this once before training
    min_vals = None
    max_vals = None
    mask = None

    for batch in dataloader:
        batch = batch['obs']
        batch_reshaped = batch.view(-1, batch.shape[-1])
        mask = torch.ones(batch_reshaped.shape[1], dtype=torch.bool)
        mask[39] = False
        batch_min = batch_reshaped[:, mask].min(dim=0)[0]
        batch_max = batch_reshaped[:, mask].max(dim=0)[0]
    
        if min_vals is None and max_vals is None:
            min_vals = batch_min
            max_vals = batch_max
        else:
            min_vals = torch.min(min_vals, batch_min)
            max_vals = torch.max(max_vals, batch_max)
    #print(f"Shape of min_vals: {min_vals.shape}")
    #print(f"Shape of max_vals: {max_vals.shape}")
    #print(f"Shape of mask: {mask.shape}")
    return {"obs": {"min": min_vals, "max": max_vals, "mask": mask}}

def normalize_batch(batch, stats):
    batch_reshaped = batch["obs"].view(-1, batch["obs"].shape[-1])
    mask = torch.ones(batch_reshaped.shape[1], dtype=torch.bool)
    mask[39] = False
    normalized_batch = batch_reshaped.clone()
    normalized_batch[:, mask] = (batch_reshaped[:, mask] - stats["obs"]["min"]) / (stats["obs"]["max"] - stats["obs"]["min"] + 0.1)
    #print(f"Shape of normalized_batch: {normalize_batch.shape}")
    batch["obs"] = normalized_batch.view(batch["obs"].shape)
    #print(f"Shape of normalized batch: {batch["obs"].shape}")
    return batch

def denormalize_batch(batch, stats):   
    batch_reshaped = batch["obs"].view(-1, batch["obs"].shape[-1])
    mask = torch.ones(batch_reshaped.shape[1], dtype=torch.bool)
    mask[39] = False
    denormalized_batch = batch_reshaped.clone() 
    denormalized_batch[:, mask] = batch_reshaped[:, mask] * (stats["obs"]["max"] - stats["obs"]["min"] + 0.1) + stats["obs"]["min"]
    #print(f"Shape of denormalized_batch: {denormalized_batch.shape}")
    #print(f"Batch shape: {batch['obs'].shape}")
    batch["obs"] = denormalized_batch.view(batch["obs"].shape)
    #print(f"Shape of denormalized batch: {batch["obs"].shape}")
    return batch

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    global_norm = False
    #0-8 are the joint positions
    #9-17 are the joint velocities
    #18-24 is tcp pose
    #25-31 is the goal pose
    #32-38 is obj pose
    #39 is the task id
    if global_norm:
        stats['min'][18] = -1.2
        stats['max'][18] = 1.2
        stats['min'][19] = -0.6
        stats['max'][19] = 0.6
        stats['min'][20] = 0
        stats['max'][20] = 1.0
        stats['min'][25] = -1.2
        stats['max'][25] = 1.2
        stats['min'][26] = -0.6
        stats['max'][26] = 0.6
        stats['min'][27] = 0
        stats['max'][27] = 1.0
        stats['min'][32] = -1.2
        stats['max'][32] = 1.2
        stats['min'][33] = -0.6
        stats['max'][33] = 0.6
        stats['min'][34] = 0
        stats['max'][34] = 1.0
    return stats

def normalize_data(data, stats, task_id):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'] + 0.1)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    ndata[...,39] = task_id
    return ndata

def unnormalize_data(ndata, stats, task_id):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min'] + 0.1) + stats['min']
    data[...,39] = task_id
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
        self, dataset_file: str, pred_horizon: int, obs_horizon: int, action_horizon:int, task_id: np.float32, load_count=-1, normalize=False,device=None
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
        self.end_episode = []
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
            #print(f"Episode {eps_id} has {eps_len} steps")

            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if eps_id == 0:
                self.obs = obs
            else:
                self.obs = common.append_dict_array(self.obs, obs)

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])


            end_episode = [False] * eps_len
            end_episode[-1] = True
            #is_terminated = False
            #for i in range(len(end_episode)):
            #    if trajectory["terminated"][i] == True or is_terminated:
            #        end_episode[i] = True
            #        is_terminated = True
            #    else:
            #        end_episode[i] = False

            #print(f"Episode {eps_id} has {end_episode.count(True)} end of episodes")
            self.end_episode.append(end_episode)
            #self.truncated[self.terminated:] = True

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
        self.end_episode = np.concatenate(self.end_episode)
        


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
        

         # Initialize index lists and stat dicts
        self.indices = create_sample_indices(
            episode_ends=self.end_episode, 
            sequence_length=self.pred_horizon,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1
        )

        # Added code for diffusion policy
        obs_dict = get_observations(self.obs)
        obs = convert_observation(obs_dict, self.task_id)

        self.stats = None
        
        if normalize:
            self.stats = get_data_stats(obs)
            obs = normalize_data(obs, self.stats, self.task_id)
       
        self.train_data = dict(
                        obs=obs,
                        actions=self.actions,
                        )


    def __len__(self):
        # all possible sequenzes of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Change data to fit diffusion policy
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

    
        sampled = sample_sequence(
            train_data=self.train_data, 
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