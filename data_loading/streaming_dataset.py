import math
import h5py
from torch.utils.data import IterableDataset
from tqdm import tqdm
from helper import *
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json
import torch


class StreamingTrajectoryDataset(IterableDataset):
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
        self.is_pointcloud = dataset_file.find("pointcloud") != -1
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
                terminated = trajectory["terminated"]
                truncated = trajectory["truncated"]
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
                            actions=actions,
                        )
                        for k in obs["sensor_data"].keys():
                            for kk in obs["sensor_data"][k].keys():
                                train_data[f"obs_{k}_{kk}"] = obs["sensor_data"][k][kk]
                       
                  
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
                    

   