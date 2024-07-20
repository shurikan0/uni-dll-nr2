import numpy as np
from data_loading.old.state_streaming_dataset import StateStreamingDataset

from torch.utils.data import DataLoader

import imageio
from PIL import Image

# ManiSkill File
dataset_file = "data/staterawdata.state_dict.pd_ee_delta_pos.h5"


load_count = 20  # Load all episodes
succes_only = False  # Load all episodes regardless of success
normalize = False  # Normalization not working yet
drop_last = True


# Define the horizons used for the diffusion policy
pred_horizon = 16
obs_horizon = 16
action_horizon = 8
# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


dataset1 = StateStreamingDataset(
    dataset_file, pred_horizon, obs_horizon, action_horizon
)



# Debugging
# Dataset should be 3dim (batch, sequence, features)
# Can be used by Diffusion policy

dataloader = DataLoader(dataset1, batch_size=10)

batch = next(iter(dataloader))

print(batch.keys())
print("tcp_pose:", batch['tcp_pose'].shape, batch['tcp_pose'].dtype)
print("obj_pose:", batch['obj_pose'].shape, batch['obj_pose'].dtype)
print("goal_pos:", batch['goal_pos'].shape, batch['goal_pos'].dtype)
print("qpos:", batch['qpos'].shape, batch['qpos'].dtype)
print("qvel:", batch['qvel'].shape, batch['qvel'].dtype)
print("actions:", batch['actions'].shape, batch['actions'].dtype)
    
