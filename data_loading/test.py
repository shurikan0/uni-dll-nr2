import numpy as np
from normal_dataset2 import StateDataset2, unnormalize_data2
from helper import unnormalize_data
from mani2_streaming_dataset import StreamingTrajectoryDataset2
from normal_dataset import StateNormalDataset
from torch.utils.data import DataLoader

from streaming_dataset import StreamingTrajectoryDataset
import imageio
from PIL import Image

# ManiSkill File
dataset_file1 = "data/staterawdata.state_dict.pd_ee_delta_pos.h5"


load_count = 20  # Load all episodes
succes_only = False  # Load all episodes regardless of success
normalize = False  # Normalization not working yet
drop_last = True
task_id = 0.5


# Define the horizons used for the diffusion policy
pred_horizon = 16
obs_horizon = 16
action_horizon = 8
# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


# create dataset from file
dataset = StateNormalDataset(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    task_id,
    load_count,
)

dataset2 = StateDataset2(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    load_count,
    success_only=succes_only,
)

dataloader = DataLoader(dataset, shuffle=False)
dataloader2 = DataLoader(dataset2, shuffle=False)
batch = next(iter(dataloader))
batch2 = next(iter(dataloader2))

batch["obs"] == batch2["obs"]
#print("Testing")
#print(batch["obs"].shape)
#print(batch["obs"].dtype)
#print(batch["actions"].shape)
#print(batch["actions"].dtype)
#print(batch["obs"][0].shape)
#print(batch["obs"][0][0].shape)

batch["obs"] = unnormalize_data(batch["obs"], dataset.stats["obs"])
batch2["obs"] = unnormalize_data2(batch2["obs"], dataset2.stats["obs"])
