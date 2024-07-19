import numpy as np
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
    load_count,
)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

batch = next(iter(dataloader))

print(batch["obs"].shape)
print(batch["actions"].shape)

