from normal_dataset import NormalTrajectoryDataset
from torch.utils.data import DataLoader

from streaming_dataset import StreamingTrajectoryDataset

#ManiSkill File
dataset_file = 'data/trajectory.rgbd.pd_ee_delta_pos.h5'

load_count = 20 # Load all episodes
succes_only = False # Load all episodes regardless of success
normalize = False # Normalization not working yet


# Define the horizons used for the diffusion policy
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


# create dataset from file
dataset = NormalTrajectoryDataset(dataset_file, pred_horizon, obs_horizon, action_horizon, load_count, succes_only, normalize)
dataset2 = StreamingTrajectoryDataset(dataset_file, pred_horizon, obs_horizon, action_horizon, normalize)

# Debugging
# Dataset should be 3dim (batch, sequence, features)
# Can be used by Diffusion policy

dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=None, num_workers=4)

batch_count = 0
for batch in iter(dataloader):
    batch_count += 1

batch_count2 = 0
for batch in iter(dataloader2):
    batch_count2 += 1

print("Dataset 1",batch_count)
print("Dataset 2",batch_count2)
    



