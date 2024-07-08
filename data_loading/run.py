from pointcloud_dataset import PointCloudManiSkillTrajectoryDataset
from torch.utils.data import DataLoader

from streaming_dataset import TrajectoryDataset

#ManiSkill File
dataset_file = 'data/trajectory.pointcloud.pd_joint_delta_pos.h5'

load_count = -1 # Load all episodes
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
dataset = PointCloudManiSkillTrajectoryDataset(dataset_file, pred_horizon, obs_horizon, action_horizon, load_count, succes_only, normalize)
dataset2 = TrajectoryDataset(dataset_file, pred_horizon, obs_horizon, action_horizon, normalize)

# Debugging
# Dataset should be 3dim (batch, sequence, features)
# Can be used by Diffusion policy

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=10, num_workers=4)

first_batch = next(iter(dataloader))
first_batch2 = next(iter(dataloader2))

batch_count = 0
for batch in iter(dataloader):
    print(batch_count)
    batch_count += 1

batch_count2 = 0
for batch in iter(dataloader2):
    print(batch_count2)
    batch_count2 += 1
    



