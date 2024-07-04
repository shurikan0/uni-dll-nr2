from pointCloudDataClass import PointCloudManiSkillTrajectoryDataset
from torch.utils.data import DataLoader

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


# Debugging
# Dataset should be 3dim (batch, sequence, features)
# Can be used by Diffusion policy

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

first_batch = next(iter(dataloader))

print(first_batch.keys())


