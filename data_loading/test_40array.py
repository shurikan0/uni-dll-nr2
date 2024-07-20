import numpy as np

from dataset_old import StateNormalDataset
from dataset import StateDataset
from torch.utils.data import DataLoader

env_id = "PickCube-v1"
obs_mode = "state_dict"
control_mode = "pd_joint_delta_pos"


#variables
dataset_file1 = f'data/Generated/{env_id}/motionplanning/trajectory.{obs_mode}.{control_mode}.h5'
load_count = 1 
task_id = 0.1
pred_horizon = 16
obs_horizon = 2
action_horizon = 8


# create dataset from file
dataset = StateDataset(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    task_id,
    load_count,
)

dataset2 = StateNormalDataset(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    load_count,
)


dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
dataloader2 = DataLoader(dataset2, shuffle=False, batch_size=1)

iterator = iter(dataloader)
batch = next(iterator)
batch2 = next(iterator)
batch3 = next(iterator)

iterator2 = iter(dataloader2)
batch4 = next(iterator2)
batch5 = next(iterator2)
batch6 = next(iterator2)

print(batch['obs'].shape)
print(batch['obs'].dtype)
print(batch['obs'][0][0])
print(batch['obs'][0][1])
print(batch2['obs'][0][0])
print(batch2['obs'][0][1])
print(batch3['obs'][0][0])
print(batch3['obs'][0][1])

#print(batch4['obs'].shape)
#print(batch4['obs'].dtype)
#print(batch4['obs'][0][0])
#print(batch4['obs'][0][1])
#print(batch5['obs'][0][0])
#print(batch5['obs'][0][1])
#print(batch6['obs'][0][0])
#print(batch6['obs'][0][1])