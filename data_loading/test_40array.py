import numpy as np

from dataset import StateDataset
from torch.utils.data import DataLoader

env_id = "PegInsertionSide-v2"
obs_mode = "state_dict"
control_mode = "pd_joint_delta_pos"


#variables
dataset_file1 = f'data/{env_id}/motionplanning/trajectory.{obs_mode}.{control_mode}.h5'
load_count = 1
task_id = 0.1
pred_horizon = 1
obs_horizon = 1
action_horizon = 1


# create dataset from file
dataset = StateDataset(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    task_id,
    load_count,
)

print(len(dataset))

dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

dataloader_iter = iter(dataloader)

for batch in dataloader_iter:
    print(batch["actions"][-1])

