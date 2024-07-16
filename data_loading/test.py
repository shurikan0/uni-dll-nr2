import numpy as np
from mani2_streaming_dataset import StreamingTrajectoryDataset2
from normal_dataset import NormalTrajectoryDataset
from torch.utils.data import DataLoader

from streaming_dataset import StreamingTrajectoryDataset
import imageio
from PIL import Image

# ManiSkill File
dataset_file1 = "data/trajectory.rgbd.pd_ee_delta_pos.h5"
dataset_file2 = "data/trajectory.rgbd.pd_ee_delta_pose.h5"

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
dataset = NormalTrajectoryDataset(
    dataset_file1,
    pred_horizon,
    obs_horizon,
    action_horizon,
    load_count,
    succes_only,
    normalize,
)

dataset1 = StreamingTrajectoryDataset(
    dataset_file2, pred_horizon, obs_horizon, action_horizon
)

dataset2 = StreamingTrajectoryDataset2(
    dataset_file2, pred_horizon, obs_horizon, action_horizon
)

# Debugging
# Dataset should be 3dim (batch, sequence, features)
# Can be used by Diffusion policy

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=10, num_workers=1, drop_last=drop_last)

batch_count = 10
iterator = iter(dataloader)
pil_images = []
while batch_count > 0:
    batch_count -= 1
    batch = next(iterator)
    
    for i in range(batch["obs_base_camera_rgb"].shape[0]):
        sequence = batch["obs_base_camera_rgb"][i]
        for j in range(sequence.shape[0]):
            pil_images.append(Image.fromarray(sequence[j].numpy().astype(np.uint8)))
        #print(sequence.shape)
        #pil_images.append(Image.fromarray(sequence[0].numpy().astype(np.uint8)))

imageio.mimsave(f"outputs/sequencemani3.gif", pil_images, fps=30)
        
    

batch_count2 = 10
iterator2 = iter(dataloader2)
pil_images2 = []
while batch_count2 > 0:
    batch_count2 -= 1
    batch = next(iterator2)
    
    for i in range(batch["rgbd"].shape[0]):
        sequence = batch["rgbd"][i]
        for j in range(sequence.shape[0]):
            image = sequence[j][ :, :, :3].numpy().astype(np.uint8)
            print(image.shape)
            pil_images2.append(Image.fromarray(image))
        
imageio.mimsave(f"outputs/sequencemani2.gif", pil_images2, fps=30)
      