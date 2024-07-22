import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import os

from data_loading.state_dataset import StateDataset
from unet_module import ConditionalUnet1D

# Parameters
dataset_path = '../Data/Training/Generated/PlugCharger-v1/motionplanning/Data1000.state_dict.pd_joint_pos.h5'
model_path = "../Data/Checkpoints/model_plugcharger_state_dict_pd_joint_pos.pt"
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

# Create dataset and dataloader
dataset = StateDataset(
    dataset_file=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    device=None
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=1,
    persistent_workers=True,
    shuffle=True
)

# Example batch to get dimensions
batch = next(iter(dataloader))
obs_dim = batch['obs'].shape[-1]
action_dim = batch['actions'].shape[-1]

# Create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
)

# Example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# Noise prediction network
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1)
)

# Denoise action
denoised_action = noised_action - noise

# Diffusion scheduler
num_diffusion_iters = 10
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

# Device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

num_epochs = 10

# Exponential Moving Average
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75
)

# Optimizer
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6
)

# Learning rate scheduler
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nobs = nbatch['obs'].to(device)
                naction = nbatch['actions'].to(device)
                B = nobs.shape[0]

                obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)
                noise = torch.randn(naction.shape, device=device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                loss = nn.functional.mse_loss(noise_pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(noise_pred_net.parameters())

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

# Ensure the directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the model, optimizer, and scheduler states
torch.save({
    'model_state_dict': ema_noise_pred_net.state_dict(),
    'ema_model_state_dict': ema.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    'epoch': epoch_idx,
    'loss': np.mean(epoch_loss),  # Save the average loss of the last epoch
}, model_path)