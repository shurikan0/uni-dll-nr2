import torch
import numpy as np
import gymnasium as gym
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image
from IPython.display import display, Image as IPImage
import io

from unet_module import ConditionalUnet1D
from data_loading.dataset_utils import get_observations, convert_observation

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

env_id = "PickCube-v1"
obs_mode = "state_dict"
control_mode = "pd_joint_pos"
action_dim = 8  # if control_mode = "pd_joint_pos"
obs_dim = 35

env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode, render_mode='human')


noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

model_path = '../Data/Checkpoints/model_pickcube_state_dict_pd_joint_pos.pt'
state_dict = torch.load(model_path, map_location='cuda')
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict["model_state_dict"])
print('Pretrained weights loaded.')

# limit enviornment interaction to 200 steps before termination
result_path = "../Data/Results/videos"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

max_steps = 200

# reset
obs, info = env.reset()
print(env.action_space)
obs = get_observations(obs)
obs = convert_observation(obs)

# save observations
obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

# save visualization
imgs = []
rewards = []
done = False
step_idx = 0

with tqdm(total=max_steps, desc="Eval") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        #TODO: normalize observation
        #nobs = normalize_data(obs_seq, stats=stats['obs'])
        nobs = obs_seq
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        #TODO: unnormalize action
        #action_pred = unnormalize_data(naction, stats=stats['action'])
        action_pred = naction

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])

            # process observation
            # From the observation dictionary, we concatenate all the observations
            # as done in the training data
            obs = get_observations(obs)
            obs = convert_observation(obs)

            # save observations
            obs_deque.append(obs)

            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render())

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break


# print out the maximum target coverage
print('Score: ', max(rewards))


print("Image shape:", imgs[0].shape)  # Print shape to check if it's (H, W, 3) for RGB
print("Image dtype:", imgs[0].dtype)  # Should be uint8

images = [Image.fromarray(img.squeeze(0).cpu().numpy()) for img in imgs]

# Save to a bytes buffer
buffer = io.BytesIO()
images[0].save(buffer, format='GIF', save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)
buffer.seek(0)

# Save to a file
with open('drive/MyDrive/Data/animation.gif', 'wb') as f:
    f.write(buffer.getvalue())

# Display the GIF (optional)
display(IPImage(data=buffer.getvalue()))