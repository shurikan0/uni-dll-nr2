import csv

# limit enviornment interaction to 200 steps before termination
env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode, render_mode='rgb_array')

max_steps = 400

num_episodes = 50
mean_success = 0 
mean_reward = 0
rewards = []
csv_file = f"{results_path}/results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Max Reward', 'Success'])
    print(f"Opened file {csv_file} for writing.")

    with tqdm(range(num_episodes), desc='Epoch') as episodes:

        for episode in episodes:
            
            # reset 
            obs, info = env.reset()
            obs = get_observations(obs)
            obs = convert_observation(obs, task_id[env_id])

            # save observations
            obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

            # save visualization
            imgs = []
            rewards = []
            done = False
            step_idx = 0
            unsuccessful = False


            with tqdm(total=max_steps, desc="Eval", leave=False) as pbar:
                while not done:
                    B = 1
                    # stack the last obs_horizon (2) number of observations
                    obs_seq = np.stack(obs_deque)
                
                    nobs = normalize_batch({'obs': torch.tensor(obs_seq, dtype=torch.float32)}, min_vals, max_vals, exclude_features)
                
                    # device transfer
                    #nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
                    nobs= nobs.to(device)

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
                    action_pred = naction[0] # we dont have to denormalize the action

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
                        obs = convert_observation(obs, task_id[env_id])

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
                            unsuccessful = True
                        if done:
                            break
            
            if not unsuccessful:
                mean_success += 1
            mean_reward += max(rewards)
            writer.writerow([episode + 1, max(rewards), int(not unsuccessful)])
        episodes.set_postfix(
            reward=mean_reward / (episode + 1),
            success=mean_success / (episode + 1)
        )

            
        

    print("Reward: ", mean_reward / num_episodes)
    print("Success: ", mean_success/num_episodes)
