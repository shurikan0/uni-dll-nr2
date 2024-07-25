import h5py
import json
import numpy as np

obs_mode = 'state_dict'
control_mode = 'pd_joint_delta_pos'

#env_ids = ['PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1', 'PlugCharger-v1', 'PushCube-v1']
env_ids = ['PegInsertionSide-v2']
base_path = 'data'

for env_id in env_ids:
    # File names
    generated_path = f'{base_path}/{env_id}/motionplanning'
    original_h5_file = f'{generated_path}/trajectory.{obs_mode}.{control_mode}.h5'
    original_json_file = f'{generated_path}/trajectory.{obs_mode}.{control_mode}.json'
    training_h5_file = f'{generated_path}/training.{obs_mode}.{control_mode}.h5'
    validation_h5_file = f'{generated_path}/validation.{obs_mode}.{control_mode}.h5'
    training_json_file = f'{generated_path}/training.{obs_mode}.{control_mode}.json'
    validation_json_file = f'{generated_path}/validation.{obs_mode}.{control_mode}.json'

    with open(original_json_file, 'r') as f:
        json_data = json.load(f)

    with h5py.File(original_h5_file, 'r') as f:
        traj_datasets = [key for key in f.keys() if key.startswith('traj_')]

        np.random.shuffle(traj_datasets)

        split_point = 300#int(0.9 * len(traj_datasets))

        training_datasets = traj_datasets[:split_point]
        validation_datasets = traj_datasets[split_point:]

        with h5py.File(training_h5_file, 'w') as f_train:
            for key in training_datasets:
                f.copy(key, f_train)

        with h5py.File(validation_h5_file, 'w') as f_val:
            for key in validation_datasets:
                f.copy(key, f_val)

    dataset_indices = {f'traj_{i}': i for i in range(len(json_data['episodes']))}

    training_episodes = [json_data['episodes'][dataset_indices[key]] for key in training_datasets]
    validation_episodes = [json_data['episodes'][dataset_indices[key]] for key in validation_datasets]

    training_json = {
        'env_info': json_data['env_info'],
        'episodes': training_episodes
    }

    validation_json = {
        'env_info': json_data['env_info'],
        'episodes': validation_episodes
    }

    with open(training_json_file, 'w') as f:
        json.dump(training_json, f, indent=4)

    with open(validation_json_file, 'w') as f:
        json.dump(validation_json, f, indent=4)