import numpy as np
import matplotlib.pyplot as plt

path = 'data/PickCube-v1_pd_joint_delta_pos_loss.npz'

# Load the data from NPZ file
with np.load(path) as data:
    training_losses = data['training_losses']
    validation_losses = data['validation_losses']

# Number of epochs
num_epochs = training_losses.shape[0]

# Calculate x-axis positions for each step, based on epoch boundaries
x_positions_train = []
x_positions_val = []
for epoch in range(num_epochs):
    epoch_start = epoch 
    epoch_end = epoch + 1

    # Training steps within the epoch
    num_steps_train = len(training_losses[epoch])
    step_positions_train = np.linspace(epoch_start, epoch_end, num_steps_train, endpoint=False)
    x_positions_train.extend(step_positions_train)

    # Validation steps within the epoch (if available)
    if epoch < len(validation_losses):  
        num_steps_val = len(validation_losses[epoch])
        step_positions_val = np.linspace(epoch_start, epoch_end, num_steps_val, endpoint=False)
        x_positions_val.extend(step_positions_val)

x_positions_train = np.array(x_positions_train)
x_positions_val = np.array(x_positions_val)

# Smoothing for better visualization (adjust window size as needed)
def smooth_curve(data, x_positions, window_size=10):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    valid_x_positions = x_positions[window_size // 2:-window_size // 2 + 1]
    if len(smoothed_data) > len(valid_x_positions):  # handle case when data is shorter than window
        smoothed_data = smoothed_data[:len(valid_x_positions)]
    return valid_x_positions, smoothed_data  

x_positions_train, smoothed_training_losses = smooth_curve(training_losses.flatten(), x_positions_train)
x_positions_val, smoothed_validation_losses = smooth_curve(validation_losses.flatten(), x_positions_val)


# Create plot
plt.figure(figsize=(12, 6))  # Adjust figure size
plt.plot(x_positions_train, smoothed_training_losses, label='Training Loss', color='blue', linestyle='-')
plt.plot(x_positions_val, smoothed_validation_losses, label='Validation Loss', color='orange', linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss per Epoch for PickCube', fontsize=14)
plt.xticks(range(num_epochs+1))  # Set x ticks at integer epoch values
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--')  # Grid only on y-axis for better readability
plt.tight_layout()
plt.savefig('outputs/PickCube-v1_pd_joint_delta_pos_loss_epochs.png')
plt.show()
