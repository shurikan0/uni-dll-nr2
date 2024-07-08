import h5py

dataset_file = "data/trajectory.pointcloud.pd_joint_delta_pos.h5"
#Open the H5 file in read mode
with h5py.File(dataset_file, 'r') as file:
	print("Keys: %s" % file.keys())
	a_group_key = list(file.keys())[0]
	
	# Getting the data
	data = list(file[a_group_key])
	print(data)
 
    #Get first episode
	episode0 = file['traj_0']
	episode1 = file['traj_1']
	episode2 = file['traj_2']
	episode3 = file['traj_3']


	print("Episode keys: %s" % episode0.keys())
	print("Episode obs: %s" % episode0['obs'])
	print("Episode actions: %s" % episode0['actions'])
	print("Episode terminated: %s" % episode0['terminated'])
	print("Episode truncated: %s" % episode0['truncated'])
	print("Observation keys: %s" % episode0['obs'].keys())
	for x in episode0['obs']['pointcloud']:
		print(x)
		print(episode0['obs']['pointcloud'][x].shape)

	print(episode0['obs']['pointcloud']["xyzw"][0].shape)
	print(episode0['obs']['pointcloud']["xyzw"][74].shape)
	print(episode1['obs']['pointcloud']["xyzw"][0].shape)
	print(episode1['obs']['pointcloud']["xyzw"][40].shape)
	print(episode2['obs']['pointcloud']["xyzw"][0].shape)
	print(episode2['obs']['pointcloud']["xyzw"][40].shape)
	print(episode3['obs']['pointcloud']["xyzw"][0].shape)
	print(episode3['obs']['pointcloud']["xyzw"][40].shape)


