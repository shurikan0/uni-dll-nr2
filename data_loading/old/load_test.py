import h5py

dataset_file = "data/pickcube2camera10.rgbd.pd_ee_delta_pos.h5"
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
	sensor_data = "sensor_data"
	for x in episode0['obs'][sensor_data].keys():
		print(x)
		print(episode0['obs'][sensor_data][x].keys())
		print(episode0['obs'][sensor_data][x]['rgb'])



