import h5py

dataset_file = ""
#Open the H5 file in read mode
with h5py.File(dataset_file, 'r') as file:
	print("Keys: %s" % file.keys())
	a_group_key = list(file.keys())[0]
	
	# Getting the data
	data = list(file[a_group_key])
	print(data)
 
    #Get first episode
	episode = file['traj_0']
	print("Episode keys: %s" % episode.keys())
	print("Episode obs: %s" % episode['obs'])
	print("Episode actions: %s" % episode['actions'])
	print("Episode terminated: %s" % episode['terminated'])
	print("Episode truncated: %s" % episode['truncated'])


