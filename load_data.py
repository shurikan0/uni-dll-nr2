import h5py

#Open the H5 file in read mode
with h5py.File('data/PegInsert_ridiculously_small.h5', 'r') as file:
	print("Keys: %s" % file.keys())
	a_group_key = list(file.keys())[0]
	
	# Getting the data
	data = list(file[a_group_key])
	print(data)
