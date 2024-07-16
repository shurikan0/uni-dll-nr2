from PIL import Image
import imageio
import zarr
import numpy as np

 # read from zarr dataset
dataset_root = zarr.open("data/pusht/pusht_cchi_v7_replay.zarr", 'r')

# float32, [0,1], (N,96,96,3)
train_image_data = dataset_root['data']['img']
print(dataset_root['data']['img'].shape)
print(dataset_root['data']['state'].shape)
#pil_images = []
#for i in range(200):
#    sequence = (train_image_data[i] * 255).astype(np.uint8)  
#    print(sequence.shape)
#    pil_images.append(sequence)
#imageio.mimsave(f"outputs/sequencedfp.gif", pil_images, fps=30)