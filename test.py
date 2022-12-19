import h5py
import numpy as np
import os

output_file_path = './data/tacos/c3d/'

if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

f = h5py.File('./data/tacos/tall_c3d_features.hdf5', 'r')
keys = list(f.keys())
for key in keys:
    npy = f.get(key)
    filename = output_file_path + key
    np.save(filename, npy)
    