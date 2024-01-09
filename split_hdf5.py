import h5py
import numpy as np
import os

# Path to your HDF5 file
hdf5_file_path = '/home/lix0i/Xiang/3DCoMPaT/BPNet/hdf5_50/test_coarse.hdf5'

# Directory where you want to save the files
save_dir = 'pt_files_test'
os.makedirs(save_dir, exist_ok=True)

# Open the HDF5 file
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Determine the number of items (assuming all keys have the same length)
    num_items = len(hdf5_file[list(hdf5_file.keys())[0]])

    # Iterate over each item index
    for i in range(num_items):
        # Collect data from each key for this item
        data_to_save = {key: hdf5_file[key][i] for key in hdf5_file.keys()}

        # Save the data into a single file
        file_path = os.path.join(save_dir, f'item_{i}.npz')
        np.savez(file_path, **data_to_save)

print("Data saved successfully.")