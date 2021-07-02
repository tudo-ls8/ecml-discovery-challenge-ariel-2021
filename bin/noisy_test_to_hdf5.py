#!/usr/bin/python3
import os

import h5py
import numpy as np
from tqdm import tqdm

data_dir = '/mnt/data/'


def my_split(line):
    return float(line.split(': ')[1])


# TODO: Directories sollten konfigurierbar sein.

if __name__ == '__main__':
    # Read all files from noisy_test.txt

    with open('noisy_test.txt') as file:
        content = file.readlines()
    content = [x.strip() for x in content]

    n_samples = len(content)

    x_test = np.empty(shape=(n_samples, 55, 300), dtype=np.float64)
    x_test_star_temp = np.empty(shape=(n_samples, 1), dtype=np.float64)
    x_test_star_logg = np.empty(shape=(n_samples, 1), dtype=np.float64)
    x_test_star_rad = np.empty(shape=(n_samples, 1), dtype=np.float64)
    x_test_star_mass = np.empty(shape=(n_samples, 1), dtype=np.float64)
    x_test_star_k_mag = np.empty(shape=(n_samples, 1), dtype=np.float64)
    x_test_period = np.empty(shape=(n_samples, 1), dtype=np.float64)

    planet_idx = np.empty(shape=(n_samples, 1))
    sun_spot_idx = np.empty(shape=(n_samples, 1))
    photon_noise_idx = np.empty(shape=(n_samples, 1))

    for arr_idx, file_name in enumerate(tqdm(content)):
        test_file = os.path.join(data_dir, file_name)
        x_test[arr_idx,] = np.loadtxt(test_file, skiprows=6, delimiter='\t', dtype=np.float64)

        # Read additional parameters.
        with open(test_file, 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            # Remove the prefix
            for line in range(6):
                lines[line] = lines[line][2:]
            x_test_star_temp[arr_idx] = my_split(lines[0])
            x_test_star_logg[arr_idx] = my_split(lines[1])
            x_test_star_rad[arr_idx] = my_split(lines[2])
            x_test_star_mass[arr_idx] = my_split(lines[3])
            x_test_star_k_mag[arr_idx] = my_split(lines[4])
            x_test_period[arr_idx] = my_split(lines[5])

            planet_idx[arr_idx] = file_name.split('/')[1].split('_')[0]
            sun_spot_idx[arr_idx] = file_name.split('/')[1].split('_')[1]
            photon_noise_idx[arr_idx] = file_name.split('/')[1].split('_')[2][:2]

    with h5py.File('/mnt/data/noisy_test.h5_named_params', 'w') as hf:
        hf.create_dataset("x_prediction", data=x_test)
        # Write additional parameters.
        hf.create_dataset('x_prediction_star_temp', data=x_test_star_temp)
        hf.create_dataset('x_prediction_star_logg', data=x_test_star_logg)
        hf.create_dataset('x_prediction_star_rad', data=x_test_star_rad)
        hf.create_dataset('x_prediction_star_mass', data=x_test_star_mass)
        hf.create_dataset('x_prediction_star_k_mag', data=x_test_star_k_mag)
        hf.create_dataset('x_prediction_period', data=x_test_period)

        hf.create_dataset('x_prediction_planet_idx', data=planet_idx)
        hf.create_dataset('x_prediction_sun_spot_idx', data=sun_spot_idx)
        hf.create_dataset('x_prediction_photon_noise_idx', data=photon_noise_idx)
