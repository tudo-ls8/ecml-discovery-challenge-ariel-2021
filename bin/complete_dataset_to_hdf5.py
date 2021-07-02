#!/usr/bin/python3

import argparse
import os
import random
import sys

import h5py
import numpy as np
from tqdm import tqdm


def log_and_exit(message, code=1):
    print(message, file=sys.stderr)
    exit(code)


def remove_prefix(element, prefix):
    return element[len(prefix):]


def my_split(line):
    return float(line.split(': ')[1])


# TODO: We could provide an argument for the output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script helps to create a list of files for training and testing.')
    parser.add_argument('--prefix', type=str, default='params_train/')
    parser.add_argument('--seed', type=int, default=876)

    parser.add_argument('-o', '--output', type=str, default='train_test_set.h5_complete_name_params',
                        help='Output file.')

    parser.add_argument('--test-percentage', type=int, default=10)
    parser.add_argument('file', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.file):
        log_and_exit('sample file has to exist.', 1)

    with open(args.file) as file:
        content = file.readlines()

    # Remove prefix and strip newlines
    content = [remove_prefix(x.strip(), args.prefix) for x in content]

    # Extract planet ids
    planets = [x[:4] for x in content]
    # Remove duplicates
    planets = list(set(planets))
    n_planets = len(planets)

    print("We have {} planets.".format(n_planets))

    random.seed(args.seed)
    random.shuffle(planets)

    n_samples = len(content)
    ratio = args.test_percentage / 100

    # Split the planets into test and train planets based on
    # ratio = n_train / n_samples
    print('{} % of the samples are used for training.'.format(ratio * 100))

    n_planets_test = int(len(planets) * ratio)
    n_planets_train = n_planets - n_planets_test

    print('n_planets_train = {}, n_planets_test = {}'.format(n_planets_train, n_planets_test))

    n_test = n_planets_test * 100
    n_train = n_planets_train * 100
    print('n_train = {}, n_test = {}.'.format(n_train, n_test))

    # Split planets based on ratio
    train_planets = planets[:n_planets_train]
    test_planets = planets[n_planets_train:]

    x_train = np.empty(shape=(n_train, 55, 300), dtype=np.float64)
    y_train = np.empty(shape=(n_train, 55), dtype=np.float64)

    x_test = np.empty(shape=(n_test, 55, 300), dtype=np.float64)
    y_test = np.empty(shape=(n_test, 55), dtype=np.float64)

    x_train_star_temp = []
    x_train_star_logg = []
    x_train_star_rad = []
    x_train_star_mass = []
    x_train_k_mag = []
    x_train_period = []

    y_train_sma, y_train_incl = [], []

    x_test_star_temp = []
    x_test_star_logg = []
    x_test_star_rad = []
    x_test_star_mass = []
    x_test_k_mag = []
    x_test_period = []

    y_test_sma, y_test_incl = [], []

    x_train_planet_idx = []
    x_train_sun_spot = []
    x_train_photon_noise_idx = []

    x_test_planet_idx = []
    x_test_sun_spot = []
    x_test_photon_noise_idx = []

    idx = 0
    with tqdm(total=n_train) as pbar:
        for planet_idx in train_planets:
            for idx_0 in range(1, 11):
                for idx_1 in range(1, 11):
                    file_name = '{planet}_{id2:02d}_{id3:02d}.txt'.format(planet=planet_idx, id2=idx_0, id3=idx_1)
                    # print(file_name)
                    train_file = os.path.join('/mnt/data/noisy_train', file_name)
                    target_file = os.path.join('/mnt/data/params_train', file_name)
                    with open(train_file, 'r') as f:
                        lines = [x.strip() for x in f.readlines()]
                    # Remove the prefix
                    for line in range(6):
                        lines[line] = lines[line][2:]
                    x_train_star_temp.append(my_split(lines[0]))
                    x_train_star_logg.append(my_split(lines[1]))
                    x_train_star_rad.append(my_split(lines[2]))
                    x_train_star_mass.append(my_split(lines[3]))
                    x_train_k_mag.append(my_split(lines[4]))
                    x_train_period.append(my_split(lines[5]))
                    x_train_planet_idx.append(float(planet_idx))
                    x_train_sun_spot.append(idx_0)
                    x_train_photon_noise_idx.append(idx_1)
                    with open(target_file, 'r') as f:
                        lines = []
                        for i in range(2):
                            line = f.readline().strip()
                            lines.append(line[2:])
                        y_train_sma.append(my_split(lines[0]))
                        y_train_incl.append(my_split(lines[1]))

                    x_train[idx,] = np.loadtxt(train_file, skiprows=6, delimiter='\t', dtype=np.float64)
                    y_train[idx] = np.loadtxt(target_file, skiprows=2, delimiter='\t', dtype=np.float64)
                    idx += 1
                    pbar.update(1)
    assert idx == n_train

    idx = 0
    with tqdm(total=n_test) as pbar:
        for planet_idx in test_planets:
            for idx_0 in range(1, 11):
                for idx_1 in range(1, 11):
                    file_name = '{planet}_{id2:02d}_{id3:02d}.txt'.format(planet=planet_idx, id2=idx_0, id3=idx_1)
                    train_file = os.path.join('/mnt/data/noisy_train', file_name)
                    target_file = os.path.join('/mnt/data/params_train', file_name)
                    with open(train_file, 'r') as f:
                        lines = [x.strip() for x in f.readlines()]
                    # Remove the prefix
                    for line in range(6):
                        lines[line] = lines[line][2:]
                    x_test_star_temp.append(my_split(lines[0]))
                    x_test_star_logg.append(my_split(lines[1]))
                    x_test_star_rad.append(my_split(lines[2]))
                    x_test_star_mass.append(my_split(lines[3]))
                    x_test_k_mag.append(my_split(lines[4]))
                    x_test_period.append(my_split(lines[5]))
                    x_test_planet_idx.append(float(planet_idx))
                    x_test_sun_spot.append(idx_0)
                    x_test_photon_noise_idx.append(idx_1)

                    with open(target_file, 'r') as f:
                        lines = []
                        for i in range(2):
                            line = f.readline().strip()
                            lines.append(line[2:])
                        y_test_sma.append(my_split(lines[0]))
                        y_test_incl.append(my_split(lines[1]))

                    x_test[idx,] = np.loadtxt(train_file, skiprows=6, delimiter='\t', dtype=np.float64)
                    y_test[idx] = np.loadtxt(target_file, skiprows=2, delimiter='\t', dtype=np.float64)
                    idx += 1
                    pbar.update(1)
    assert idx == n_test

    # Shuffle the data.
    train_perm = np.random.permutation(n_train)
    test_perm = np.random.permutation(n_test)

    # Shuffle train
    x_train = x_train[train_perm]
    x_train_star_temp = np.array(x_train_star_temp)
    x_train_star_temp = x_train_star_temp[train_perm]

    x_train_star_logg = np.array(x_train_star_logg)
    x_train_star_logg = x_train_star_logg[train_perm]

    x_train_star_rad = np.array(x_train_star_rad)
    x_train_star_rad = x_train_star_rad[train_perm]

    x_train_star_mass = np.array(x_train_star_mass)
    x_train_star_mass = x_train_star_mass[train_perm]

    x_train_k_mag = np.array(x_train_k_mag)
    x_train_k_mag = x_train_k_mag[train_perm]

    x_train_period = np.array(x_train_period)
    x_train_period = x_train_period[train_perm]

    y_train = y_train[train_perm]

    y_train_incl = np.array(y_train_incl)
    y_train_incl = y_train_incl[train_perm]

    y_train_sma = np.array(y_train_sma)
    y_train_sma = y_train_sma[train_perm]

    ####################

    x_test = x_test[test_perm]
    x_test_star_temp = np.array(x_test_star_temp)
    x_test_star_temp = x_test_star_temp[test_perm]

    x_test_star_logg = np.array(x_test_star_logg)
    x_test_star_logg = x_test_star_logg[test_perm]

    x_test_star_rad = np.array(x_test_star_rad)
    x_test_star_rad = x_test_star_rad[test_perm]

    x_test_star_mass = np.array(x_test_star_mass)
    x_test_star_mass = x_test_star_mass[test_perm]

    x_test_k_mag = np.array(x_test_k_mag)
    x_test_k_mag = x_test_k_mag[test_perm]

    x_test_period = np.array(x_test_period)
    x_test_period = x_test_period[test_perm]

    y_test = y_test[test_perm]

    y_test_incl = np.array(y_test_incl)
    y_test_incl = y_test_incl[test_perm]

    y_test_sma = np.array(y_test_sma)
    y_test_sma = y_test_sma[test_perm]

    x_train_planet_idx = np.array(x_train_planet_idx)
    x_train_planet_idx = x_train_planet_idx[train_perm]

    x_train_sun_spot = np.array(x_train_sun_spot)
    x_train_sun_spot = x_train_sun_spot[train_perm]

    x_train_photon_noise_idx = np.array(x_train_photon_noise_idx)
    x_train_photon_noise_idx = x_train_photon_noise_idx[train_perm]

    x_test_planet_idx = np.array(x_test_planet_idx)
    x_test_planet_idx = x_test_planet_idx[test_perm]

    x_test_sun_spot = np.array(x_test_sun_spot)
    x_test_sun_spot = x_test_sun_spot[test_perm]

    x_test_photon_noise_idx = np.array(x_test_photon_noise_idx)
    x_test_photon_noise_idx = x_test_photon_noise_idx[test_perm]

    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('x_train', data=x_train)
        hf.create_dataset('y_train', data=y_train)
        # Additional train params
        hf.create_dataset('x_train_star_temp', data=x_train_star_temp)
        hf.create_dataset('x_train_star_logg', data=x_train_star_logg)
        hf.create_dataset('x_train_star_rad', data=x_train_star_rad)
        hf.create_dataset('x_train_star_mass', data=x_train_star_mass)
        hf.create_dataset('x_train_k_mag', data=x_train_k_mag)
        hf.create_dataset('x_train_period', data=x_train_period)
        # Additional target params
        hf.create_dataset('y_train_incl', data=y_train_incl)
        hf.create_dataset('y_train_sma', data=y_train_sma)

        hf.create_dataset('x_test', data=x_test)
        hf.create_dataset('y_test', data=y_test)
        # Additional test params
        hf.create_dataset('x_test_star_temp', data=x_test_star_temp)
        hf.create_dataset('x_test_star_logg', data=x_test_star_logg)
        hf.create_dataset('x_test_star_rad', data=x_test_star_rad)
        hf.create_dataset('x_test_star_mass', data=x_test_star_mass)
        hf.create_dataset('x_test_k_mag', data=x_test_k_mag)
        hf.create_dataset('x_test_period', data=x_test_period)
        # Additional target params
        hf.create_dataset('y_test_incl', data=y_test_incl)
        hf.create_dataset('y_test_sma', data=y_test_sma)

        # data set meta data
        hf.create_dataset('x_train_planet_idx', data=x_train_planet_idx)
        hf.create_dataset('x_train_sun_spot_idx', data=x_train_sun_spot)
        hf.create_dataset('x_train_photon_noise_idx', data=x_train_photon_noise_idx)

        hf.create_dataset('x_test_planet_idx', data=x_test_planet_idx)
        hf.create_dataset('x_test_sun_spot_idx', data=x_test_sun_spot)
        hf.create_dataset('x_test_photon_noise_idx', data=x_test_photon_noise_idx)
