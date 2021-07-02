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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script helps to create a list of files for training and testing.')
    parser.add_argument('--prefix', type=str, default='params_train/')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--output', default='', help='Output file')

    parser.add_argument('n_samples', type=int, help='Number of samples from training data for training.')
    parser.add_argument('n_train', type=int, help='Number of n_samples, which should be used for training.')
    parser.add_argument('file', type=str)

    args = parser.parse_args()

    # Verify preconditions
    if args.n_train >= args.n_samples:
        log_and_exit('n_train has to be less than n_samples.', 1)
    if not os.path.exists(args.file):
        log_and_exit('sample file has to exist.', 1)
    if args.output == '' or os.path.isfile(args.output):
        log_and_exit('The output file \'{}\' must be specified and it must not exist'.format(args.output))

    with open(args.file) as file:
        content = file.readlines()

    # Remove prefix and strip newlines
    content = [remove_prefix(x.strip(), args.prefix) for x in content]
    print('Found content {}, {}, {}...'.format(content[0], content[1], content[2]))

    # Extract planet ids
    planets = [x[:4] for x in content]
    # Remove duplicates
    planets = list(set(planets))
    n_planets = len(planets)

    random.seed(args.seed)
    random.shuffle(planets)

    n_samples = args.n_samples
    n_train = args.n_train
    n_test = n_samples - n_train

    # Split the planets into test and train planets based on
    ratio = n_train / n_samples
    print('{} % of the samples are used for training.'.format(ratio))

    n_planets_train = int(len(planets) * ratio)
    n_planets_test = n_planets - n_planets_train

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

    y_train_sma = []
    y_train_incl = []

    x_test_star_temp = []
    x_test_star_logg = []
    x_test_star_rad = []
    x_test_star_mass = []
    x_test_k_mag = []
    x_test_period = []

    y_test_sma = []
    y_test_incl = []

    planet_train = []
    planet_test = []
    sn_train = [] # stellar noise IDs
    sn_test = []
    pn_train = [] # photon noise IDs
    pn_test = []


    # Generative process:
    for idx in tqdm(range(n_train)):
        # Sample planet
        planet = train_planets[random.randint(0, n_planets_train - 1)]
        stellar_noise_id = random.randint(1, 10)
        photon_noise_id = random.randint(1, 10)
        file_name = '{planet}_{id2:02d}_{id3:02d}.txt'.format(planet=planet, id2=stellar_noise_id, id3=photon_noise_id)
        assert file_name in content

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

        with open(target_file, 'r') as f:
            lines = []
            for i in range(2):
                line = f.readline().strip()
                lines.append(line[2:])
            y_train_sma.append(my_split(lines[0]))
            y_train_incl.append(my_split(lines[1]))

        x_train[idx,] = np.loadtxt(train_file, skiprows=6, delimiter='\t', dtype=np.float64)
        y_train[idx] = np.loadtxt(target_file, skiprows=2, delimiter='\t', dtype=np.float64)

        planet_train.append(int(planet))
        sn_train.append(stellar_noise_id)
        pn_train.append(photon_noise_id)

    for idx in tqdm(range(n_test)):
        # Sample planet
        planet = test_planets[random.randint(0, n_planets_test - 1)]
        stellar_noise_id = random.randint(1, 10)
        photon_noise_id = random.randint(1, 10)
        file_name = '{planet}_{id2:02d}_{id3:02d}.txt'.format(planet=planet, id2=stellar_noise_id, id3=photon_noise_id)
        assert file_name in content

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

        with open(target_file, 'r') as f:
            lines = []
            for i in range(2):
                line = f.readline().strip()
                lines.append(line[2:])
            y_test_sma.append(my_split(lines[0]))
            y_test_incl.append(my_split(lines[1]))

        x_test[idx,] = np.loadtxt(train_file, skiprows=6, delimiter='\t', dtype=np.float64)
        y_test[idx] = np.loadtxt(target_file, skiprows=2, delimiter='\t', dtype=np.float64)

        planet_test.append(int(planet))
        sn_test.append(stellar_noise_id)
        pn_test.append(photon_noise_id)

    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('x_train', data=x_train)
        hf.create_dataset('y_train', data=y_train)
        # Additional train params
        hf.create_dataset('x_train_star_temp', data=np.array(x_train_star_temp))
        hf.create_dataset('x_train_star_logg', data=np.array(x_train_star_logg))
        hf.create_dataset('x_train_star_rad', data=np.array(x_train_star_rad))
        hf.create_dataset('x_train_star_mass', data=np.array(x_train_star_mass))
        hf.create_dataset('x_train_k_mag', data=np.array(x_train_k_mag))
        hf.create_dataset('x_train_period', data=np.array(x_train_period))
        # Additional target params
        hf.create_dataset('y_train_incl', data=np.array(y_train_incl))
        hf.create_dataset('y_train_sma', data=np.array(y_train_sma))

        hf.create_dataset('x_test', data=x_test)
        hf.create_dataset('y_test', data=y_test)
        # Additional test params
        hf.create_dataset('x_test_star_temp', data=np.array(x_test_star_temp))
        hf.create_dataset('x_test_star_logg', data=np.array(x_test_star_logg))
        hf.create_dataset('x_test_star_rad', data=np.array(x_test_star_rad))
        hf.create_dataset('x_test_star_mass', data=np.array(x_test_star_mass))
        hf.create_dataset('x_test_k_mag', data=np.array(x_test_k_mag))
        hf.create_dataset('x_test_period', data=np.array(x_test_period))
        # Additional target params
        hf.create_dataset('y_test_incl', data=np.array(y_test_incl))
        hf.create_dataset('y_test_sma', data=np.array(y_test_sma))

        hf.create_dataset('x_train_planet_idx', data=np.array(planet_train))
        hf.create_dataset('x_train_sun_spot_idx', data=np.array(sn_train))
        hf.create_dataset('x_train_photon_noise_idx', data=np.array(pn_train))

        hf.create_dataset('x_test_planet_idx', data=np.array(planet_test))
        hf.create_dataset('x_test_sun_spot_idx', data=np.array(sn_test))
        hf.create_dataset('x_test_photon_noise_idx', data=np.array(sn_test))
