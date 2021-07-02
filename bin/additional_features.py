#!/usr/bin/python3

import h5py

import numpy as np

from sklearn.preprocessing import MinMaxScaler


def merge_data(x, star_temp, star_log, star_rad, star_mass, k_mag, period):
    stack_additional_params = np.vstack((star_temp, star_log, star_rad, star_mass, k_mag, period))
    scaler = MinMaxScaler()
    transormed = scaler.fit_transform(stack_additional_params.T)
    tmp = np.repeat(transormed[:, np.newaxis, :], 55, axis=1)
    res = np.concatenate((x, tmp), axis=2)
    return res


with h5py.File('/mnt/data/train_test_set.h5', 'r') as hf:
    x_train = hf['x_train'][:]
    y_train = hf['y_train'][:]

    x_test = hf['x_test'][:]
    y_test = hf['y_test'][:]

    x_train_star_temp = hf['x_train_star_temp'][:]
    x_train_star_logg = hf['x_train_star_logg'][:]
    x_train_star_rad = hf['x_train_star_rad'][:]
    x_train_star_mass = hf['x_train_star_mass'][:]
    x_train_k_mag = hf['x_train_k_mag'][:]
    x_train_period = hf['x_train_period'][:]

    x_test_star_temp = hf['x_test_star_temp'][:]
    x_test_star_logg = hf['x_test_star_logg'][:]
    x_test_star_rad = hf['x_test_star_rad'][:]
    x_test_star_mass = hf['x_test_star_mass'][:]
    x_test_k_mag = hf['x_test_k_mag'][:]
    x_test_period = hf['x_test_period'][:]

merged = merge_data(x_train, x_train_star_temp, x_train_star_logg, x_train_star_rad,
                    x_train_star_mass, x_train_k_mag, x_train_period)
# print(x_train_star_temp)

# TODO: Normalize this values to [0, 1] (MinMaxScaler?)
#
stack_additional_params = np.vstack((x_train_star_temp, x_train_star_logg, x_train_star_rad,
                                     x_train_star_mass, x_train_k_mag, x_train_period))
print(stack_additional_params)
print(stack_additional_params.shape)

scaler = MinMaxScaler()
# TODO: Is this correct????
transormed = scaler.fit_transform(stack_additional_params.T)

b = np.repeat(transormed[:, np.newaxis, :], 55, axis=1)

# for i in range(stack_additional_params.shape[0]):
#
#    stack_additional_params[i] = scaler.fit_transform(stack_additional_params[i])

print(transormed)
print(transormed.shape)

print(transormed[0])

print(b)
print(b.shape)

print(b[0])

res = np.concatenate((x_train, b), axis=2)

print(res.shape)
print(res[0][0])
print(res[0][1])
print('-' * 80)
print(res[1])

print('-' * 80)
print(merged.shape)
print(merged[0][0])
print(merged[0][1])
