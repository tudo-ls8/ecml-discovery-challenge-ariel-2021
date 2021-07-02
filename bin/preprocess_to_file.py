import argparse
import h5py
import os

import numpy as np
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler

DATA_PATH_DEFAULT = '/mnt/data/train_test_set.h5_small_named_params'
DATA_PATH_COMPLETE = '/mnt/data/train_test_set.h5_complete_name_params'  # TODO fix this typo
PREDICTION_DATA_PATH = '/mnt/data/noisy_test.h5_named_params'


def get_data(complete_data=False, return_additional_params=True, transpose_x=True):
    # read the raw data
    data_path = DATA_PATH_COMPLETE if complete_data else DATA_PATH_DEFAULT
    with h5py.File(data_path, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]

    # also read the prediction data, which is at least used for normalization
    with h5py.File(PREDICTION_DATA_PATH, 'r') as hf:
        x_prediction = hf['x_prediction'][:]

    # fulfill the shape convention (n_samples, n_timesteps, n_channels)
    if transpose_x:
        x_train = np.swapaxes(x_train, 1, 2)
        x_test = np.swapaxes(x_test, 1, 2)
        x_prediction = np.swapaxes(x_prediction, 1, 2)

    print('Read the raw data from {} and {}:\nx_train {}, y_train {}, x_test {}, y_test {}, x_prediction {}'.format(
        data_path,
        PREDICTION_DATA_PATH,
        x_train.shape,
        y_train.shape,
        x_test.shape,
        y_test.shape,
        x_prediction.shape
    ))

    # return the 'raw' data without additional parameters
    if not return_additional_params:
        return x_train, y_train, x_test, y_test, x_prediction

    # read auxiliary parameters and labels
    with h5py.File(data_path, 'r') as hf:
        aux_train = np.hstack((
            hf['x_train_star_temp'][:].reshape(-1, 1),
            hf['x_train_star_logg'][:].reshape(-1, 1),
            hf['x_train_star_rad'][:].reshape(-1, 1),
            hf['x_train_star_mass'][:].reshape(-1, 1),
            hf['x_train_k_mag'][:].reshape(-1, 1),
            hf['x_train_period'][:].reshape(-1, 1),
            hf['x_train_sun_spot_idx'][:].reshape(-1, 1),
            hf['x_train_photon_noise_idx'][:].reshape(-1, 1)
        ))
        aux_test = np.hstack((
            hf['x_test_star_temp'][:].reshape(-1, 1),
            hf['x_test_star_logg'][:].reshape(-1, 1),
            hf['x_test_star_rad'][:].reshape(-1, 1),
            hf['x_test_star_mass'][:].reshape(-1, 1),
            hf['x_test_k_mag'][:].reshape(-1, 1),
            hf['x_test_period'][:].reshape(-1, 1),
            hf['x_test_sun_spot_idx'][:].reshape(-1, 1),
            hf['x_test_photon_noise_idx'][:].reshape(-1, 1)
        ))
        aux_y_train = np.hstack((
            hf['y_train_sma'][:].reshape(-1, 1),
            hf['y_train_incl'][:].reshape(-1, 1)
        ))
        aux_y_test = np.hstack((
            hf['y_test_sma'][:].reshape(-1, 1),
            hf['y_test_incl'][:].reshape(-1, 1)
        ))
        planet_train = hf['x_train_planet_idx'][:].reshape(-1, 1)
        planet_test = hf['x_test_planet_idx'][:].reshape(-1, 1)

    with h5py.File(PREDICTION_DATA_PATH, 'r') as hf:
        aux_prediction = np.hstack((
            hf['x_prediction_star_temp'][:].reshape(-1, 1),
            hf['x_prediction_star_logg'][:].reshape(-1, 1),
            hf['x_prediction_star_rad'][:].reshape(-1, 1),
            hf['x_prediction_star_mass'][:].reshape(-1, 1),
            hf['x_prediction_star_k_mag'][:].reshape(-1, 1),
            hf['x_prediction_period'][:].reshape(-1, 1),
            hf['x_prediction_sun_spot_idx'][:].reshape(-1, 1),
            hf['x_prediction_photon_noise_idx'][:].reshape(-1, 1)
        ))
        planet_prediction = hf['x_prediction_planet_idx'][:].reshape(-1, 1)

    print('Read the auxiliary parameters:\naux_train {}, aux_test {}, aux_prediction {}'.format(
        aux_train.shape,
        aux_test.shape,
        aux_prediction.shape
    ))

    # scale the stuff
    aux_scaler = MinMaxScaler()
    aux_scaler.fit(np.concatenate((aux_train, aux_test, aux_prediction), axis=0))  # fit to all data sets!
    aux_train = aux_scaler.transform(aux_train)
    aux_test = aux_scaler.transform(aux_test)
    aux_prediction = aux_scaler.transform(aux_prediction)

    # also scale the labels
    aux_y_scaler = MinMaxScaler()
    aux_y_scaler.fit(np.concatenate((aux_y_train, aux_y_test), axis=0))
    aux_y_train = aux_y_scaler.transform(aux_y_train)
    aux_y_test = aux_y_scaler.transform(aux_y_test)

    return x_train, y_train, x_test, y_test, x_prediction, aux_train, aux_test, aux_prediction, aux_y_train, aux_y_test, planet_train, planet_test, planet_prediction


def scale_time_series(x_train, x_test, x_prediction, return_mean_std=True):
    # scale each time series (individually and on each channel) to zero mean and unit variance
    print('Scaling each time series to zero mean and unit variance')
    n_channels = x_train.shape[2]

    def fit_transform_scale(x):
        mean_x = np.mean(x, axis=1, keepdims=True)  # like in tslearn.TimeSeriesScalerMeanVariance
        std_x = np.std(x, axis=1, keepdims=True)
        std_x[std_x == 0.] == 1.
        skew_x = stats.skew(x, axis=1)  # the third moment, not used for transformation but as an additional feature
        kurt_x = stats.kurtosis(x, axis=1)  # the fourth moment
        x = (x - mean_x) / std_x  # zero mean, unit standard deviation
        return x, np.concatenate((
            np.squeeze(mean_x),
            np.squeeze(std_x),
            skew_x,
            kurt_x
        ), axis=1)  # transformed data plus additional features

    x_train, ms_train = fit_transform_scale(x_train)
    x_test, ms_test = fit_transform_scale(x_test)
    x_prediction, ms_prediction = fit_transform_scale(x_prediction)

    # return the mean and std of each time series as additional features
    if return_mean_std:
        print('Preparing means, variances, skews, and kurtosises as additional features')
        ms_scaler = MinMaxScaler()
        ms_scaler.fit(np.concatenate((ms_train, ms_test, ms_prediction), axis=0))
        ms_train = ms_scaler.transform(ms_train).reshape(ms_train.shape[0], -1, n_channels)
        ms_test = ms_scaler.transform(ms_test).reshape(ms_test.shape[0], -1, n_channels)
        ms_prediction = ms_scaler.transform(ms_prediction).reshape(ms_prediction.shape[0], -1, n_channels)
        return x_train, x_test, x_prediction, ms_train, ms_test, ms_prediction
    else:
        return x_train, x_test, x_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-process the data and store the results in an intermediate file',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--complete-data', help='Use the complete data set', action='store_true')
    parser.add_argument('output', help='The path to write to, which must not exist.')
    args = parser.parse_args()

    if os.path.isfile(args.output):
        raise ValueError('The output file {} must not exist'.format(args.output))

    # scale each time series to zero mean and unit variance, use the first four moments of each series as auxiliary features
    x_train, y_train, x_test, y_test, x_prediction, aux_train, aux_test, aux_prediction, aux_y_train, aux_y_test, planet_train, planet_test, planet_prediction = get_data(
        complete_data=args.complete_data)
    x_train, x_test, x_prediction, ms_train, ms_test, ms_prediction = scale_time_series(x_train, x_test, x_prediction)

    print('Now writing to {}'.format(args.output))
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('x_train', data=x_train)
        hf.create_dataset('y_train', data=y_train)

        hf.create_dataset('x_test', data=x_test)
        hf.create_dataset('y_test', data=y_test)

        hf.create_dataset('x_prediction', data=x_prediction)

        hf.create_dataset('aux_train', data=aux_train)
        hf.create_dataset('aux_test', data=aux_test)
        hf.create_dataset('aux_prediction', data=aux_prediction)

        hf.create_dataset('ms_train', data=ms_train)
        hf.create_dataset('ms_test', data=ms_test)
        hf.create_dataset('ms_prediction', data=ms_prediction)

        hf.create_dataset('aux_y_train', data=aux_y_train)
        hf.create_dataset('aux_y_test', data=aux_y_test)

        hf.create_dataset('planet_train', data=planet_train)
        hf.create_dataset('planet_test', data=planet_test)
        hf.create_dataset('planet_prediction', data=planet_prediction)
