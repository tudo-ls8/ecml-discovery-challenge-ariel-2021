"""Train the model and predict the data"""
import argparse
import os
import warnings

import h5py
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import optimizers, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Concatenate
from keras.models import Model
from scipy import stats, spatial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.utils import resample
import tensorflow
from tslearn.piecewise import PiecewiseAggregateApproximation

DATA_PATH_DEFAULT = '/mnt/data/data_set_preprocessed.h5_small_named_params'
DATA_PATH_COMPLETE = '/mnt/data/data_set_preprocessed.h5_complete_named_params'

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smse(y_true, y_pred):
    return (y_true - y_pred) ** 2 / K.mean(y_true, axis=0)

# compute a combined prediction for each planet - if metric is given (see scipy.spatial.distance.cdist)
# then not the combined prediction is taken directly, but the actual prediction that is closest to it.
def planetwise_prediction(predictions, planets, combination=np.median, metric=None):
    print('Selecting predictions with {} combination and {} metric'.format(combination, metric))
    predictions = predictions.copy()
    for planet_id in np.unique(planets):
        mask = (planets == planet_id).flatten()
        comb = combination(predictions[mask], axis=0, keepdims=True)  # the 'median' prediction
        if metric is not None:
            distances = spatial.distance.cdist(predictions[mask], comb, metric=metric).flatten()
            prediction_id = np.argmin(distances)
            predictions[mask] = predictions[mask][prediction_id]
        else:
            predictions[mask] = np.squeeze(comb)
    return predictions

def eval_pred(y_true, y_pred):
    print('MSE : {:10f}'.format(mean_squared_error(y_true, y_pred)))
    print('MAE : {:10f}'.format(mean_absolute_error(y_true, y_pred)))
    print('MAPE: {:10f}'.format(mean_absolute_percentage_error(y_true, y_pred)))
    print('EVS : {:10f}'.format(explained_variance_score(y_true, y_pred)))
    print('R2  : {:10f}'.format(r2_score(y_true, y_pred)))

# Transformer class with scaled SAX and an optional second round of SAX evaluation
class ScaledPAA:
    def __init__(self, n_segments, shift=0.):
        self.n_segments = n_segments
        self.shift = shift
        if shift > 0.:
            self.n_segments -= 1  # omit one segment to maintain the intended segment size
        self.paa_start = 0.  # default

    def fit(self, x):
        if self.shift > 0.:
            paa_width = x.shape[1] / (self.n_segments + 1)
            self.paa_start = int(paa_width * self.shift)
            self.paa_end = self.paa_start + self.n_segments * int(paa_width)
            x = x[:, self.paa_start:self.paa_end, :]
        print('Fitting a PAA with {} segments and shift {} = {}*segment_width'.format(self.n_segments, self.paa_start,
                                                                                      self.shift))
        self.paa = PiecewiseAggregateApproximation(n_segments=self.n_segments)
        self.paa.fit(x)

    def transform(self, x):
        if self.shift > 0.:
            x = x[:, self.paa_start:self.paa_end, :]
        x_paa = self.paa.transform(x)
        error = np.linalg.norm(self.paa.inverse_transform(x_paa) - x, axis=1, keepdims=True)
        print('PAA: x_paa {}, x {}, error={}'.format(x_paa.shape, x.shape, error.shape))
        x_paa = np.concatenate((x_paa, error), axis=1) # attach the reconstruction error
        return x_paa

# Pipeline-style wrapper around a SAX transformer and a fitted keras model
class PAANet:
    def __init__(self, paa, net):
        self.paa = paa
        self.net = net
    def __transform(self, x):
        x, ms, aux = x  # unpack the input tuple
        return [np.concatenate((self.paa.transform(x), ms), axis=1), aux]
    def fit(self, x, y, **kwargs):
        return self.net.fit(self.__transform(x), y, **kwargs)
    def predict(self, x, **kwargs):
        return self.net.predict(self.__transform(x), **kwargs)
    def evaluate(self, x, y):
        return self.net.evaluate(self.__transform(x), y)

def build_paa_model(x_train, ms_train, aux_train, y_train, paa_segments, epochs, batch_size, paa_shift=None, bootstrap=True):
    if bootstrap:
        x_train, ms_train, aux_train, y_train = resample(x_train, ms_train, aux_train, y_train)

    # Piecewise Aggregate Approximation
    paa = ScaledPAA(paa_segments, shift=paa_shift)
    paa.fit(x_train)
    x_train = np.concatenate((paa.transform(x_train), ms_train), axis=1)  # transform only the training data, for now

    # auxiliary model
    aux_input = Input(shape=aux_train.shape[1:])
    aux = aux_input

    # series model
    series_input = Input(shape=x_train.shape[1:])
    series_dense = Dense(256, activation='relu')(series_input)
    series_dense = Dense(256, activation='relu')(series_dense)
    series_dense = Dense(256, activation='relu')(series_dense)
    flatten = Flatten()(series_dense)

    # combination/head model
    merged_model = Concatenate()([flatten, aux])
    merged_model = Dense(128, activation='relu')(merged_model)
    merged_model = Dense(128, activation='relu')(merged_model)
    merged_model = Dense(128, activation='relu')(merged_model)
    merged_model = Dense(y_train.shape[1], activation='linear')(merged_model)
    base_model = Model(inputs=[series_input, aux_input], outputs=merged_model)

    # compile and train
    adam = optimizers.Adam(lr=1e-3, decay=0.001)
    base_model.compile(optimizer=adam,
                       loss='mae',
                       metrics=['mse', smse])
    base_model.summary()

    callbacks = [EarlyStopping(patience=10, min_delta=0.0001, restore_best_weights=True)]
    base_model.fit([x_train, aux_train], y_train, epochs=epochs, verbose=2,
                   batch_size=batch_size, validation_split=0.1, callbacks=callbacks)
    return PAANet(paa, base_model)  # wrap with pre-processing

def trainpredict(predict, seed=876, batch_size=128, epochs=100, paa_segments=20, complete_data=True, train_with_test=True, new_test_split=None):
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    K.set_floatx('float32')

    # read all data, scaled with zero mean and unit variance (time series) or to the [0,1] interval (auxiliary data)
    data_path = DATA_PATH_COMPLETE if complete_data else DATA_PATH_DEFAULT
    with h5py.File(data_path, 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]

        x_test = hf['x_test'][:]
        y_test = hf['y_test'][:]

        x_prediction = hf['x_prediction'][:]

        aux_train = hf['aux_train'][:]
        aux_test = hf['aux_test'][:]
        aux_prediction = hf['aux_prediction'][:]

        ms_train = hf['ms_train'][:]
        ms_test = hf['ms_test'][:]
        ms_prediction = hf['ms_prediction'][:]

        planet_train = hf['planet_train'][:]
        planet_test = hf['planet_test'][:]
        planet_prediction = hf['planet_prediction'][:]

    print('Read x_train {}, x_test {}, with {} aux features from {}'.format(
        x_train.shape,
        x_test.shape,
        aux_train.shape[1],
        data_path
    ))

    print('Number of planets: x_train {}, x_test {}, x_prediction {}'.format(
        len(np.unique(planet_train)),
        len(np.unique(planet_test)),
        len(np.unique(planet_prediction)),
    ))

    print('Minimum number of examples per planet: x_train {}, x_test {}, x_prediction {}'.format(
        np.min(np.unique(planet_train, return_counts=True)[1]),
        np.min(np.unique(planet_test, return_counts=True)[1]),
        np.min(np.unique(planet_prediction, return_counts=True)[1]),
    ))

    if train_with_test:
        print('Adding all test instances to the training set')
        x_train = np.concatenate((x_train, x_test), axis=0)
        ms_train = np.concatenate((ms_train, ms_test), axis=0)
        aux_train = np.concatenate((aux_train, aux_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        planet_train = np.concatenate((planet_train, planet_test), axis=0)

    if new_test_split is not None:
        print('Setting up a new test split with factor {}'.format(new_test_split))
        all_planets = np.unique(planet_train)
        np.random.shuffle(all_planets)
        new_test_split_index = int(round(new_test_split * len(all_planets)))
        mask_train = np.zeros(y_train.shape[0], dtype='bool')
        mask_test = np.zeros(y_train.shape[0], dtype='bool')
        for i, planet_id in enumerate(all_planets):
            mask = (planet_train == planet_id).flatten()
            if i < new_test_split_index:
                mask_test = mask_test | mask
            else:
                mask_train = mask_train | mask
        x_test = x_train[mask_test, :, :]
        ms_test = ms_train[mask_test, :, :]
        aux_test = aux_train[mask_test, :]
        y_test = y_train[mask_test, :]
        planet_test = planet_train[mask_test]
        x_train = x_train[mask_train, :, :]
        ms_train = ms_train[mask_train, :, :]
        aux_train = aux_train[mask_train, :]
        y_train = y_train[mask_train, :]
        planet_train = planet_train[mask_train]
        print('The new test split consists in x_train {} ({} planets) and x_test {} ({} planets)'.format(
            x_train.shape,
            len(np.unique(planet_train)),
            x_test.shape,
            len(np.unique(planet_test))
        ))
        print('Test planets with seed {}: {}'.format(seed, all_planets[:new_test_split_index]))

    # build and train a set of models
    paa_width = int(x_train.shape[1] / paa_segments)
    n_shift_tiles = 3  # 20 PAA segments result in a width of 15 and a maximum of 45 models
    paa_shifts = np.tile(np.arange(paa_width), n_shift_tiles) / paa_width
    print('Training up to {} models, with {} shift tiles resulting in the following shift values:\n{}\n{}'.format(
        paa_shifts.shape[0],
        n_shift_tiles,
        paa_shifts,
        (paa_shifts * paa_width).astype(int)
    ))

    preds_test = []  # ensemble member predictions for the test set
    preds_pred = []  # ensemble member predictions for the upload/prediction set
    for model_index, paa_shift in enumerate(paa_shifts):
        current_model = build_paa_model(x_train, ms_train, aux_train, y_train, paa_segments, epochs, batch_size, paa_shift=paa_shift)

        # evaluation
        current_preds_test = planetwise_prediction(current_model.predict([x_test, ms_test, aux_test]), planet_test)
        preds_test.append(current_preds_test)  # individual planet-wise predictions for the test data
        print('\nEnsemble predictions of planet-wise members (ensemble median, {} members):'.format(model_index + 1))
        eval_pred(y_test, np.median(preds_test, axis=0))

        # prediction
        if predict:
            if not os.path.exists(predict):
                os.makedirs(predict)

            print('Predicting the upload data')
            current_preds_pred = planetwise_prediction(
                current_model.predict([x_prediction, ms_prediction, aux_prediction]), planet_prediction)
            preds_pred.append(current_preds_pred)  # individual planet-wise predictions for the upload/prediction data
            output_path = predict + "_{:02d}".format(model_index + 1)

            print('Writing the median prediction of {} ensemble members to {}'.format(model_index + 1, output_path))
            np.savetxt(output_path, np.median(preds_pred, axis=0), delimiter='\t', fmt='%.14f')
    print('\nDONE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model and predict the data', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--predict', default=None, help='optional path to store predictions at')
    parser.add_argument('--seed', type=int, default=876, help='random seed for numpy and tensorflow')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--paa-segments', type=int, default=20, help='number of PAA segments')
    parser.add_argument('--complete-data', help='whether to use the complete data set', action='store_true')
    parser.add_argument('--train-with-test', help='whether to include our own test data during training', action='store_true')
    parser.add_argument('--new-test-split', type=float, default=0.0, help='make a new train-test split if > 0')
    args = parser.parse_args()
    new_test_split = args.new_test_split if args.new_test_split > 0 else None
    if (args.predict and not args.complete_data) or (args.predict and not args.train_with_test):
        print('WARNING: Not using the complete data while predicting. Specify --complete-data and --train-with-test to train with all data.')
    if args.new_test_split is not None and not args.train_with_test:
        print('WARNING: Using a --new-test-split without --train-with-test will reduce the amount of training data.')
    trainpredict(args.predict, args.seed, args.batch_size, args.epochs, args.paa_segments, args.complete_data, args.train_with_test, new_test_split)
