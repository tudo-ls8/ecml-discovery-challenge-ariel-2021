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
from numpy.random import seed
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


class ReportMetricsCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, planet_test=None):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.planet_test = planet_test

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x_test, verbose=1)
        if self.planet_test is not None:
            print('\nPlanet-wise -- median + None')
            eval_pred(self.y_test,
                      planetwise_prediction(predictions, self.planet_test, combination=np.median, metric=None))
        else:
            print('\nIndividual predictions')
            eval_pred(self.y_test, predictions)


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
    def __init__(self, n_segments, shift=0., scale_after=False, second_run=False, attach_error=True):
        self.n_segments = n_segments
        self.shift = shift
        if shift > 0.:
            self.n_segments -= 1  # omit one segment to maintain the intended segment size
        self.scale_after = scale_after
        self.second_run = second_run
        self.attach_error = attach_error
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
        if self.second_run or self.scale_after:
            x_paa = self.paa.transform(x)  # needed for subsequent processing

        # shift time series and compute more PAA windows -> overlapping representation
        if self.second_run:
            print('Fitting a second PAA, using segments that overlap with the previous run')
            paa_width = x.shape[1] / self.n_segments
            self.paa_start = int(paa_width / 2)
            self.paa_end = self.paa_start + (self.n_segments - 1) * int(paa_width)
            x = x[:, self.paa_start:self.paa_end, :]
            print(
                'The overlap is achieved by shifting the input time series by {} to the shape {}'.format(self.paa_start,
                                                                                                         x.shape[1:]))
            self.paa2 = PiecewiseAggregateApproximation(n_segments=self.n_segments - 1)
            x_paa = np.concatenate((x_paa, self.paa2.fit_transform(x)), axis=1)
            print('The final, overlapping SAX shape is {}'.format(x_paa.shape[1:]))

        if self.scale_after:
            print('SAX output: {}'.format(stats.describe(x_paa, axis=None)))
            self.paa_min = np.min(x_paa)  # like in tslearn.TimeSeriesScalerMeanVariance
            self.paa_range = np.max(x_paa) - self.paa_min

    def transform(self, x):
        if self.shift > 0.:
            x = x[:, self.paa_start:self.paa_end, :]
        x_paa = self.paa.transform(x)
        error = np.linalg.norm(self.paa.inverse_transform(x_paa) - x, axis=1, keepdims=True)
        print('PAA: x_paa {}, x {}, error={}'.format(x_paa.shape, x.shape, error.shape))
        if self.second_run:
            x_paa = np.concatenate((x_paa, self.paa2.transform(x[:, self.paa_start:self.paa_end, :])), axis=1)
        if self.scale_after:
            print('Post-scaling the SAX output to the [0, 1] interval')
            x_paa = (x_paa - self.paa_min) / self.paa_range
        if self.attach_error:
            x_paa = np.concatenate((x_paa, error), axis=1)
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


def build_paa_model(x_train, ms_train, aux_train, y_train, args, paa_shift=None, bootstrap=True):
    if bootstrap:
        x_train, ms_train, aux_train, y_train = resample(x_train, ms_train, aux_train, y_train)

    # Piecewise Aggregate Approximation
    paa = ScaledPAA(args.paa_segments, shift=paa_shift)
    paa.fit(x_train)  # paa.fit does only look at the data shape if scale_after=False
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
    base_model.fit([x_train, aux_train], y_train, epochs=args.epochs, verbose=2,
                   batch_size=args.batch_size, validation_split=0.1, callbacks=callbacks)
    return PAANet(paa, base_model)  # wrap with pre-processing


def main(args):
    # read all data, scaled with zero mean and unit variance (time series) or to the [0,1] interval (auxiliary data)
    data_path = DATA_PATH_COMPLETE if args.complete_data else DATA_PATH_DEFAULT
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

    if args.train_with_test:
        x_train = np.concatenate((x_train, x_test), axis=0)
        ms_train = np.concatenate((ms_train, ms_test), axis=0)
        aux_train = np.concatenate((aux_train, aux_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        planet_train = np.concatenate((planet_train, planet_test), axis=0)

    # build and train a set of models
    paa_width = int(x_train.shape[1] / args.paa_segments)
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
        current_model = build_paa_model(x_train, ms_train, aux_train, y_train, args, paa_shift=paa_shift)

        # evaluation
        current_preds_test = planetwise_prediction(current_model.predict([x_test, ms_test, aux_test]), planet_test)
        preds_test.append(current_preds_test)  # individual planet-wise predictions for the test data
        print('\nEnsemble predictions of planet-wise members (ensemble median, {} members):'.format(model_index + 1))
        eval_pred(y_test, np.median(preds_test, axis=0))

        # prediction
        if args.predict:
            if not os.path.exists(args.predict):
                os.makedirs(args.predict)

            print('Predicting the upload data')
            current_preds_pred = planetwise_prediction(
                current_model.predict([x_prediction, ms_prediction, aux_prediction]), planet_prediction)
            preds_pred.append(current_preds_pred)  # individual planet-wise predictions for the upload/prediction data
            output_path = args.predict + "_{:02d}".format(model_index + 1)

            print('Writing the median prediction of {} ensemble members to {}'.format(model_index + 1, output_path))
            np.savetxt(output_path, np.median(preds_pred, axis=0), delimiter='\t', fmt='%.14f')

    print('\nDONE')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP on ariel data.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=876, help='Random seed for numpy and tensorflow.')

    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training the neural network.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--paa-segments', type=int, default=20, help='Number of PAA segments')

    parser.add_argument('--predict', default=None,
                        help='If this path is specified \
                             the trained model will be used to generate predictions for the noisy test files.')
    parser.add_argument('--complete-data', help='Use the complete data set', action='store_true')
    parser.add_argument('--train-with-test', help='Also use our own test data for training', action='store_true')
    ###################################
    args = parser.parse_args()
    if (args.predict and not args.complete_data) or (args.predict and not args.train_with_test):
        print(
            'WARNING: Writing predictions but not using the complete data set. Specify --complete-data and --train-with-test to use all examples for training.')
    ###################################
    seed(args.seed)
    # Set TF seed
    tensorflow.random.set_seed(args.seed)
    ###################################
    # TensorFlow wizardryx_test
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Disable tf logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    # config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 1

    # Create a session with the above options specified.
    # K.tensorflow_backend.set_session(tf.Session(config=config))
    K.set_floatx('float32')
    ###################################
    main(args)
