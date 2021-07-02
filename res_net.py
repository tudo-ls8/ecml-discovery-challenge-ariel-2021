import argparse
import os
import warnings

import h5py
import keras
import keras.backend as K
import numpy as np
# from tensorflow.random import set_random_seed
import tensorflow
from keras import optimizers, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Concatenate
from keras.models import Model
from numpy.random import seed
from scipy import stats, spatial
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.utils import resample
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


def make_fcn_net(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=40, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=20, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(55, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae', optimizer=keras.optimizers.Adam(),
                  metrics=['mse', smse])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)

    # file_path = self.output_directory + 'best_model.hdf5'
    #
    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
    #                                                    save_best_only=True)

    # self.callbacks = [reduce_lr, model_checkpoint]

    return model, [reduce_lr]


def make_res_net(input_shape):
    n_feature_maps = 64

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=16, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=16, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=16, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(55, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae', optimizer=keras.optimizers.Adam(),
                  metrics=['mse', smse])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    return model, [reduce_lr]


def _inception_module(input_tensor, stride=1, n_filter=32, kernel_size=41, bottleneck_size=32, activation='linear'):
    if int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=n_filter, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=n_filter, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x


def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def make_inception_net(input_shape):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(6):

        x = _inception_module(x)

        if d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    # gap_layer = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    output_layer = keras.layers.Dense(55, activation='linear')(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae', optimizer=keras.optimizers.Adam(),
                  metrics=['mse'])
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)

    return model, [reduce_lr]


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
    #
    # # build and train a set of models
    # paa_width = int(x_train.shape[1] / args.paa_segments)
    # n_shift_tiles = 3  # 20 PAA segments result in a width of 15 and a maximum of 45 models
    # paa_shifts = np.tile(np.arange(paa_width), n_shift_tiles) / paa_width
    # print('Training up to {} models, with {} shift tiles resulting in the following shift values:\n{}\n{}'.format(
    #     paa_shifts.shape[0],
    #     n_shift_tiles,
    #     paa_shifts,
    #     (paa_shifts * paa_width).astype(int)
    # ))

    # Piecewise
    # Aggregate
    # Approximation
    # paa = ScaledPAA(args.paa_segments, shift=paa_shift)
    # paa.fit(x_train)  # paa.fit does only look at the data shape if scale_after=False
    # x_train = np.concatenate((paa.transform(x_train), ms_train), axis=1)  # transform only the training data, for now

    # auxiliary model
    # aux_input = Input(shape=aux_train.shape[1:])
    # aux = aux_input

    # We could use an ensemble of 10 resnets with different initializations.
    # model, cbs = make_res_net(x_train.shape[1:])
    # x_train = np.concatenate((x_train, ms_train), axis=1)

    input_layer = keras.layers.Input(x_train.shape[1:])

    ms_input = keras.layers.Input(ms_train.shape[1:])
    aux_input = keras.layers.Input(aux_train.shape[1:])

    conv1 = keras.layers.Conv1D(filters=32, kernel_size=10, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=40, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=32, kernel_size=20, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    concat_layer = keras.layers.Concatenate([gap_layer, ms_input, aux_input])

    output_layer = keras.layers.Dense(55, activation='linear')(concat_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae',
                  optimizer=keras.optimizers.Adam(lr=1e-3, decay=0.001),
                  metrics=['mse', smse])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=0.0001)
    cbs = [reduce_lr, EarlyStopping(patience=10, min_delta=0.0001, restore_best_weights=True)]
    # model, cbs = make_fcn_net(x_train.shape[1:])
    model.summary()

    # series model
    # series_input = Input(shape=x_train.shape[1:])
    # series_dense = Dense(256, activation='relu')(series_input)
    # series_dense = Dense(256, activation='relu')(series_dense)
    # series_dense = Dense(256, activation='relu')(series_dense)
    # flatten = Flatten()(series_dense)
    #
    # # This would be replaced by
    #
    # # combination/head model
    # merged_model = Concatenate()([flatten, aux])
    # merged_model = Dense(128, activation='relu')(merged_model)
    # merged_model = Dense(128, activation='relu')(merged_model)
    # merged_model = Dense(128, activation='relu')(merged_model)
    # merged_model = Dense(y_train.shape[1], activation='linear')(merged_model)
    # base_model = Model(inputs=[series_input, aux_input], outputs=merged_model)
    #
    # # compile and train
    # adam = optimizers.Adam(lr=1e-3, decay=0.001)
    # base_model.compile(optimizer=adam,
    #                    loss='mae',
    #                    metrics=['mse', smse])
    # base_model.summary()
    #
    # callbacks = [EarlyStopping(patience=10, min_delta=0.0001, restore_best_weights=True)]
    model.fit([x_train, ms_train, aux_train], y_train, epochs=args.epochs, verbose=1,
              batch_size=args.batch_size, validation_split=0.1, callbacks=cbs)

    # preds_test = []  # ensemble member predictions for the test set
    # preds_pred = []  # ensemble member predictions for the upload/prediction set
    # for model_index, paa_shift in enumerate(paa_shifts):
    #     current_model = build_paa_model(x_train, ms_train, aux_train, y_train, args, paa_shift=paa_shift)
    #
    #     # evaluation
    #     current_preds_test = planetwise_prediction(current_model.predict([x_test, ms_test, aux_test]), planet_test)
    #     preds_test.append(current_preds_test)  # individual planet-wise predictions for the test data
    #     print('\nEnsemble predictions of planet-wise members (ensemble median, {} members):'.format(model_index + 1))
    #     eval_pred(y_test, np.median(preds_test, axis=0))
    #
    #     # prediction
    #     if args.predict:
    #         print('Predicting the upload data')
    #         current_preds_pred = planetwise_prediction(
    #             current_model.predict([x_prediction, ms_prediction, aux_prediction]), planet_prediction)
    #         preds_pred.append(current_preds_pred)  # individual planet-wise predictions for the upload/prediction data
    #         output_path = args.predict + "_{:02d}".format(model_index + 1)
    #
    #         print('Writing the median prediction of {} ensemble members to {}'.format(model_index + 1, output_path))
    #         np.savetxt(output_path, np.median(preds_pred, axis=0), delimiter='\t', fmt='%.14f')
    #
    # print('\nDONE')


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
