# ARIEL ML data challenge

The goal is to predict 55 regression targets from multivariate time series consisting of 55 channels and 300 time steps.
What the target variables model is the relative size of a planet transiting a star at different wavelengths.
Each time series is the (noisy) relative flux of light that is recorded at the respective wavelength.

## What we did

- We sticked to using deep dense neural networks. This approach was already suggested by the baseline solution and
  for us, it performed best in the end. Intermediate solutions were good with LSTMs, but after proper pre-processing
  the dense nets worked best. Moreover, we could not make classical ML methods work with the same convenience and power.
- Of course, we tuned the meta-parameters (depth and width of the net). In our case, there was a wide valley of many
  near-optimal parameter combinations, so at some point we just stuck with one of them and tuned other aspects.
- We use the 6 additional features that are available for each observation, feeding them as additional input into
  the network at a layer quite in the middle.
- We use Z-Scaling (zero mean, unit standard deviation) with a subsequent Piecewise Aggregate Approximation (PAA,
  only keeping the mean values of each time series in multiple equi-width segments) as pre-processing. This step
  actually made the largest difference observed by us.
- We aggregate the predictions for each planet. Each planet is identified by a running number. The target values are
  constant among all observations of a planet. Each of the 100 observations of a planet has another noise instance.
- We wrap our PAA-neural nets in a bagging ensemble. Each member of this ensemble has another shift in the underlying
  time series which results in one distinct PAA representation per member, thus an increased variability among members.
- There are two additional target variables which could have been used for training.
  In a few attempts to do so, we did not find them helpful, so we abandoned this opportunity.
- We did not leverage the final evaluation data, even though the un-labeled observations could have been used, e.g. in an
  EM or semi-supervised setting.


# How to run our experiments?

## Docker setup

The code is intended to run from a container, so you won't have to deal with setup of paths and frameworks. Initially you have to prepare a directory, which contains the data provided by the workshop committee as well as this git repository. The next step is to build the docker image. Please adjust group- and user IDs in the `Dockerfile` to the user of your PC. After these adjustments, build the container by:

```bash
cd ./docker
make image
```

Assuming you created a directory `/home/user/src/ariel/` with the data and the git repo, start the container with this directory mounted to `/mnt/data`, e.g.

```bash
docker run -it -v /home/user/src/ariel/:/data IMAGE-NAME /bin/bash
```

You can check `docker/run.sh` for many more details on how we call `docker run` at our research group.

## Data massage

The first ARIEL discovery challenge in 2019 supplied `.txt` listings of all data files in the directories `noisy_train/`, `noisy_test/`, and `params_train/`. For the second challenge in 2021, we need to generate these files ourselves, by finding all files in the directories:

```bash
find noisy_train -name "*.txt" | sort > noisy_train.txt
find noisy_test -name "*.txt" | sort > noisy_test.txt
find params_train -name "*.txt" | sort > params_train.txt
```

### tl;dnr

Generate the preprocessed HDF5 files from which we learn. These steps are detailed below.

```bash
python3 ecml-discovery-challenge/bin/complete_dataset_to_hdf5.py --seed 1111 --test-percentage 10 params_train.txt
python3 ecml-discovery-challenge/bin/noisy_test_to_hdf5.py
python3 ecml-discovery-challenge/bin/preprocess_to_file.py --complete-data /mnt/data/data_set_preprocessed.h5_complete_named_params
```

### Details

As first step you have to convert the given `.txt` files into a larger HDF5 file. This allows much faster reads of the training data. For the following commands we assume, that the current working directoy is `/mnt/data` and the directories `noisy_test`, `noisy_train` and `params_train` as well as their corresponding `.txt` files are located there. 

To generate the HDF5-file `train_test_set.h5_complete_name_params` run:

```bash
python3 ecml-discovery-challenge/bin/complete_dataset_to_hdf5.py --seed 1111 --test-percentage 10 params_train.txt
```

The parameter `--test-percentage` will determine the percentage of planets, which will be used for testing the model and can be controlled via the random-seed. The script will read all files from the params_train and noisy_train directories and store their data to an HDF5 file. Timeseries can be accessed via the keys: `x_train, y_train` and `x_test, y_test`. Additional parameters are stored prefixed with either `x_test` or `x_train` and suffixed with `_param_name`. Following parameter names are available:

- `_star_temp`
- `_star_logg`
- `_star_rad`
- `_star_mass`
- `_star_k_mag`
- `_star_period`

- `_star_planet_idx`
- `_star_sun_spot_idx`
- `_star_photon_noise_idx`

The additional test parameters are:

- `_incl`
- `_sma`

Afterwards you have to create an HDF5 from the targets via the following command:

```bash
python3 ecml-discovery-challenge/bin/noisy_test_to_hdf5.py
```

which results in the following file `/mnt/data/noisy_test.h5_named_params`. Within this file the keys are prefixed with `x_prediction` and the suffixes are the same as above.

Those two files are further precprocessed by the script `preprocess_to_file.py`, which computes some statistics for the values, applies normalization and merges them.

```bash
python3 ecml-discovery-challenge/bin/preprocess_to_file.py --complete-data /mnt/data/data_set_preprocessed.h5_complete_named_params
```

The resulting file `/mnt/data/data_set_preprocessed.h5_complete_named_params` contains the scaled test, train and prediction data alongside the preprocessed auxiliary features.


## Model training and data prediction

Training and predictions are taken out in one step. The `main.py` script fits up to 45 ensemble members. Each member is forgotten as soon as it has contributed its predictions. The path of the `--predict` argument is appended with the ensemble size so that predictions of smaller ensembles can be uploaded before the script has finished.

We run model training and prediciton upload via the Baselbot. Assuming an instance of headless selenium is running on ```localhost:444```, you can use
```bash
python3 -m baselbot --url 'http://localhost:4444/wd/hub'
```

## Automated uploads through Selenium

```bash
docker run -it --name selenium -d -p 4444:4444 -v /dev/shm:/dev/shm selenium/standalone-firefox
venv/bin/python submit_data.py --url http://localhost:4444/wd/hub --username TheReturnOfBasel321 --password XXXX -i pred.csv_05_20
```


# Further documentation

The following information was collected from the website mentioned above.

### Problem

This data challenge is trying to identify and correct the effect of stellar spots -literally spots on the surface
of the star- in noisy transiting lightcurves of extrasolar planets. Exoplanets are planets orbiting other stars,
like our own solar system planets orbit our sun. When analysing these distant worlds, the effects of stellar variability is one of the major data analysis challenges in the field and directly impacts our measurements. Without correcting for brightness variabilities and ‘star spots’, we are not able to measure the radius of the planet correctly and, perhaps more importantly, the chemistry of their atmospheres.

### Task

We are in a multi target regression setting. The goal is to predict 55 output numerical values, given 55 numerical time 
series with 300 observations each. Furthermore, there are 6 additional features.

### Dataset

The dataset consists of 146800 training and 62900 test samples. 
Each sample is stored in a file called AAAA_BB_CC.txt. 
The values AAAA range from 0001 to 2097 and represent the the index of the observed planet,
BB ranges from 01 to 10 and is an index for the stellar spot noise instance and CC ranges from 
01 to 10 is an index for the gaussian photon noise instance observed. 

The dataset consists of two types of files, ‘noisy’ files (containing the features) and ‘parameters’ files (containing the targets of the training examples).

It was downloaded from the challenge web site: https://ariel-datachallenge.azurewebsites.net/ML/documentation/data

#### Noisy files (./noisy_[train|test])

The 'noisy' files contain the features. There are 6 stellar and planet parameters, which are located at the first
6 rows of each file. They are prefixed by '# ' and stored as key-value pairs, e.g. `# star_temp: 6000`. Besides these
parameters, there are 55 timeseries with 300 steps each, which contain the relative fluxes of different wavelength during 
a transit. Each timeseries is stored in one row and the values are separated by `\t`.

```bash
#star_temp: 5196.0
#star_logg: 4.5
#star_rad: 0.94
#star_mass: 0.91
#star_k_mag: 4.015
#period: 0.736539
      (t1)            (t2)            ...   (t300)
(w1)  1.00010151742   1.00010218526   ...   1.00001215251
(w2)  0.999857792623  1.00009976297   ...   1.00007764626
(...) ...             ...             ...   ...
(w55) 0.999523150082  0.999468565171  ...   0.999934661757
```

#### Parameter files (./params_train)

The parameter files contain the targets for our problem. The two additional parameters sma (semimajor axis) and incl 
(inclination) can be used as additional target or be ignored. The main goal is to predict the 55 relative radii
 (planet-to-star-radius ratios), where every column corresponds to a particular wavelength channel.

```bash
#sma: 2314065295
#incl: 83.3
              (w1)            (w2)            (...) (w55)           
(AAAA_BB_CC)  0.0195608058653 0.019439812298  ...   0.0271040897872
```

#### Size

| | noisy_train  | noisy_test  | params_train |
|---|---|---|---|
|  N |146800 | 62900 | 146800 | 
| Size  | 34G | 15G |  574M |

For each of the file types, there is one summary file (noisy_test.txt, noisy_train.txt, params_train.txt),
 which contains a list of all files stored within the directory.

###  Baseline

As a baseline solution, a feedforward neural network was trained on a sample of 5000 training examples selected uniformly at random. The neural network uses 
all 55 noisy light curves to predict the 55 relative raii directly. It does not use the stellar parameters, nor does is predict intermediate targets to do so.

Train / Validation Split:
    - 4020 training and 980 validation examples in a way, that both sets contained no planets in common

#### Selected hyperparameters

| Hyperparameter  | Value  |
|---|---|
|  Loss | Avg. MSE along all wavelengths |
| Optimizer  | ADAM  |
| Learning rate  | 0.0001  |
| Weight decay  | 0.01  |
| Batch size  | 128  |
| Rest:  | Keras default  |

### References

- [Ariel Space Mission](https://arielmission.space/)
- [Workshop](https://ariel-datachallenge.azurewebsites.net/ML/documentation/description)
- [Exoplanets 101](https://exoplanets.nasa.gov/the-search-for-life/exoplanets-101/)

### Ideas

- Remove samples with really noisy targets and train model.
- Get the data with nice curves and train a model.
    - How to detect those curves? (Divide time series in 3 intervalls, compute their mean.) If middle is significant lower, we detect a real transit.
- Have a look at previous publications from challenge hosts.
