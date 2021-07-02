import numpy as np
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation

def store_csv(fname, Y):
	"""Prepare a CSV file to be used in pgfplots"""
	Y = np.squeeze(Y) # flatten the tslearn-shaped input
	X = np.arange(len(Y)) + 50 # x coordinates of the plot
	np.savetxt(fname, np.stack((X, Y)).T)
	print('Wrote {} values to {}'.format(len(Y), fname))

# 1) raw data - TODO load some real data
np.random.seed(876)
Y = random_walks(n_ts=1, sz=105, d=1) # random time series
Y = .9 - (np.flip(Y) + Y.min()) / (Y.max() - Y.min()) * 0.05
store_csv('z_paa_1raw.csv', Y)

# 2) z normalization
Y = TimeSeriesScalerMeanVariance().fit_transform(Y)
store_csv('z_paa_2z.csv', Y)

# 3) PAA
paa = PiecewiseAggregateApproximation(n_segments=int(Y.size/15))
Y = paa.inverse_transform(paa.fit_transform(Y))
store_csv('z_paa_3paa.csv', Y)
