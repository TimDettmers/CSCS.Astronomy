import numpy as np
from util import *


path = '/home/tim/data/astro/'

X1 = load_hdf5_matrix(path + 'out_processed.hdf5')
X2 = load_hdf5_matrix(path + 'noise_processed.hdf5')

X2 = X2.reshape(X1.shape[0],X1.shape[1]*X1.shape[2])
X1 = X1.reshape(X1.shape[0],X1.shape[1]*X1.shape[2])

np.max(X1)

print np.mean(X1)
print np.mean(X2)
print np.std(X2,1)
print np.mean(np.mean(X1,1).shape)
print np.mean(np.mean(X2,1).shape)

#X = load_hdf5_matrix(path + 'X.hdf5')
#y = load_hdf5_matrix(path + 'y.hdf5')

#print np.max(X)
#print np.min(X)
#print np.std(X)