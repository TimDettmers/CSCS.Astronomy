import numpy as np
from util import *


path = '/home/tim/data/astro/'

X = load_hdf5_matrix(path + 'X.hdf5')
y = load_hdf5_matrix(path + 'y.hdf5')

print np.max(X)
print np.min(X)
print np.std(X)