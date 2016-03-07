import numpy as np
import cPickle as pickle
from util import *

path = '/home/tim/data/astro/'

X1 = load_hdf5_matrix(path + 'out.hdf5')
X2 = load_hdf5_matrix(path + 'control_images.hdf5')

'''
for i in range(X2.shape[0]):
    img = X2[i]
    img -= np.min(img)
    if np.max(img) > 0.0:
        img = img/np.max(img)
    X2[i] = img
'''

X1 = X1 - np.min(X1)
X1 = X1 / np.max(X1,0)

X2 = X2 - np.min(X2)
X2 = X2 / np.max(X2,0)

y1 = np.ones((X1.shape[0],1))
y2 = np.zeros((X2.shape[0],1))

X = np.vstack([X1,X2])
print X[0].sum()
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
print X[0].sum()

y = np.vstack([y1,y2])

idx = np.arange(X.shape[0])
np.random.shuffle(idx)

X = X[idx]
y = y[idx]


X = X - np.min(X)
X = X / np.max(X,0)

print type(X)
print type(y)

save_hdf5_matrix(path + 'X.hdf5', np.float32(X))
save_hdf5_matrix(path + 'y.hdf5', np.float32(y))

