import numpy as np
from util import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import gc

path = '/home/tim/data/astro/'
#X = load_hdf5_matrix(path + 'X_processed.hdf5')[0:100000]
#y = load_hdf5_matrix(path + 'y_processed.hdf5').ravel()[0:100000]

X = load_hdf5_matrix(path + 'hog.hdf5')
y = np.zeros((X.shape[0],))
y[X.shape[0]/2:] = 1.0

rdm = np.random.RandomState(234)

idx = np.arange(X.shape[0])

rdm.shuffle(idx)

X = X[idx]
y = y[idx]

X = X[:10000]
y = y[:10000]

print y.shape
print X.shape

cv = int(0.2*X.shape[0])

y_test = y[-cv:]
y_train = y[:-cv]

X_test = X[-cv:]
X_train = X[:-cv]

del X
del y
gc.collect()

print 'rbf'
clf = SVC(C=0.5, kernel='rbf',cache_size=2048)

print 'fitting...'
clf.fit(X_train, y_train)

print clf.score(X_train, y_train)
print clf.score(X_test, y_test)


