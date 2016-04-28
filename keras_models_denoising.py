from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gc
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from util import *
from keras.layers.normalization import BatchNormalization
import itertools
import numpy as np


def max_norm_whole_dataset(X):    
    X -=np.min(X,0)
    X/= np.max(X,0)
    
    print(np.min(X,0))
    print(np.max(X,0))
    return X

batch_size = 128
nb_classes = 10000
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 100, 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets

print('loading data...')
path = '/home/tim/data/astro/'
X = load_hdf5_matrix(path + 'out.hdf5')
print(X.shape)
X = X.reshape(X.shape[0],1,100,100)
y = np.float32(load_hdf5_matrix(path + 'clean.hdf5') > 1e-10)
weight = np.copy(y)
weight+= 0.01
print(y)
print(np.max(y))
print(np.max(X))
print('data loaded!')


def get_rotations(x,y):
    return [np.vstack([x,np.fliplr(x), np.flipud(x), np.flipud(np.fliplr(x))]), np.vstack([y,np.fliplr(y), np.flipud(y), np.flipud(np.fliplr(y))])]


n=X.shape[0]
test = 0.1
cv = 0.2
train = 0.7
print('slicing...')

testn = int(test*n)
cvn = int(cv*n)
trainn = int(train*n)

idx = np.arange(X.shape[0])
np.random.shuffle(idx)
weight = weight[idx] 
X = X[idx]
y = y[idx]

X = max_norm_whole_dataset(X)
y = max_norm_whole_dataset(y)
y = y.reshape(y.shape[0],100*100)
weight = weight.reshape(weight.shape[0],100*100)

X_train = np.copy(X[:trainn])
y_train = np.copy(y[:trainn])
w_train = np.copy(weight[:trainn])

X_cv = np.copy(X[trainn:trainn+cvn])
y_cv = np.copy(y[trainn:trainn+cvn])

X_test = np.copy(X[-testn:])
y_test = np.copy(y[-testn:])



print(X_train.shape)
model = Sequential()
'''
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
#model.add(BatchNormalization(axis=1))

model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
'''
model.add(Flatten(input_shape=(1, img_rows, img_cols)))
model.add(Dense(1024))
#model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(Dense(1024))
#model.add(BatchNormalization(mode=1))
model.add(Activation('relu'))
model.add(Dense(nb_classes))

learning_rate = 0.01
print(learning_rate)
print("with batch normalization")
#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt = RMSprop(lr=learning_rate)
model.compile(loss='mean_absolute_error', optimizer=opt)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_cv, y_cv))
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
pred = model.predict(X_test, verbose=1)


errors = np.equal(y_test, np.argmax(pred,axis=1).reshape(-1,1))==0
save_hdf5_matrix(path + "error_idx.hdf5", errors)
#np.equal(y_test,)


pred = model.predict(X, verbose=1).reshape(X.shape[0],100,100)
save_hdf5_matrix(path + "denoised.hdf5",pred)
print('Test score:', score[0])
print('Test accuracy:', score[1])
