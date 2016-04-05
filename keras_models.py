from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from util import *
from keras.layers.normalization import BatchNormalization

batch_size = 128
nb_classes = 2
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 100, 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets

path = '/home/tim/data/astro/'
X = load_hdf5_matrix(path + 'X_processed.hdf5')
X = X.reshape(X.shape[0],1,100,100)
y = load_hdf5_matrix(path + 'y_processed.hdf5')

cv = 80000

y_test = y[-cv:]
y_train = y[:-cv]

X_test = X[-cv:]
X_train = X[:-cv]

print(X_test.shape)
print(X_train.shape)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dropout(0.2, input_shape=(1, img_rows, img_cols)))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt = RMSprop(lr=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])