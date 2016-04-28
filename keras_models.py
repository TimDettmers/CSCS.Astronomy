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


def max_norm_whole_dataset(X):    
    X -=np.min(X,0)
    X/= np.max(X,0)
    
    print(np.min(X,0))
    print(np.max(X,0))
    return X

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

print('loading data...')
path = '/home/tim/data/astro/'
#X = load_hdf5_matrix(path + 'X_processed.hdf5')
#y = load_hdf5_matrix(path + 'y_processed.hdf5')
out = load_hdf5_matrix(path + 'out.hdf5')
noise = load_hdf5_matrix(path + 'noise.hdf5')
X = np.vstack([out, noise])
print(X.shape)
X = X.reshape(X.shape[0],1,100,100)

y1 = np.ones((out.shape[0],1))
y2 = np.zeros((noise.shape[0],1))
y = np.vstack([y1,y2])


idx_X = np.arange(X.shape[0])
rdm = np.random.RandomState(234)

rdm.shuffle(idx_X)
X = X[idx_X]
y = y[idx_X]
print(y)

idx = load_hdf5_matrix(path + 'idx.hdf5')
print('data loaded!')


def get_rotations(x,y):
    return [np.vstack([x,np.fliplr(x), np.flipud(x), np.flipud(np.fliplr(x))]), np.vstack([y,y,y,y])]

def get_rdm_crops(A, Y, shape, offset = 3):
    crops = np.arange(shape[0],A.shape[2],offset)
    
    patchidx = list(itertools.permutations(crops,2))
    idxend = np.array(patchidx)
    idxstart = np.array(patchidx)-shape[0]
    patches = []
    for x,y in zip(idxstart, idxend):
        patches.append(A[:,:,x[0]:y[0],x[1]:y[1]])
        
    return [np.vstack(patches), np.tile(Y,(idxstart.shape[0],1))]
        
    
def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    if not nb_classes:
        nb_classes = int(np.max(y)+1)
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[np.int32(i), np.int32(y[i])] = 1.
    return Y   

print(X.shape)
    
  

n=X.shape[0]
test = 0.2
cv = 0.1
train = 0.7
print('slicing...')

testn = int(test*n)
cvn = int(cv*n)
trainn = int(train*n)

X_train = np.copy(X[:trainn])
y_train = np.copy(y[:trainn])

X_cv = np.copy(X[trainn:trainn+cvn])
y_cv = np.copy(y[trainn:trainn+cvn])

X_test = np.copy(X[-testn:])
y_test = np.copy(y[-testn:])
del X
del y
gc.collect()

'''
print('cropping...')
X_train, y_train = get_rdm_crops(X_train, y_train, (90,90), offset=5)
X_cv, y_cv = get_rdm_crops(X_cv, y_cv, (90,90), offset=5)
gc.collect()
#X_train, y_train = get_rotations(X_train, y_train)
#X_cv, y_cv = get_rotations(X_cv, y_cv)
print('normalizing...')
gc.collect()

print('categorizing...')
# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_cv = to_categorical(y_cv, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
gc.collect()

X_train = X_train[idx_train]
Y_train = Y_train[idx_train]
'''

print('normalizing...')
X_train = max_norm_whole_dataset(X_train)
X_cv = max_norm_whole_dataset(X_cv)
X_test = max_norm_whole_dataset(X_test)
gc.collect()



idx_train = np.arange(X_train.shape[0])



print('categorizing...')
# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_cv = to_categorical(y_cv, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
gc.collect()
print(X_train.dtype)

print(X_train.shape)
model = Sequential()

model.add(Dropout(0.2, input_shape=(1, img_rows, img_cols)))
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

model.add(Flatten())
model.add(Dense(1024))
#model.add(BatchNormalization(mode=1))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1024))
#model.add(BatchNormalization(mode=1))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

learning_rate = 0.0001
print(learning_rate)
print("with batch normalization")
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#opt = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_cv, Y_cv))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
pred = model.predict(X_test, verbose=1)


errors = np.equal(y_test, np.argmax(pred,axis=1).reshape(-1,1))==0
idx_test = idx[-testn:]
save_hdf5_matrix(path + "error_idx.hdf5", errors)
#np.equal(y_test,)



print('Test score:', score[0])
print('Test accuracy:', score[1])
