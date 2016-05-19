import numpy as np
import cPickle as pickle
from util import *
import gc
from os.path import join
import os
import pyfits
from skimage import exposure
import time

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening


def std_norm_whole_dataset(X):
    print np.mean(X,0).shape
    X -=np.mean(X,0)
    X/= np.std(X,0)
    
    print np.mean(X,0)
    print np.std(X,0)
    return X

def max_norm_whole_dataset(X):
    print np.mean(X,0).shape
    X -=np.min(X,0)
    X/= np.max(X,0)
    
    print np.min(X,0)
    print np.max(X,0)
    return X

def max_norm_per_image_dataset(X):
    for i in range(X.shape[0]):
        img = X[i]
        img -= np.min(img)
        if np.max(img) > 0.0:
            img = img/np.max(img)
        X[i] = img
    return X

def std_norm_per_image_dataset(X):
    for i in range(X.shape[0]):
        img = X[i]
        img -= np.mean(img)
        img/= np.std(img)
        #print np.mean(img)
        #print np.std(img)
        X[i] = img
    return X


'''
X1 = X1 - np.min(X1)
X1 = X1 / np.max(X1,0)

X2 = X2 - np.min(X2)
X2 = X2 / np.max(X2,0)
'''

#X = X - np.min(X)
#X = X / np.max(X,0)


mypath = '/users/dettmers/data/'

folder_names = ['out_strong','out/noise']

def get_data(mypath):
	t0 = time.time()
	print mypath
	n = 100000
	data = []
	paths = []
	for i in range(n):

	    if i % 100 == 0:
		if i > 0:
			elapsed = time.time() - t0
			left = n-i
			rate = i/elapsed
			ETA = left/rate
			print "ETA: {0}min".format(int(ETA/60))
	    path = join(mypath, str(i))+'.fits'
	    if not os.path.exists(path): continue
	    img = pyfits.getdata(path,0,memmap=False)           
            img_adapteq = np.abs(img)
            img_adapteq = exposure.equalize_adapthist(np.log(img_adapteq + 1.0), clip_limit=0.5,kernel_size=(4,4))


	    paths.append(path)
	    data.append(img_adapteq)
	return [np.array(data,dtype=np.float32), np.array(paths)]




X1, P1 = get_data(join(mypath, folder_names[0]))
X2, P2 = get_data(join(mypath, folder_names[1]))

gc.collect()

P1 = P1.reshape(-1,1)
P2 = P2.reshape(-1,1)

print X1.shape
print P1.shape
print X2.shape
print P2.shape

X2 = X2[:X1.shape[0]]
P2 = P2[:X1.shape[0]]

X1[np.isnan(X1)] = 0.0
X2[np.isnan(X2)] = 0.0

gc.collect()

print np.sum(np.isnan(X1))
print np.sum(np.isnan(X2))
#X1 = load_hdf5_matrix(path + 'out.hdf5')
#X2 = load_hdf5_matrix(path + 'noise.hdf5')


y1 = np.ones((X1.shape[0],1))
y2 = np.zeros((X2.shape[0],1))

X = np.vstack([X1,np.fliplr(X1), np.flipud(X1), np.flipud(np.fliplr(X1)), X2,np.fliplr(X2), np.flipud(X2), np.flipud(np.fliplr(X2))])
P = np.vstack([P1,P1,P1,P1,P2,P2,P2,P2])
del X1
del X2
gc.collect()
#X = np.vstack([X1,X2])
print X[0].sum()
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
print X[0].sum()

y = np.vstack([y1,y1,y1,y1,y2,y2,y2,y2])
#y = np.vstack([y1,y2])

idx = np.arange(X.shape[0])
rdm = np.random.RandomState(1234)
rdm.shuffle(idx)

gc.collect()
X = X[idx]
P = P[idx]
y = y[idx]

print P[0:5]

gc.collect()

#X = std_norm_per_image_dataset(X)

save_hdf5_matrix(mypath + 'X_processed.hdf5', np.float32(X))
save_hdf5_matrix(mypath + 'y_processed.hdf5', np.float32(y))
save_hdf5_matrix(mypath + 'idx.hdf5', idx)
pickle.dump(P, open(join(mypath, 'P.p'), 'wb'))

