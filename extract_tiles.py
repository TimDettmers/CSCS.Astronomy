import numpy as np
from util import *
import cPickle as pickle

path = '/home/tim/data/astro/clean.hdf5'
path_out = '/home/tim/data/astro/out.hdf5'

X = load_hdf5_matrix(path)
out = load_hdf5_matrix(path_out)

#X[X == 4.22075929e-26] = 0.0
#save_hdf5_matrix(path, X)
#print np.histogram(X[0], bins=10)

a = X[0][0:4,0:4]
print a
print a[:-1,1:]

tiles = []
for imgno, img in enumerate(X):
    for i in range(1,100):
        if np.sum(img[:-i,i:]) == 0:
            tiles.append(out[imgno][:-i,i:])
            #print out[imgno][:-i,i:].max()
            break
        
    for i in range(1,100):
        if np.sum(img[:-i,:-i]) == 0:
            tiles.append(out[imgno][:-i,:-i])
            #print out[imgno][:-i,:-i].max()
            break      
          
    for i in range(1,100):
        if np.sum(img[i:,:-i]) == 0:
            tiles.append(out[imgno][i:,:-i])
            #print out[imgno][i:,:-i].max()
            break
          
    for i in range(1,100):
        if np.sum(img[i:,i:]) == 0:
            tiles.append(out[imgno][i:,i:])
            #print out[imgno][i:,i:].max()
            break
            
    if imgno % 1000 == 0: print imgno
    
    
    
pickle.dump(tiles,open('/home/tim/data/astro/tiles.p','wb'))