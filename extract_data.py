import pyfits
from os import listdir
from os.path import isfile, join
import numpy as np
from util import *
import gc
import PIL
import os

mypath = '/users/dettmers/data/out_strong'


n = 100000 

data = []
for i in range(n):
    
    if i % 100 == 0: 
        print i
    path = join(mypath, str(i))+'.fits'
    if not os.path.exists(path): continue
    x = pyfits.getdata(path,0,memmap=False)
    
    
    
    
    #y = x - np.min(x)
    #if np.max(y) != 0.0:
        #y = y / np.max(y)
    
    
    data.append(x)
    
X = np.array(data,dtype=np.float32)
print X.shape
save_hdf5_matrix('/users/dettmers/data/out_strong.hdf5',X)
    

    
