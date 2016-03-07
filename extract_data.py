import pyfits
from os import listdir
from os.path import isfile, join
import numpy as np
from util import *

mypath = '/home/tim/sync/RADIO_MAPS/out/clean/'

data = []
for i in range(50000):
    if i % 100 == 0: print i
    path = join(mypath, str(i))+'.fits'
    x = pyfits.getdata(path,0)
    y = x - np.min(x)
    if np.max(y) != 0.0:
        y = y / np.max(y)
    
    data.append(y)
    
    
X = np.array(data,dtype=np.float32)
print X.shape
save_hdf5_matrix('/home/tim/data/astro/clean.hdf5', X)
    

    
