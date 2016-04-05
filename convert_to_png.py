import pyfits
from os import listdir
from os.path import isfile, join
import numpy as np
from util import *
import gc
import Image

mypath = '/home/tim/sync/RADIO_MAPS/'

folders = ['out','out/clean','out/noise']

imgs = range(100) + range(10000,10100) + range(40000,40100)

print imgs

data = []
for folder in folders:
    for i in imgs:
        if i % 100 == 0: 
            print i
        path = join(mypath, folder,str(i))+'.fits'
        x = pyfits.getdata(path,0,memmap=False)
        #y = x - np.min(x)
        #if np.max(y) != 0.0:
            #y = y / np.max(y)
        
        
        y = x - np.min(x)
        y = y / np.max(y)
        
        y = np.fliplr(y)
        im = Image.fromarray(np.uint8(y*255))
        im.save(join('/home/tim/images/flipped/',folder, str(i)) +'.png')
        