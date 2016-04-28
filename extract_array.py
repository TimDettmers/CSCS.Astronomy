import numpy as np
from util import *
import Image
from os.path import join
import os



 
  
 #0=correctly identified as not containing filaments;
 #1=correctly identified as containing a filament; 
 #2=wrongly identified as not containing a filament; 
 #3=wrongly identified as containing a filament
 
path = '/home/tim/data/astro/'
X = load_hdf5_matrix(path + 'X_processed.hdf5')
idx = load_hdf5_matrix(path + 'idx.hdf5')
error_idx = load_hdf5_matrix(path + 'error_idx.hdf5')

test = 0.2
cv = 0.1
train = 0.7
n = idx.shape[0]
print n
testn = int(test*n)
cvn = int(cv*n)
trainn = int(train*n)

print X.shape
idx_test = idx[-testn:]
X_test = X[-testn:]
print error_idx.shape

img = []
quadrant = {}
for i, (idx_img, error) in enumerate(zip(idx_test, error_idx)):
    #0-50000 out
    #50000-100000 noise
    #-> repeat
    if not (idx_img > 0 and idx_img < 50000 or 
       idx_img > 200000 and idx_img < 250000): continue
    isFilament =  idx_img < 200000
    if isFilament:
        file_path = 'out/{0}.jpg'.format(idx_img%50000)
        
        if error == 0:
            quadrant[idx_img%50000] = 1
        else:
            quadrant[idx_img%50000] = 2
    else:
        file_path = 'out/noise/{0}.jpg'.format(idx_img%50000)
        if error == 0:
            quadrant[idx_img%50000] = 0
        else:
            quadrant[idx_img%50000] = 3
    print idx_img, isFilament, idx_img%50000, file_path
    arr = X_test[i].reshape(100,100)
    im = Image.fromarray(np.uint8(arr*255))
    
    folder = os.path.dirname(join(path,file_path))
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    im.save(join(path, file_path))

for imgno in quadrant:
    cls = quadrant[imgno]
    if cls == 3 or cls == 1:
        img.append(['out/noise/{0}.jpg'.format(imgno),cls])
    else:
        img.append(['out/{0}.jpg'.format(imgno),cls])
        
                   

        
        
with open(path + 'test_data.txt','wb') as f:
    for value in img:

        
        f.write("{0}, {1}\n".format(value[0],value[1]))
    

print len(quadrant.keys())