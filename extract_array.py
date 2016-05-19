import numpy as np
from util import *
from os.path import join
import os
import cPickle as pickle
from PIL import Image



 
  
 #0=correctly identified as not containing filaments;
 #1=correctly identified as containing a filament; 
 #2=wrongly identified as not containing a filament; 
 #3=wrongly identified as containing a filament
 
path = '/users/dettmers/data/'
X = load_hdf5_matrix(path + 'X_processed.hdf5')
idx = load_hdf5_matrix(path + 'idx.hdf5')
P = pickle.load(open(join(path,'P_weak.p')))
error_idx = load_hdf5_matrix(path + 'error_idx.hdf5')
pred  = load_hdf5_matrix(path + 'error_softmax.hdf5')

if not os.path.exists(path.replace('/data','/results')):
	os.mkdir(path.replace('/data','/results'))
	os.mkdir(join(path.replace('/data','/results'),'out'))
	os.mkdir(join(path.replace('/data','/results'),'out','noise'))
	os.mkdir(join(path.replace('/data','/results'),'out_weak'))
	os.mkdir(join(path.replace('/data','/results'),'out_strong'))

test = 0.2
cv = 0.1
train = 0.7
n = idx.shape[0]
print n
testn = int(test*n)
cvn = int(cv*n)
trainn = int(train*n)

print X.shape
idx_test = P[-testn:]
X_test = X[-testn:]
print error_idx.shape


img = []
quadrant = {}
with open(path + 'test_data.txt','wb') as f:
	for i, (img_path, error) in enumerate(zip(idx_test, error_idx)):
	    #0-50000 out
	    #50000-100000 noise
	    #-> repeat
	    isFilament =  'noise' not in img_path[0]
	    if isFilament:
		if error == 0:
        	    f.write("{0}, {1}, {2}, {3}\n".format(img_path[0].replace('fits','jpg'),1,pred[i,0],pred[i,1]))
		else:
        	    f.write("{0}, {1}, {2}, {3}\n".format(img_path[0].replace('fits','jpg'),2,pred[i,0],pred[i,1]))
	    else:
		if error == 0:
        	    f.write("{0}, {1}, {2}, {3}\n".format(img_path[0].replace('fits','jpg'),0,pred[i,0],pred[i,1]))
		else:
        	    f.write("{0}, {1}, {2}, {3}\n".format(img_path[0].replace('fits','jpg'),3,pred[i,0],pred[i,1]))

	    arr = X_test[i].reshape(100,100)
	    im = Image.fromarray(np.uint8(arr*255))
		
	    im.save(join(path, img_path[0].replace('/data','/results').replace('fits','jpg')))

