import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from skimage import exposure
from skimage import restoration


path = '/home/tim/data/astro/'

X1 = load_hdf5_matrix(path + 'out.hdf5')
X2 = load_hdf5_matrix(path + 'noise.hdf5')

data = [X1, X2]
for i in range(10):
    print np.min(X1[i])
    print np.max(X1[i])

bin_n = 16 # Number of bins
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    
    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

hog_features = []
hog_features_processed = []
for X in data:
    for i, img in enumerate(X):        
        if i % 100 == 0: 
            if i > 0: 
                print i
                
        img_adapteq = np.abs(img)
        img_adapteq = exposure.equalize_adapthist(np.log(img_adapteq + 1.0), clip_limit=0.5,kernel_size=(4,4))
        X[i] = img_adapteq
        
        '''
        hog_features.append(hog(img)[:64])
        img -=  np.min(img)
        img /= np.max(img)
        img = np.uint8(img*255)
        img = cv2.fastNlMeansDenoising(img)
        X[i] = img
        hog_features_processed.append(hog(img)[:64])
        '''
        
        #img = restoration.denoise_tv_chambolle(img, weight=0.01)
        #X[i] = img
    
#save_hdf5_matrix(path + 'hog_processed.hdf5', np.array(hog_features_processed, dtype=np.float32))
#save_hdf5_matrix(path + 'hog.hdf5',np.array(hog_features, dtype=np.float32))
        
#print 'HOG features saved!'
        
save_hdf5_matrix(path + 'out_processed.hdf5', X1)
save_hdf5_matrix(path + 'noise_processed.hdf5', X2)




