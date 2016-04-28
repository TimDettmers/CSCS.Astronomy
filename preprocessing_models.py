import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from skimage import exposure
from skimage import restoration
from scipy.ndimage import median_filter
import Image

SZ=20
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

path = '/home/tim/data/astro/'

X1 = load_hdf5_matrix(path + 'out.hdf5')
#X2 = load_hdf5_matrix(path + 'noise.hdf5')
X3 = load_hdf5_matrix(path + 'clean.hdf5')

for i in range(0,50000,1000):
    
    img3 = Image.open("/home/tim/sift_keypoints.jpg")
    img = X1[i]
    real = X3[i]
    
    log_img = np.log(np.abs(img) + 1.0)
    
    p2, p98 = np.percentile(log_img, (2, 98))
    img_rescale = exposure.rescale_intensity(log_img, in_range=(p2, p98))
    img_eq = exposure.equalize_hist(log_img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.5,kernel_size=(4,4))
    
    #img_adapteq3 = img - np.min(img)
    img_adapteq3 = exposure.equalize_adapthist(log_img, clip_limit=0.5,kernel_size=(4,4))
    img_adapteq4 = exposure.equalize_adapthist(log_img, clip_limit=0.01,kernel_size=(2,2),nbins=12)
    
    
    
    #denois_img = img - np.min(img)
    denois_img = img_adapteq - np.min(img_adapteq)
    denois_img /= np.max(denois_img)
    denois_img = np.uint8(denois_img*255)
    
    
    tv_coins = restoration.richardson_lucy(log_img,)
    better_contrast = exposure.rescale_intensity(denois_img)
    dst = cv2.fastNlMeansDenoising(better_contrast)
    dst2 = cv2.fastNlMeansDenoising(denois_img)
    im_med = median_filter(img, 5)
    #sift = cv2.Feat .SIFT_create()
    
          
    
    print hog(img)
    
    #print des.shape    
    
    plt.subplot(131),plt.imshow(real)
    plt.subplot(132),plt.imshow(tv_coins )
    plt.subplot(133),plt.imshow( img_adapteq4)
    plt.show()




