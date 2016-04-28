from util import *
from matplotlib import pyplot as plt


path = '/home/tim/data/astro/'


X1 = load_hdf5_matrix(path + 'out.hdf5')
X2 = load_hdf5_matrix(path + 'denoised.hdf5')
X3 = load_hdf5_matrix(path + 'clean.hdf5')

for i in range(0,50000,1000):
    

    img1 = X1[i]
    img2 = X2[i]
    img3 = X3[i]
    
    print img1[10:20,0]
    print img2[10:20,0]
    print img3[10:20,0]
    
    #print des.shape    
    
    plt.subplot(131),plt.imshow(img1)
    plt.subplot(132),plt.imshow( img2)
    plt.subplot(133),plt.imshow( img3)
    plt.show()




