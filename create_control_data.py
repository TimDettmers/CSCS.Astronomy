import numpy as np
import cPickle as pickle
from util import *

path = '/home/tim/data/astro/'

tiles = pickle.load(open(path + 'tiles.p'))

X = load_hdf5_matrix(path + "out.hdf5")

for tile in tiles:
    print tile[0].shape

print len(tiles)

control_images = []
for tile in tiles:    
    if tile.shape[0] >= 10:
        x = np.ones((100,100),dtype=np.float32)*-100
        offsetx = 0
        offsety = 0
        size = tile.shape[0]
        while offsetx < 100 and offsety < 100:     
            helper = x[offsetx:offsetx + size, offsety:offsety+size]       
            x[offsetx:offsetx + size, offsety:offsety+size] = tile[0:helper.shape[0],0:helper.shape[1]]
            offsetx += size
            if offsetx > 100: 
                offsetx = 0
                offsety += size
        if np.min(x) != -100.0:
            print np.min(x)
            control_images.append(x)
        
save_hdf5_matrix(path + 'control_images.hdf5', np.array(control_images, dtype=np.float32))
            
        
'''
def flip_vert(picture):
    width = getWidth(picture)
    height = getHeight(picture)

    for y in range(0, height/2):
        for x in range(0, width):
            sourcePixel = getPixel(picture, x, y)
            targetPixel = getPixel(picture, x, height - y - 1)
            color = getColor(sourcePixel)
            setColor(sourcePixel, getColor(targetPixel))
            setColor(targetPixel, color)

    return picture 


def flip_horiz(picture):
    width = getWidth(picture)
    height = getHeight(picture)

    for y in range(0, height):
        for x in range(0, width/2):
            sourcePixel = getPixel(picture, x, y)
            targetPixel = getPixel(picture, width - x - 1, y)
            color = getColor(sourcePixel)
            setColor(sourcePixel, getColor(targetPixel))
            setColor(targetPixel, color)

    return picture 
'''