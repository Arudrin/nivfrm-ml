import cv2

import os,glob
from os import listdir,makedirs
from os.path import isfile,join

import subprocess
import numpy as np

	
	
source = '/home/arudrin/Documents/nivfrm-ml/results/destination/' # Source Folder / INPUT
dstpath = '/home/arudrin/Documents/nivfrm-ml/results/destination/map' # Destination Folder / OUTPUT


try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")

files = list(filter(lambda f: isfile(join(source,f)), listdir(source)))

first = True
for image in files:
	if first == True:
		bitmap = cv2.imread(os.path.join(source,image), cv2.IMREAD_GRAYSCALE).astype(int) # as type int is needed to exceed 255 cap
		first = False
		continue
	
	img = cv2.imread(os.path.join(source,image), cv2.IMREAD_GRAYSCALE).astype(int)
	#print(img)
	#bitmap = img + bitmap
	bitmap = cv2.add(bitmap, img)





#loop through bitmap - the image to be analyzed	
y_index = 0	
blank = [] #array to be transformed to a heatmap
for row in bitmap:
	x_index = 0
	blank_row = []
	for value in row:
		blank_row.append((255,0,0))
	y_index+=1
	blank.append(blank_row)

maparray = np.array(blank)
heatmap = maparray.astype(np.uint8)
print(type(heatmap))
print(heatmap)
#heat = cv2.applyColorMap(bitmap,cv2.COLORMAP_JET)
cv2.imshow('heat',heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

 # bitmap[y,x]


