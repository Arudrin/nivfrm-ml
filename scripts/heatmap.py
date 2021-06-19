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

#create blank_image to become heatmap
blank_image = np.zeros((240,240,3), np.uint8)


#loop through bitmap - the image to be analyzed	
y_index = 0	
for row in bitmap:
	x_index = 0
	for value in row:
		blank_image[y_index,x_index] = [255,255,255]
	y_index+=1


#heat = cv2.applyColorMap(bitmap,cv2.COLORMAP_JET)
cv2.imshow('heat',blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(bitmap[184,124]) # bitmap[y,x]


