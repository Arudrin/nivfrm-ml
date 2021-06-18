import cv2

import os,glob
from os import listdir,makedirs
from os.path import isfile,join

import subprocess


	
	
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
	

print(bitmap[184,124]) # bitmap[y,x]


