import cv2
import numpy as np
import os
import generate_and_apply_masks_to_waterstreams as g

video = cv2.VideoCapture('../data/slouch2.mp4')


path = '../data/test-data/slouch' 
count = 0
ret = 1

while ret:

    ret, frame = video.read()
    if not ret:
    	break
    	
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #frame = cv2.resize(frame,(1080,1080))
    #frame,_,_,_,_ = g.resize_image(frame, min_dim=1080, max_dim=1080, min_scale=None, mode="square")
    

    count+=1
    cv2.imwrite( os.path.join(path , "slouch"+str(count)+".jpg"), image)
    
    
print("END")


