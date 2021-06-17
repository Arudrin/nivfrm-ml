import cv2
import numpy as np
import os
import generate_and_apply_masks_to_waterstreams as g

video = cv2.VideoCapture('data_in_scriptfolder/forward_p2_1.mov')


path = '/home/arudrin/Documents/nivfrm-ml/scripts/data_in_scriptfolder/checkingpool'
count = 0
ret = 1
ctr = 0

while ret:

    ret, frame = video.read()
    if not ret:
    	break
    ctr+=1
    if ctr == 1201:
    	break
    
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame,_,_,_,_ = g.resize_image(frame, min_dim=240, max_dim=240, min_scale=None, mode="square")
    #frame = g.to_gray(frame)
    count+=1
    cv2.imwrite( os.path.join(path ,str(count)+".jpg"), frame)
    ctr
    
print("END")


