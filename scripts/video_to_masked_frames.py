import cv2
import numpy as np
import os
import generate_and_apply_masks_to_waterstreams as g

image = cv2.imread("white.png") 
image = cv2.resize(image,(1080,1920))
blackimage = cv2.imread("blackimage.jpg")
blackimage = cv2.resize(blackimage,(1080,1920))
video = cv2.VideoCapture('data/techneck4green.mp4')



path = '/home/arudrin/Documents/nivfrm-ml/data/test-data-mask/techneck'
count = 2400
ret = 1
ctr = 0
while ret:  #CTRL C at console/terminal to force stop loop
  
    ret, frame = video.read()
    if not ret:
    	print("END")
    	break
    ctr+=1
    if ctr==1201:
    	break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_green = np.array([65,45, 45]) 
    u_green = np.array([98, 255, 255])
     

  
    mask = cv2.inRange(hsv, l_green, u_green)
    edges = cv2.Canny(mask,150,200) 
    res = cv2.bitwise_and(frame, frame, mask = mask)
    contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    f = frame - res 
    f = np.where(f == 0, blackimage, image)
    cv2.drawContours(f,contours, -1, (255,255,255), 3)
    
    
    f,_,_,_,_ = g.resize_image(f, min_dim=240, max_dim=240, min_scale=None, mode="square")
    count+=1
    #f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    f = g.to_gray(f)
    cv2.imwrite( os.path.join(path , str(count)+".jpg"), f)

print("END")
