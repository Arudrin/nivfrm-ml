import cv2

import os,glob
from os import listdir,makedirs
from os.path import isfile,join

import subprocess

import tensorflow.compat.v1 as compat
import tensorflow.keras.optimizers as optmzrs
from keras.models import load_model

from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.predict import predict
from keras_segmentation.train import pixel_wise_loss,iou,dice



enable_cuda_cache = "export CUDA_CACHE_DISABLE=0"
set_cuda_cache    = "export CUDA_CACHE_MAXSIZE=2147483648"
allow_gpu_growth  = "export TF_FORCE_GPU_ALLOW_GROWTH=true"

_ = subprocess.check_output(['bash','-c', enable_cuda_cache])
_ = subprocess.check_output(['bash','-c', set_cuda_cache])
_ = subprocess.check_output(['bash','-c', allow_gpu_growth])

config = compat.ConfigProto()
config.gpu_options.allow_growth = True
session = compat.InteractiveSession(config=config)

batch_size = 1
model = mobilenet_unet(n_classes=2, input_height=240, input_width=240, batch_size=batch_size)
logs_path = "../logs/mobilenet_unet" 

#adam = optmzrs.Adam(amsgrad=True)
model.compile()

model.load_weights('/home/arudrin/Documents/nivfrm-ml/logs/mobilenet_unet_50e-0.015LR/checkpoints/weights-050-0.5345.hdf5')
	
	
	
source = '/home/arudrin/Documents/nivfrm-ml/results/source' # Source Folder / INPUT
dstpath = '/home/arudrin/Documents/nivfrm-ml/results/destination' # Destination Folder / OUTPUT


try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")
# Folder won't used
files = list(filter(lambda f: isfile(join(source,f)), listdir(source)))
for image in files:
	inp = os.path.join(source,image)
	out = os.path.join(dstpath,image)
	predict(model,inp,out)
	
	
exit()
#    try:
#        img = cv2.imread(os.path.join(path,image))
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        dstPath = join(dstpath,image)
#        cv2.imwrite(dstPath,gray)
#    except:
#        print ("{} is not converted".format(image))
        
        
        
        
#for fil in glob.glob("*.jpg"):
#    try:
#        image = cv2.imread(fil) 
#        gray_image = cv2.cvtColor(os.path.join(path,image), cv2.COLOR_BGR2GRAY) # convert to greyscale
#        cv2.imwrite(os.path.join(dstpath,fil),gray_image)
#    except:
#        print('{} is not converted')
