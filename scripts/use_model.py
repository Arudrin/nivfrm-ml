import os
import subprocess

import tensorflow.compat.v1 as compat
import tensorflow.keras.optimizers as optmzrs
from keras.models import load_model

from keras_segmentation.models.segnet import mobilenet_segnet
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


#FOR LOADING SAVED MODELS , GIVES VALUE ERROR, asks for custom_objects
#model = load_model('/home/arudrin/Documents/nivfrm-ml/scripts/mobilenet_50E-0.015LR_saved_model')

model = mobilenet_unet(n_classes=2, input_height=240, input_width=240, batch_size=batch_size)
logs_path = "../logs/mobilenet_unet" 

adam = optmzrs.Adam(amsgrad=True)
model.compile(loss = pixel_wise_loss, optimizer = adam, metrics = [iou, dice])

model.load_weights('/home/arudrin/Documents/nivfrm-ml/logs/mobilenet_unet_50e-0.015LR/checkpoints/weights-050-0.5345.hdf5')

#val = inference result, inp = input
val = '/home/arudrin/Documents/nivfrm-ml/val/where.jpg'
inp = '/home/arudrin/Documents/nivfrm-ml/scripts/data_in_scriptfolder/raw-240x240/proper/1.jpg' 
predict(model,inp,val) #modified model.predict from /keras_segmentation/predict.py
	

