import os
import subprocess

import tensorflow.compat.v1 as compat
import tensorflow.keras.optimizers as optmzrs
from keras.models import load_model

from keras_segmentation.models.segnet import mobilenet_segnet
from keras_segmentation.models.unet import mobilenet_unet

orientations = ['vertical', 'horizontal']
pipe_sizes   = ['one', 'three-fourth', 'one-half']
labels       = ['25', '50', '75', '100']

preprocessed_dir = '../data/preprocessed-images'
generated_dir = '../data/generated-masks'
predicted_dir = '../data/predicted-masks'


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

# model = mobilenet_segnet(n_classes=2, input_height=240, input_width=240)
# logs_path = "../logs/mobilenet_segnet"

model = mobilenet_unet(n_classes=2, input_height=240, input_width=240, batch_size=batch_size)
logs_path = "../logs/mobilenet_unet" 

os.makedirs(logs_path, exist_ok=True)

adam = optmzrs.Adam(amsgrad=True)
model = model.train(optimizer = adam, logs_path = logs_path, batch_size = batch_size)

model.convert_to_lite(path = logs_path)

inp_dirs = []
for orientation in orientations:
    for pipe_size in pipe_sizes:
        for label in labels:
            inp_dir = os.path.join(preprocessed_dir, orientation, pipe_size, label)
            inp_dirs.append(inp_dir)

            out_dir = inp_dir.replace(preprocessed_dir, predicted_dir)
            os.makedirs(out_dir, exist_ok=True)

model.predict_multiple(inp_dir=inp_dirs, checkpoints_path=logs_path)
