import glob
import random
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from keras.models import load_model

from .data_utils.data_loader import get_image_array, get_segmentation_array, DATA_LOADER_SEED, class_colors
from .models.config import IMAGE_ORDERING
from . import metrics

import six

random.seed(DATA_LOADER_SEED)

preprocessed_dir = '/home/arudrin/Documents/nivfrm-ml/data/test-data'
predicted_dir = '/home/arudrin/Documents/nivfrm-ml/data/predicted-mask'


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None):

    assert (inp is not None)
    assert((type(inp) is np.ndarray) or isinstance(inp, six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes))
    pr = pr.argmax(axis=2)
    
    seg_img = np.zeros((output_height, output_width))
    colors = class_colors
    colors[0] = 0
    colors[1] = 255

    for c in range(n_classes):
        seg_img[:, :] = ((pr[:, :] == c) * (colors[c])).astype('uint8')

    orininal_h = orininal_w = 240
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        
    return pr
    
def call_predict(model=None, inp=None,out_dir=None):
	return predict(model,inp,out_dir)


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None):

	batch_size = 100
	inps = []
	for number in range(1,batch_size+1):
		random_frame_number =random.randrange(3240)+1
		for i in inp_dir:
			img_path = os.path.join(i,str(random_frame_number) + '.jpg')
			inps += [img_path]
			
	assert type(inps) is list
	
	for i,inp in enumerate(tqdm(inps)):
		out_dir = inp.replace(preprocessed_dir, predicted_dir)
		print( predict(model,inp,out_dir))
    #for batch_start in range(1000, 1001, batch_size):
    #    inps = []

     #   for frame_number in range(batch_start, batch_start+batch_size):
     #       for i in inp_dir:
      #          img_path = os.path.join(i, str(frame_number) + '.jpg')                
      #          inps += [img_path]
#OLD
          #assert type(inps) is list

        #for i, inp in enumerate(tqdm(inps)):
          #  out_dir = inp.replace(preprocessed_dir, predicted_dir)
           # _ = predict(model, inp, out_dir)

def evaluate(model=None, inp_images=None, annotations=None,
             checkpoints_path=None):

    assert False, "not implemented "

    ious = []
    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(
            ann, model.n_classes,  model.output_width, model.output_height)
        gt = gt.argmax(-1)
        iou = metrics.get_iou(gt, pr, model.n_classes)
        ious.append(iou)
    ious = np.array(ious)
    print("Class wise IoU ",  np.mean(ious, axis=0))
    print("Total  IoU ",  np.mean(ious))
