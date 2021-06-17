import argparse
import glob
import json
import os
import random

import keras.backend as K
import matplotlib.pyplot as plt
import six
import tensorflow as tf
import keras.callbacks as kcb
from .data_utils.data_loader import (image_segmentation_generator,
                                     verify_segmentation_dataset)

#orientations   = ['vertical', 'horizontal']
#pipe_sizes     = ['one', 'three-fourth', 'one-half']
#valve_openings = ['25', '50', '75', '100']

#preprocessed_dir = '../data/preprocessed-images'
#masks_dir        = '../data/generated-masks'
#data_dir         = masks_dir

postures = ['proper', 'slouch', 'techneck']

preprocessed_dir = '/home/arudrin/Documents/nivfrm-ml/data/test-data'
masks_dir ='/home/arudrin/Documents/nivfrm-ml/data/test-data-mask'
data_dir = masks_dir

def train(model, optimizer, logs_path, epochs = 10, batch_size = 1):
    n_classes    = model.n_classes
    input_height = model.input_height
    input_width  = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    
    train_val_split = 0.8

    number_of_labels            = 3
    samples_each_label          = 3240
    number_of_available_samples = 10800

    label = 0
    img_seg_pairs = [[] for _ in range(number_of_labels)]
    for posture in postures:
    	picked_frames = []
    	for frame_number in range(samples_each_label):
    		frame_number = get_random_unpicked_sample_number(samples_each_label, picked_frames)
    		picked_frames.append(frame_number)
    		filename = str(frame_number) + '.jpg'
    		mask_path = os.path.join(masks_dir,posture,filename)
    		stream_path = mask_path.replace(masks_dir, preprocessed_dir)
    		img_seg_pairs[label].append((stream_path, mask_path))
    	label = label + 1
    
    train_val_boundary = int(train_val_split * samples_each_label)

    train_partition = []
    validation_partition = []

    for label in range(number_of_labels):
        train_partition      += img_seg_pairs[label][:train_val_boundary]
        validation_partition += img_seg_pairs[label][train_val_boundary:]

    training_steps   = len(train_partition)      // batch_size
    validation_steps = len(validation_partition) // batch_size

    train_generator      = image_segmentation_generator(train_partition     ,  batch_size, n_classes, input_height, input_width, output_height, output_width)
    validation_generator = image_segmentation_generator(validation_partition,  batch_size, n_classes, input_height, input_width, output_height, output_width)

    os.makedirs(logs_path, exist_ok=True)
    checkpoints_name = "weights-{epoch:03d}-{loss:.4f}.hdf5"  #"weights-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5" removed val_loss so i could callback
    checkpoints_path = os.path.join(logs_path + '/' + 'checkpoints', checkpoints_name)

    model_chkpnts = kcb.ModelCheckpoint(checkpoints_path, verbose=1, save_freq=5*8640)
    early_stop    = kcb.EarlyStopping(monitor='val_loss', patience=25, verbose=1)
    csv_logger    = kcb.CSVLogger(logs_path + '/' + 'training_log.csv', append=True)

    model.compile(loss = pixel_wise_loss, optimizer = optimizer, metrics = [iou, dice])
    model.summary()

    history = model.fit_generator( epochs           = epochs
                                 , steps_per_epoch  = training_steps
                                 , generator        = train_generator
                                 , validation_data  = validation_generator
                                 , validation_steps = validation_steps
                                 , callbacks        = [ model_chkpnts
                                                      , csv_logger
                                                      , early_stop
                                                      ]
                                 , shuffle          = True
                                 )

    plot_training_history(history, logs_path)

    return model

def get_random_unpicked_sample_number(number_of_samples, picked_samples):
    sample_number = random.randint(1, number_of_samples)
    while sample_number in picked_samples:
       sample_number = random.randint(1, number_of_samples)
    return sample_number

def plot_training_history(history, path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Pixel Wise Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(path + '/' + 'segmentation-pw-loss.png', bbox_inches='tight')

    plt.clf()

    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.title('Model IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(path + '/' + 'segmentation-iou.png', bbox_inches='tight')

    plt.clf()

    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.title('Model Dice')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(path + '/' + 'segmentation-dice.png', bbox_inches='tight')

def pixel_wise_loss(y_true, y_pred):
    pos_weight = tf.constant([[1.0, 2.0]])
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        y_pred,
        pos_weight,
        name=None
    )

    return K.mean(loss, axis=-1)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    # average over classes and batch
    return 1 - np.mean(numerator / (denominator + epsilon))

def soft_dice(y_true, y_pred):
    return soft_dice_loss(y_true, y_pred)

# thanks to: wassname https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
def jaccard_distance_loss(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# thanks to: https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
def mean_iou(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes):  # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        # returns average IoU of the same objects
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches))))
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

# thanks to: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou(y_true, y_pred):
    smooth = 1
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1)+K.sum(y_pred, -1)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice(y_true, y_pred):
    smooth = 1
    intersection = K.sum(y_true * y_pred, axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
