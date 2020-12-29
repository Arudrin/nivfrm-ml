import contextlib
import csv
import datetime as dt
import math
import os
import subprocess

import cv2
import keras
import keras.backend as kbe
import keras.callbacks as kcb
import keras.layers as kl
import keras.models as km
import keras.utils as kutils
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as sklp
import tensorflow as tf
import tensorflow.compat.v1 as compat
import tensorflow.keras.optimizers as optmzrs
from keras.layers import (RNN, BatchNormalization, Conv2D, Dense, Dropout,
                          Flatten, Input, LSTMCell, MaxPooling2D)

"""

Environment

tf-nightly-gpu == 2.5.0-dev20201210
cuda           == 11.0.3-1
libcudnn8      == 8.0.4.30
libcudnn8-dev  == 8.0.4.30

RTX 3070 8GB
Driver Version 455.45.01
Ubuntu 18.04
Python 3.6.9

"""

### Knobs and Dials

EPOCHS     = 1000
BATCH_SIZE = 1

# 1 - use every frame, no interleave
# 2 - use every other frame
INTERLEAVE = 1
FRAMES = 30

SAMPLES = 100

WIDTH   = 240
HEIGHT  = 240

TRAINING_SET_PERCENT   = 0.7
VALIDATION_SET_PERCENT = 0.2
TEST_SET_PERCENT       = 0.1

## Directories

WATERSTREAMS_DIR = '../data/waterstreams'

LOGS_DIR     = '../logs'
SESSION_NAME = str(FRAMES) + 'F-' + str(INTERLEAVE) +'i-' + str(SAMPLES) + 'S'
SESSION_DIR  = LOGS_DIR + '/' + SESSION_NAME

CHKPNT_DIR  =  LOGS_DIR + '/' + SESSION_NAME + '/' + 'checkpoints'
CHKPNT_NAME = "weights–{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5"

## Data

ORIENTATIONS   = ['vertical', 'horizontal']
PIPE_SIZES     = ['one', 'three-fourth', 'one-half']
VALVE_OPENINGS = ['25', '50', '75', '100']


def main():
    N_CLASSES = len(ORIENTATIONS) * len(PIPE_SIZES) * len(VALVE_OPENINGS)

    # Parameters for the Data Generators
    params = { 'dim'        : (WIDTH, HEIGHT)
             , 'batch_size' : BATCH_SIZE 
             , 'n_classes'  : N_CLASSES
             , 'n_channels' : 1
             }

    timestamp  = '{:%Y%m%d–%H%M%S}'.format(dt.datetime.now())
    model_name = timestamp + '.h5'
    model_path = SESSION_DIR + '/' + model_name

    create_dirs()

    labels, partition = load_dataset()

    training_generator   = DataGenerator(partition['trn'], labels, shuffle=True , **params)
    validation_generator = DataGenerator(partition['val'], labels, shuffle=True , **params)
    test_generator       = DataGenerator(partition['tst'], labels, shuffle=False, **params)

    training_samples   = len(partition['trn'])
    validation_samples = len(partition['val'])
    test_samples       = len(partition['tst'])

    training_steps   = training_samples   // BATCH_SIZE
    validation_steps = validation_samples // BATCH_SIZE
    test_steps       = test_samples       // BATCH_SIZE

    model_chkpnt = kcb.ModelCheckpoint(CHKPNT_DIR + '/' + CHKPNT_NAME)
    csv_logger   = kcb.CSVLogger(SESSION_DIR + '/' + 'training-log.csv')
    early_stop   = kcb.EarlyStopping(monitor='val_loss', patience=25, verbose=1)

    model = build_model(input_shape=(WIDTH, HEIGHT, 1))
    kutils.plot_model(model, SESSION_DIR + '/' + 'model-plot.png')

    history = model.fit_generator( epochs           = EPOCHS
                                 , generator        = training_generator
                                 , steps_per_epoch  = training_steps
                                 , validation_data  = validation_generator
                                 , validation_steps = validation_steps
                                 , callbacks        = [ model_chkpnt
                                                      , csv_logger
                                                      , early_stop
                                                      ]
                                 )

    model.save(model_path)    

    test_mse, test_mae = model.evaluate_generator( generator = test_generator
                                                 , steps= test_steps
                                                 )
    log('test_mse', test_mse)
    log('test_mae', test_mae)
    print('test_mse: ', test_mse)
    print('test_mae: ', test_mae)
    
    test_predictions = model.predict_generator( generator = test_generator
                                              , steps     = test_steps
                                              )

    test_predictions = test_predictions.flatten()

    plot_training_history(history)
    plot_test_predictions(partition['tst'], test_predictions)
    convert_to_lite(model)   

def convert_to_lite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(SESSION_DIR + '/' + 'regression.tflite', 'wb') as f:
        f.write(tflite_model)


def get_label(uid):
    for key in flow_map.keys():
        if key in uid:
            return key
    return uid

def create_dirs():
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(CHKPNT_DIR , exist_ok=True)

def plot_training_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(SESSION_DIR + '/mse.png', bbox_inches='tight')

    plt.clf()

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model Mean Aboslute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(SESSION_DIR + '/mae.png', bbox_inches='tight')

def plot_test_predictions(test_samples, test_predictions):
    test_ground_truths = get_ground_truths(test_samples)

    plt.clf()
    plt.axes(aspect='equal')
    plt.scatter(test_ground_truths, test_predictions)
    plt.xlabel('Ground Truth [L/min]')
    plt.ylabel('Predictions [L/min]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig(SESSION_DIR + '/test-predictions-scatter.png', bbox_inches='tight')

    plt.clf()

    error = test_predictions - test_ground_truths
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    plt.ylabel('Count')
    plt.savefig(SESSION_DIR + '/test-predictions-distribution.png', bbox_inches='tight')

    test_set_length = len(test_predictions)
    with open(SESSION_DIR + '/model-vs-groundtruth-test-predictions.csv', 'w') as pred_csvfile:
        writer = csv.writer(pred_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for idx in range(test_set_length):
            writer.writerow([test_samples[idx], test_ground_truths[idx], test_predictions[idx]])

def get_ground_truths(samples : []) -> []:
    ground_truths = []
    for sample in samples:
        flow = get_flow_of_path(sample)
        ground_truths.append(flow)
    return ground_truths

flow_map = {
      'vertical/one/25'           : 0.11
    , 'vertical/one/50'           : 0.32
    , 'vertical/one/75'           : 0.57
    , 'vertical/one/100'          : 0.77
    , 'vertical/one-half/25'      : 0.07
    , 'vertical/one-half/50'      : 0.16
    , 'vertical/one-half/75'      : 0.30
    , 'vertical/one-half/100'     : 0.41
    , 'vertical/three-fourth/25'  : 0.10
    , 'vertical/three-fourth/50'  : 0.21
    , 'vertical/three-fourth/75'  : 0.36
    , 'vertical/three-fourth/100' : 0.45

    , 'horizontal/one/25'           : 0.15
    , 'horizontal/one/50'           : 0.42
    , 'horizontal/one/75'           : 0.61
    , 'horizontal/one/100'          : 0.76
    , 'horizontal/one-half/25'      : 0.07
    , 'horizontal/one-half/50'      : 0.17
    , 'horizontal/one-half/75'      : 0.26
    , 'horizontal/one-half/100'     : 0.32
    , 'horizontal/three-fourth/25'  : 0.10
    , 'horizontal/three-fourth/50'  : 0.23
    , 'horizontal/three-fourth/75'  : 0.32
    , 'horizontal/three-fourth/100' : 0.43 }

def get_flow_of_path(path):
    for key in flow_map.keys():
        if (key in path):
            return flow_map[key] * 60 # convert L/s to L/min

def log(tag, message):
    with open(SESSION_DIR + '/' + 'log.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print(tag, ': ', message)

def load_dataset():
    labels = {}
    trn = []
    val = []
    tst = []

    for orientation in ORIENTATIONS:
        for pipe_size in PIPE_SIZES:
            for vo in VALVE_OPENINGS:
                images_dir = os.path.join(WATERSTREAMS_DIR, orientation, pipe_size, vo)
                flow = get_flow_of_path(images_dir)
                (_, _, filenames) = next(os.walk(images_dir))
                filenames = filenames[:FRAMES*INTERLEAVE*SAMPLES]
                filenames_length = len(filenames)
                image_paths = []
                for frame_start in range(0, filenames_length, FRAMES*INTERLEAVE):
                    image_filename = str(frame_start) + '.jpg'
                    image_path  = os.path.join(images_dir, image_filename)
                    labels[image_path] = flow
                    image_paths.append(image_path)

                image_paths_length = len(image_paths)
                boundary1  = math.ceil(image_paths_length * TRAINING_SET_PERCENT)
                boundary2  = math.ceil(image_paths_length * (TRAINING_SET_PERCENT + VALIDATION_SET_PERCENT))

                np.random.shuffle(image_paths)
                trn += image_paths[         :boundary1]
                val += image_paths[boundary1:boundary2]
                tst += image_paths[boundary2:         ]

    log('# of samples'           , len(labels))
    log('# of training samples'  , len(trn)   )
    log('# of validation samples', len(val)   )
    log('# of test samples'      , len(tst)   )

    print('# of samples'           , len(labels))
    print('# of training samples'  , len(trn)   )
    print('# of validation samples', len(val)   )
    print('# of test samples'      , len(tst)   )

    partition        = {}
    partition['trn'] = trn
    partition['val'] = val
    partition['tst'] = tst

    write_list_to_csv(trn, 'training-samples'  )
    write_list_to_csv(val, 'validation-samples')
    write_list_to_csv(tst, 'test-samples'      )

    return labels, partition

def write_list_to_csv(the_list, csv_filename):
    with open(SESSION_DIR + '/' + csv_filename + '.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for item in the_list:
                writer.writerow([item])

### credits to: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True):
        'Initialization'
        self.dim        = dim
        self.labels     = labels
        self.shuffle    = shuffle
        self.list_IDs   = list_IDs
        self.n_classes  = n_classes
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        batch_per_epoch = int(np.floor(len(self.list_IDs) / self.batch_size))
        return batch_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Xs = [X] * FRAMES
        y = np.zeros((self.batch_size), dtype=np.float32)

        # Generate data
        for batch, ID in enumerate(list_IDs_temp):
            filename   = ID.split('/')[-1]        
            image_path = ID.split('/')[:-1]     
            image_path = os.path.join(*image_path)

            frame_start = int(filename.split('.')[0])
            label       = self.labels[ID] 
            
            for frame in range(FRAMES):
                frame_number   = frame_start + (frame * INTERLEAVE)
                image_filename = str(frame_number) + '.jpg'

                abs_path = os.path.join(image_path, image_filename)
                extracted = cv2.imread(abs_path)                    

                extracted = extracted[:,:,0]

                mm_scaler = sklp.MinMaxScaler()
                extracted = mm_scaler.fit_transform(extracted)

                Xs[frame][batch] = extracted.reshape((WIDTH, HEIGHT, 1))

            y[batch] = label
       
        return Xs, y

def build_model(input_shape):
    batch_shape = (BATCH_SIZE,) + input_shape

    inputs = []
    for _ in range(FRAMES):
        input = Input(batch_shape = batch_shape)
        inputs.append(input)
    
    layers = 1
    conv_outputs = []
    for input in inputs:
        filters = 8
    
        conv_input = input
        for _ in range(layers):
            conv = Conv2D(filters, kernel_size=3, activation='relu', padding='same')(conv_input)
            bnrm = BatchNormalization()(conv)        
            pool = MaxPooling2D(8)(bnrm)
            conv_input = pool
            filters *= 2

        flattened = Flatten()(conv_input)
        conv_outputs.append(flattened)
    
    stacked = Stack()(conv_outputs)

    units = FRAMES
    
    cell  = LSTMCell(units)
    lstm  = RNN(cell, unroll=True)(stacked)
    drpt1 = Dropout(0.2)(lstm)

    dense = Dense(units//2, activation='relu')(drpt1)
    drpt2 = Dropout(0.2)(dense)
    
    output = Dense(1)(drpt2)

    model = km.Model(inputs=inputs, outputs=output)

    adam = optmzrs.Adam(learning_rate=0.0001, amsgrad=True)
    model.compile( loss      = 'mse'
                 , optimizer = adam
                 , metrics   = ['mae']
                 )

    model.summary()

    return model

class Stack(kl.Layer):
    'Stacks sequences'

    def __init__(self, **kwargs):
        super(Stack, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Stack, self).build(input_shape)

    def call(self, inputs):
        stacked = kbe.stack(inputs, axis = 1)
        return stacked

    def compute_output_shape(self, input_shape):
        number_of_frames = len(input_shape)
        a = input_shape[0]
        batch_size = a[0]
        sequences  = a[1]
        output_shape = (batch_size, number_of_frames, sequences)
        return output_shape

    def get_config(self):
        return { }

if __name__ == "__main__":
    enable_cuda_cache = "export CUDA_CACHE_DISABLE=0"
    set_cuda_cache    = "export CUDA_CACHE_MAXSIZE=2147483648"
    allow_gpu_growth  = "export TF_FORCE_GPU_ALLOW_GROWTH=true"

    _ = subprocess.check_output(['bash','-c', enable_cuda_cache])
    _ = subprocess.check_output(['bash','-c', set_cuda_cache])
    _ = subprocess.check_output(['bash','-c', allow_gpu_growth])
    
    config = compat.ConfigProto()
    config.gpu_options.allow_growth = True
    session = compat.InteractiveSession(config=config)

    main()