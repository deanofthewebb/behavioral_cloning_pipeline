# Dataset Parameters
DRIVING_LOG_CSV = 'full_driving_log.csv'
MODEL_DATA = 'model.h5'

# Image Augmentation
CORRECTION_ANGLE = 0.25
NB_AUGMENTED_SAMPLES = 1000

# Image Processing
DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_DEPTH = (64 , 64, 3)
DEFAULT_RESOLUTION = (DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_DEPTH) if DEFAULT_DEPTH > 1 else (DEFAULT_LENGTH, DEFAULT_WIDTH)
DATASET_DIRECTORY = 'merged_data/'

# Validation Dataset
VALIDATION_PORTION = 0.222

import csv
import cv2
import os
import numpy as np
import math

def read_csv(filepath, num_features=7, delimiter=';'):
    data_array = np.array(np.zeros(shape=num_features), ndmin=2)
    with open(filepath, newline='') as csvfile:
        annotations_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in annotations_reader:
            data_array = np.vstack((data_array, np.array(row, ndmin=2)))
    return data_array[2:]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

drive_data = pd.read_csv(os.path.join(DATASET_DIRECTORY,DRIVING_LOG_CSV))
drive_data.head()

def translate_image(image,steer,trans_range):
    # Translation
    delta_x = trans_range*np.random.uniform()-trans_range/2
    steering_angle = steer + delta_x/trans_range*2*.2
    delta_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    translated_image = cv2.warpAffine(image,Trans_M,(DEFAULT_LENGTH,DEFAULT_WIDTH))

    return translated_image,steering_angle

def augment_brightness_camera_images(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def randomly_add_shadow_effect(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    return cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

def resize_image(image):
    shape = image.shape
    # Crop numpy array of image to remove extraneous pixels
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    if (shape[0] != DEFAULT_RESOLUTION[0] or shape[1] != DEFAULT_RESOLUTION[1]):
        # Resize numpy array, note numpy arrays are formatted with (ROW, COL, CH)
        image = cv2.resize(image,(DEFAULT_RESOLUTION[1],DEFAULT_RESOLUTION[0]), interpolation=cv2.INTER_AREA)
    return image


def augment_brightness_camera_images(image):
    v_ch = 2
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_light = .25+np.random.uniform()
    image1[:,:,v_ch] = image1[:,:,v_ch]*random_light
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def translate_image(image,steer,trans_range):
    shape = image.shape
    # Translation
    delta_x = trans_range*np.random.uniform()-trans_range/2
    steering_angle = steer + delta_x/trans_range*2*.2
    delta_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    translated_image = cv2.warpAffine(image,Trans_M,(shape[0], shape[1]))
    return translated_image,steering_angle


def randomly_add_shadow_effect(image):
    top_y = DEFAULT_LENGTH*np.random.uniform()
    top_x = 0
    bot_x = DEFAULT_WIDTH
    bot_y = DEFAULT_LENGTH*np.random.uniform()
    s_ch = 1
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) #HLS
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,s_ch][cond1] = image_hls[:,:,s_ch][cond1]*random_bright
        else:
            image_hls[:,:,s_ch][cond0] = image_hls[:,:,s_ch][cond0]*random_bright
    return cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

def randomly_flip_image(image, measurement):
    if (np.random.randint(2) == 0):
        image = cv2.flip(image,1)
        measurement = -measurement
    return image, measurement


def preprocess_image(line_data, features):
    random_index = np.random.randint(3)
    if (random_index == 0):
        filepath = line_data['left'][0].strip()
        shifted_ang = CORRECTION_ANGLE
    if (random_index == 1):
        filepath = line_data['center'][0].strip()
        shifted_ang = 0.
    if (random_index == 2):
        filepath = line_data['right'][0].strip()
        shifted_ang = -CORRECTION_ANGLE

    #Scale Steering Angle back up
    mean, std = features['steering']
    steering_angle = float(line_data['steering'][0])*std + mean + shifted_ang

    full_path = os.path.join(DATASET_DIRECTORY, filepath.strip())
    if os.path.exists(full_path):
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #print('image before translate:', image.shape)
        image, steering_angle = translate_image(image, steering_angle, 100)
        #print('image before augment_brightness:', image.shape)
        image = augment_brightness_camera_images(image)
        #print('image before resize:', image.shape)
        #image = resize_image(image)
        image = np.array(image)
        #print('image after array and before random flip:', image.shape)
        image, steering_angle = randomly_flip_image(image, steering_angle)

        #image = randomly_add_shadow_effect(image)

    else:
        print('Image Path:', full_path, "does not exist")

    return image, steering_angle

def get_scaled_features(target_fields = ['steering', 'throttle', 'brake', 'speed']):
    data=pd.read_csv(os.path.join(DATASET_DIRECTORY, DRIVING_LOG_CSV))
    # Store scalings in a dictionary for converting back later
    scaled_feats = {}

    for each in target_fields:
        mean, std = data[each].mean(), data[each].std()

        scaled_feats[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
    return data, scaled_feats


import pandas as pd
def generate_augmented_training_batch(pr_threshold = 1, batch_size = 256):
    target_fields = ['steering', 'throttle', 'brake', 'speed']
    data, scaled_feats = get_scaled_features(target_fields)
    # Separate the data by features and targets
    camera_data, sensor_data = data.drop(target_fields, axis=1), data[target_fields]
    batch_images = np.zeros((batch_size, DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[2]))
    batch_measurements = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            index = np.random.randint(len(data))
            line_data = data.iloc[[index]].reset_index()
            keep_pr = 0
            while keep_pr == 0:
                image, measurement = preprocess_image(line_data, scaled_feats)
                pr_unif = np.random
                if (abs(measurement) < .1):
                    pr_val = np.random.uniform()
                    if (pr_val > pr_threshold):
                        keep_pr = 1
                else:
                    keep_pr = 1
            batch_images[i_batch] = image
            batch_measurements[i_batch] = measurement
        yield batch_images, batch_measurements

import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Model
import keras.backend as K
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import sys


#Hyperparameters
batch_size = 256
nb_epochs = 1
top_crop = 65
bottom_crop = 25

inputs = Input(shape=(DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[2]))

# 3 1x1 filters
conv_1 = Convolution2D(3, 1, 1, init='glorot_uniform',border_mode='same')(inputs)
lrelu_1 = LeakyReLU()(conv_1)


# 3 convolutional blocks
conv_2 = Convolution2D(32, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_1)
lrelu_2 = LeakyReLU()(conv_2)
conv_2 = Convolution2D(32, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_2)
lrelu_2 = LeakyReLU()(conv_2)
maxpool_1 = MaxPooling2D((2,2))(lrelu_2)
dropout_1 = Dropout(0.5)(maxpool_1)


conv_3 = Convolution2D(64, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_2)
lrelu_3 = LeakyReLU()(conv_3)
conv_3 = Convolution2D(64, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_3)
lrelu_3 = LeakyReLU()(conv_3)
maxpool_2 = MaxPooling2D((2,2))(lrelu_3)
dropout_2 = Dropout(0.5)(maxpool_2)


conv_4 = Convolution2D(128, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_3)
lrelu_4 = LeakyReLU()(conv_4)
conv_4 = Convolution2D(128, 3, 3, init='glorot_uniform',border_mode='same')(lrelu_4)
lrelu_4 = LeakyReLU()(conv_4)
maxpool_3 = MaxPooling2D((2,2))(lrelu_4)
dropout_3 = Dropout(0.5)(maxpool_3)


flatten = Flatten()(dropout_3)
fc_1 = Dense(512)(flatten)
fc_2 = Dense(64)(fc_1)
fc_3 = Dense(16)(fc_2)


predictions = Dense(1, activation='tanh')(fc_3)

model = Model(input=inputs, output=predictions)
adam = Adam(lr=0.0007, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse',
             optimizer=adam,
             metrics=['msle'])
print(model.summary())

callback1 = ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss',
                            verbose=0, save_best_only=False, mode='auto')
pr_threshold = 1
for e in range(nb_epochs):
    generator = generate_augmented_training_batch(pr_threshold, batch_size)
    #validation_generator = generate_augmented_training_batch(pr_threshold,batch_size)
    #validation_size = int(NB_AUGMENTED_SAMPLES*VALIDATION_PORTION)

#     model.fit_generator(generator, samples_per_epoch=NB_AUGMENTED_SAMPLES,
#                     nb_epoch=1, callbacks=[callback1], validation_data=validation_generator,
#                    nb_val_samples = validation_size, verbose=1)
    model.fit_generator(generator, samples_per_epoch=NB_AUGMENTED_SAMPLES,
                        nb_epoch=1, callbacks=[callback1], verbose=1)
    pr_threshold = 1/((e+1)*1.)

model.save(MODEL_DATA)
exit()
