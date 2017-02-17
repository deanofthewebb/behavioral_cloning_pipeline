# Dataset Parameters
DRIVING_LOG_CSV = 'driving_log.csv'
MODEL_DATA = 'model.h5'

# Image Augmentation
CORRECTION_ANGLE = 0.25
NB_AUGMENTED_SAMPLES = 5000
PR_THRESHOLD = 1


# Image Processing
DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_DEPTH = (64, 64, 3)
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

def resize_image(filepath):
    if os.path.exists(filepath):
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        shape = image.shape
        image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]] #Crop image to remove extraneous pixels

        if (shape[0] != DEFAULT_RESOLUTION[1] or shape[1] != DEFAULT_RESOLUTION[0]):
            image = cv2.resize(image,(DEFAULT_RESOLUTION[1],DEFAULT_RESOLUTION[0]), interpolation=cv2.INTER_AREA)
    else:
        print("File {0} does not exist! Skipping..".format(filepath))
    return image

def randomly_flip_image(image, measurement):
    if (np.random.randint(2) == 0):
        image = cv2.flip(image,1)
        measurement = -measurement
    return image, measurement

def preprocess_image(line_data):
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

    full_path = os.path.join(DATASET_DIRECTORY, filepath.strip())
    if os.path.exists(full_path):
        image = resize_image(full_path)
        steering_angle = float(line_data['steering'][0]) + shifted_ang
        image = randomly_add_shadow_effect(image)
        image, steering_angle = translate_image(image, steering_angle, 100)
        image = augment_brightness_camera_images(image)
        image, steering_angle = randomly_flip_image(image, steering_angle)
    else:
        print('Image Path:', full_path, "does not exist")

    return image, steering_angle

import pandas as pd

def generate_augmented_training_batch(pr_threshold = 1, batch_size = 256):
    data=pd.read_csv(os.path.join(DATASET_DIRECTORY, DRIVING_LOG_CSV))

    batch_images = np.zeros((batch_size, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[2]))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            index = np.random.randint(len(data))
            line_data = data.iloc[[index]].reset_index()
            keep_pr = 0
            while keep_pr == 0:
                image, measurement = preprocess_image(line_data)
                pr_unif = np.random
                if (abs(measurement) < .1):
                    pr_val = np.random.uniform()
                    if (pr_val > pr_threshold):
                        keep_pr = 1
                else:
                    keep_pr = 1

            batch_images[i_batch] = image
            batch_steering[i_batch] = measurement
        yield batch_images, batch_steering

import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#Hyperparameters
batch_size = 128
nb_epochs = 1

inputs = Input(shape=DEFAULT_RESOLUTION)
#crop = Cropping2D(cropping=((top_crop,bottom_crop), (0,0)))(inputs)
lambda_1 = Lambda(lambda x: x/127.5 - 1.)(inputs)
conv_1 = Convolution2D(16, 8, 8, init='glorot_uniform',
                             subsample=(4,4),border_mode='same')(lambda_1)
lrelu_1 = LeakyReLU()(conv_1)
conv_2 = Convolution2D(32, 5, 5, init='glorot_uniform',
                             subsample=(2,2),border_mode='same')(lrelu_1)
lrelu_2 = LeakyReLU()(conv_2)
conv_3 = Convolution2D(64, 5, 5, init='glorot_uniform',
                             subsample=(2,2),border_mode='same')(lrelu_1)
flatten = Flatten()(conv_3)
dropout_1 = Dropout(0.2)(flatten)
lrelu_3 = LeakyReLU()(dropout_1)
fc_1 = Dense(512)(lrelu_3)
dropout_2 = Dropout(0.5)(fc_1)
lrelu_4 = LeakyReLU()(dropout_2)
predictions = Dense(1, activation='tanh')(lrelu_4)

model = Model(input=inputs, output=predictions)
adam = Adam(lr=0.0007, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse',
             optimizer=adam,
             metrics=['msle'])
print(model.summary())

callback1 = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                            verbose=0, save_best_only=False, mode='auto')
for epoch in range(nb_epochs):

    generator = generate_augmented_training_batch(batch_size)
    validation_generator = generate_augmented_training_batch(batch_size)
    validation_size = int(NB_AUGMENTED_SAMPLES*VALIDATION_PORTION)

    model.fit_generator(generator, samples_per_epoch=NB_AUGMENTED_SAMPLES,
                    nb_epoch=1, callbacks=[callback1], validation_data=validation_generator,
                   nb_val_samples = validation_size, verbose=1)

model.save(MODEL_DATA)
exit()
