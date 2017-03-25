
# coding: utf-8

# 
# [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# ## Dean Webb - Vehicle Detection & Tracking Pipeline
# 
# In this project, my goal will be to use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using [Keras](https://github.com/fchollet/keras). The model will output a steering angle to an autonomous vehicle.
# 
# We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.
# 
# We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.
# 
# To meet specifications, the project will require submitting five files: 
# * model.py (script used to create and train the model)
# * drive.py (script to drive the car - feel free to modify this file)
# * model.h5 (a trained Keras model)
# * a report writeup file (either markdown or pdf)
# * video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
# 
# Project Goals
# ---
# The goals / steps of this project are the following:
# * Use the simulator to collect data of good driving behavior 
# * Design, train and validate a model that predicts a steering angle from image data
# * Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
# * Summarize the results with a written report
# 
# ### Dependencies
# 
# The following resources can be found in this github repository:
# * drive.py
# * video.py
# * writeup_template.md
# 
# The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.
# 
# ## Details About Files In This Directory
# 
# ### `drive.py`
# 
# Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
# ```sh
# model.save(filepath)
# ```
# 
# Once the model has been saved, it can be used with drive.py using this command:
# 
# ```sh
# python drive.py model.h5
# ```
# 
# The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.
# 
# Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.
# 
# #### Saving a video of the autonomous agent
# 
# ```sh
# python drive.py model.h5 run1
# ```
# 
# The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.
# 
# ```sh
# ls run1
# 
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
# [2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
# ...
# ```
# 
# The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.
# 
# ### `video.py`
# 
# ```sh
# python video.py run1
# ```
# 
# Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.
# 
# Optionally one can specify the FPS (frames per second) of the video:
# 
# ```sh
# python video.py run1 --fps 48
# ```
# 
# The video will run at 48 FPS. The default FPS is 60.

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
import gzip
import urllib.request
import zipfile
import os
import shutil
import csv
import numpy as np
import math
import matplotlib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# In[7]:

# Dataset Parameters
DRIVING_LOG_CSV = 'driving_log.csv'
DATASET_DIRECTORY = 'data/'
DATASET_FILE = 'driving_data.zip'
WORKING_DIRECTORY = 'data/'
SOURCE_URL = 'https://s3-us-west-1.amazonaws.com/sdc-gpu/data.zip'

## Image Augmentation Parameters ##
CORRECTION_ANGLE = 0.25
IMAGE_RES = (160, 320, 3)
NB_AUGMENTED_SAMPLES = 20000
YCROP_STOP = IMAGE_RES[0]-25
YCROP_START = math.floor(IMAGE_RES[0]/5)
XCROP_START = 0
XCROP_STOP = IMAGE_RES[0]
DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_DEPTH = (64, 64, 3)
DEFAULT_RESOLUTION = (DEFAULT_LENGTH, DEFAULT_WIDTH, DEFAULT_DEPTH) if DEFAULT_DEPTH > 1 else (DEFAULT_LENGTH, DEFAULT_WIDTH)

# Training Parameters
DATACACHE_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'datacache/')
MODEL_DATA = 'model.h5'


# In[3]:

def maybe_download(filename):
    zipped_file = os.path.join(WORKING_DIRECTORY, DATASET_FILE)
    if not os.path.exists(os.path.join(WORKING_DIRECTORY, DATASET_DIRECTORY)):
        if not os.path.exists(WORKING_DIRECTORY):
            os.mkdir(WORKING_DIRECTORY)
        
        #Download file from S3 bucket if not found
        if not os.path.exists(os.path.join(WORKING_DIRECTORY, filename)):
            filepath = os.path.join(WORKING_DIRECTORY, DATASET_FILE)
            zipped_file, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
            statinfo = os.stat(filepath)
            print('Succesfully downloaded:', SOURCE_URL, '| % d MB.' % int(statinfo.st_size*1e-6))
            
        #Unzip Downloaded File
        unzip_file(zipped_file, os.path.join(WORKING_DIRECTORY))


# In[4]:

def unzip_file(zipped_file, destination):
    print('Extracting zipped file: ', zipped_file)
    zipf = zipfile.ZipFile(zipped_file)
    zipf.extractall(destination)
    print('Loaded and extracted zipfile',zipf)
    zipf.close()

    #Remove Zip File
    destination = os.path.join(WORKING_DIRECTORY,DATASET_FILE)
    if os.path.exists(destination):
        shutil.rmtree(destination, ignore_errors=True)


# In[5]:

if os.path.exists(WORKING_DIRECTORY):
    shutil.rmtree(WORKING_DIRECTORY, ignore_errors=True)
if os.path.exists(DATASET_DIRECTORY):
    shutil.rmtree(DATASET_DIRECTORY, ignore_errors=True)
        
maybe_download(DATASET_FILE)


# In[8]:

drive_data = pd.read_csv(os.path.join(WORKING_DIRECTORY,DATASET_DIRECTORY,DRIVING_LOG_CSV))


# In[9]:

drive_data.head()


# ### Preprocessing - Image Augmentation
# 
# Implement various [Image Augmentation](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jao9k5lb1) techniques as described by Vivek Yadav

# In[10]:

def resize_image(image):
    img = np.copy(image)
    shape = img.shape
    # Crop numpy array of image to remove extraneous pixels
    img = img[YCROP_START:YCROP_STOP, XCROP_START:XCROP_STOP]
    scaled = cv2.resize(img,(DEFAULT_LENGTH, DEFAULT_WIDTH), interpolation=cv2.INTER_AREA)    
    return scaled


# ### Dataset - Load Data
# 
# Start by importing the simulator data from the training_data directory. To avoid storing large files on github, I used an S3 bucket to store my images and auto-download when the directory does not exist.

# In[12]:

def warp_image(image,steer,trans_range):
    shape = image.shape
    # Translation
    delta_x = trans_range*np.random.uniform()-trans_range/2
    steering_angle = steer + delta_x/trans_range*2*.2
    delta_y = 40*np.random.uniform()-40/2
    # TRANSLATION MATRIX
    Trans_M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    warped_image = cv2.warpAffine(image,Trans_M,(shape[0],shape[1]))    
    return warped_image,steering_angle


# In[13]:

def randomly_add_shadow_effect(image):   
    start_y = IMAGE_RES[0]*np.random.uniform()
    start_x = 0
    stop_x = IMAGE_RES[1]
    stop_y = IMAGE_RES[0]*np.random.uniform()
    s_ch = 1
    image_hls = cv2.cvtColor(np.copy(image),cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-start_x)*(stop_y-start_y) -(stop_x - start_x)*(Y_m-start_y) >=0)] = 1
    
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2)==1:
            image_hls[:,:,s_ch][cond1] = image_hls[:,:,s_ch][cond1]*random_bright
        else:
            image_hls[:,:,s_ch][cond0] = image_hls[:,:,s_ch][cond0]*random_bright       
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


# In[14]:

def randomly_flip_image(image, measurement):
    if (np.random.randint(2) == 0):
        image = cv2.flip(image,1)
        measurement = -measurement
    return image, measurement


# In[15]:

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
            
    full_path = os.path.join(WORKING_DIRECTORY, DATASET_DIRECTORY, 'IMG',os.path.split(filepath)[-1].strip())
    if os.path.exists(full_path):
        image = cv2.imread(full_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image, steering_angle = warp_image(image, steering_angle, 100)
        image = augment_brightness_camera_images(image)
        image = randomly_add_shadow_effect(image)
        image, steering_angle = randomly_flip_image(image, steering_angle)
        image = resize_image(image)
    else:
        print('Image Path:', full_path, "does not exist")

    return image, steering_angle


# ### Scaling target variables
# To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
# 
# The scaling factors are saved so we can go backwards when we use the network for predictions.

# In[16]:

def get_scaled_features(target_fields = ['steering', 'throttle', 'brake', 'speed']):
    data=pd.read_csv(os.path.join(WORKING_DIRECTORY, DATASET_DIRECTORY, DRIVING_LOG_CSV))
    # Store scalings in a dictionary for converting back later
    scaled_feats = {}
    
    for each in target_fields:
        # Calculate the mean and std_dev of the steering angle for image augmentation
        mean, std = data[each].mean(), data[each].std()

        scaled_feats[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
    return data, scaled_feats


# In[17]:

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


# ### Train the Network - Implemented with Modified [Nvidia Architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

# In[11]:

def augment_brightness_camera_images(image):
    v_ch = 2
    img = np.copy(image)
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv_img = np.float64(np.copy(hsv_img))
    
    random_light = .5+np.random.uniform()
    v_channel = hsv_img[:,:,v_ch]
    hsv_img[:,:,v_ch] = v_channel*random_light
    
    v_channel = hsv_img[:,:,v_ch]
    hsv_img[:,:,v_ch][v_channel>255] = 255
    hsv_img = np.uint8(np.copy(hsv_img))
    
    aug_img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB)
    return img


# In[23]:

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
nb_epochs = 20

## Using modified Nvidia model ## 
inputs = Input(shape=DEFAULT_RESOLUTION)
lambda_1 = Lambda(lambda x: x/127.5 -1.)(inputs)

conv_1 = Convolution2D(24, 5, 5, subsample=(2, 2), init='glorot_uniform',border_mode='valid')(lambda_1)
lrelu_1 = LeakyReLU()(conv_1)

conv_2 = Convolution2D(36, 5, 5, subsample=(2, 2), init='glorot_uniform',border_mode='valid')(lrelu_1)
lrelu_2 = LeakyReLU()(conv_2)

conv_3 = Convolution2D(48, 5, 5, subsample=(2, 2), init='glorot_uniform',border_mode='valid')(lrelu_2)
lrelu_3 = LeakyReLU()(conv_3)

conv_4 = Convolution2D(64, 3, 3, subsample=(1, 1), init='glorot_uniform',border_mode='valid')(lrelu_3)
lrelu_4 = LeakyReLU()(conv_4)

conv_5 = Convolution2D(64, 3, 3, subsample=(1, 1), init='glorot_uniform',border_mode='valid')(lrelu_4)
lrelu_5 = LeakyReLU()(conv_5)

flatten = Flatten()(lrelu_5)
fc_1 = Dense(1164)(flatten)
lrelu_6 = LeakyReLU()(fc_1)

fc_2 = Dense(100)(lrelu_6)
lrelu_7 = LeakyReLU()(fc_2)

fc_3 = Dense(50)(lrelu_7)
lrelu_8 = LeakyReLU()(fc_3)

fc_4 = Dense(10)(lrelu_8)
lrelu_9 = LeakyReLU()(fc_4)

# Predictions
predictions = Dense(1, activation='tanh')(lrelu_9) # Add activation function to keep values within -1 and 1

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
    validation_generator = generate_augmented_training_batch(pr_threshold, batch_size)
    model.fit_generator(generator, samples_per_epoch=NB_AUGMENTED_SAMPLES, 
                        nb_epoch=1, callbacks=[callback1], verbose=1, 
                        validation_data=validation_generator, 
                        nb_val_samples=np.int(NB_AUGMENTED_SAMPLES*.2))
    pr_threshold = 1/((e+1)*1.)

model.save(MODEL_DATA)


# In[19]:

## Save parameters to picklefile for drive.py
target_fields = ['steering', 'throttle', 'brake', 'speed']
_, scaled_feats = get_scaled_features(target_fields)

override_datacache = True
os.makedirs(DATACACHE_DIRECTORY, exist_ok=True)
keras_pickle = os.path.join(DATACACHE_DIRECTORY,"keras_pickle.p")
if override_datacache or not os.path.exists(keras_pickle): 
    keras_hyperparameters = {'scaled_feats': scaled_feats,
                             'SCALED_LENGTH':DEFAULT_LENGTH,
                             'SCALED_WIDTH':DEFAULT_WIDTH,
                             'IMAGE_RES':IMAGE_RES,
                             'YCROP_STOP':YCROP_STOP,
                             'YCROP_START':YCROP_START,
                             'XCROP_STOP':XCROP_STOP,
                             'XCROP_START':XCROP_START,
                             'CORRECTION_ANGLE':CORRECTION_ANGLE
                            }
    pickle.dump(keras_hyperparameters, open(keras_pickle, "wb"))


# In[ ]:



