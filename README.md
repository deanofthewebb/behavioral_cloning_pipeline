[![Behavioral Cloning - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## Dean Webb - Behavioral Cloning Pipeline
#### Self-Driving Car Engineer Nanodegee - Project 3
---

In this project, my goal will be to use what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using [Keras](https://github.com/fchollet/keras). The model will output a steering angle to an autonomous vehicle.

### Project Goals
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
---
The following resources can be found in [Udacity's github repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3):
* drive.py
* video.py
* writeup_template.md

[//]: # (Image References)
[image1]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/cnn-architecture.jpg "Model Visualization"
[image2]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/preprocessing.jpg "Preprocessing Dataset Snippet"
[image3]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/image_augmentation.jpg "Image Augmentation Snippet"
[image4]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/generator_function.jpg "Generator Function"
[image5]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/Network_Parameters.jpg "Network Parameters"
[image6]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/download_dataset.jpg "Download Dataset"
[image7]: https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/epoch.jpg "Epoch Validation Results"

# <font color='red'> Rubric Points</font>

In this section, I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I address each point in my implementation.  

---
###  <font color='blue'> Files Submitted &amp; Code Quality</font>

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project submission includes the following files:
- model.py containing the script to create and train the model
- behavioral-cloning-pipeline-setup.ipynb containing the model.py script for compiling interactively
- drive.py for driving the car in autonomous mode (at 13 mph)
- drive_fast.py for driving the car in autonomous mode (at 18 mph)
- drive_faster.py for driving the car in autonomous mode at full throttle (set to 30 mph)
- model.h5 containing a trained convolution neural network
- output/output_video.mp4 showing the test results of the vehicle in the simulator
- writeup_report.md (and writeup_report.pdf) summarizing the results

---
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing, for example:
```sh
    python drive.py model.h5
```
- Additionally, I provide my `drive_fast.py` and `drive_faster.py` files are also included for testing higher throttle speed (instead of the project expectations of 9 mph)
- The additional files can be a good test against overfitting by requiring the model to train against a wider range of speeds.

---
#### 3. Submission code is usable and readable
- The `model.py` file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
- Additionally, the submission includes an IPython notebook `behavioral-cloning-pipeline-example.ipynb` for running on a Jupyter server. That file also contains the code for training and saving the convolutional neural network (with added  comments and visualizations).

---

###  <font color='blue'>Model Architecture and Training Strategy - Checklist</font>

#### 1. An appropriate model architecture has been employed
- My model comprises a convolution neural network with various filter sizes (including subsampling) and depths between 32 and 128 (*See e.g., lines* **268-322** of `model.py`). More specifically, the architecture presents a slightly modified version of the popular [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
- The model includes Leaky RELU layers to introduce nonlinearity in between the many of the convolutional and fully-connected layers (*See e.g., lines* **273, 276, 279** of `model.py`).
- The data is normalized in the model using a Keras lambda layer (*See e.g., lines* **270** of `model.py`).

---

#### 2. Attempts to reduce overfitting in the model
- The model contains a validation dataset and corresponding validation generator (*See e.g., lines* **318-319** of `model.py`) in order to reduce overfitting.
- The model also implements various preprocessing and [image augmentation](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jao9k5lb1) techniques as suggested by fellow SDC classmate, Vivek Yadav) which aims to aide in generalization of the neural network. (*See e.g., lines* **150-178** of `model.py`)
- The model was trained and validated on different data sets to ensure that the model was not overfitting. The different datasets comprise images taked from various trail runs. The model was also provided sample data from a Udacity trial run.
- The model was then tested and verified by running it through the simulator and ensuring that the vehicle could stay on the track. It successfully stays on the track *indefinitely* when driving at 9 mph (As demonstrated below)

---
#### 3. Model Parameter Tuning
- The model uses an Adam optimizer, so the learning rate was not tuned manually (*See e.g., line* **304** of `model.py`)
- The other hyper parameters where used was a `batch_size = 256` and a number of epochs `nb_epochs = 20`. These parameters collectively trained the neural network to the following results:

![alt text][image7]

---
#### 4. Appropriate training data
- Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road (*See e.g., lines* **48-85** of `model.py`)
<br/>
For more details about how I extracted and preprocessed the training data, please refer to the following section.

---


###  <font color='blue'>Model Architecture and Training Strategy - Summary</font>


#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to perform image augmentation suitable to generate reliable and robust data on the fly, based on the given datasets. The goal of these techniques where to utilize the existing dataset to further mimic predicatable driver behavior and driving conditions. Examples of scenarios that where simulated were:
- Veering close to the edge of the lane (*See* `warp_image` function at lines **107-116** of `model.py`)
- Various lighting conditions (*See* `augment_brightness_camera_images` function at lines **119-134** of `model.py`)
- Random shadow effects crossing the field of view of the camera (*See* `randomly_add_shadow_effect` function at lines **137-158** of `model.py`)
- Generalizing the direction of the steering angle along *the axis of rotation* (*See* `randomly_flip_image` function at lines **107-116** of `model.py`)
<br/>
After the collection process, I had a configurable number of data points to train with due to the generator functions. I used 5,000 samples to begin with whilw I tweaked the network and the hyperparameters. I then preprocessed this data as described above.
<br/>
I finally utilized my augmented data generator to batch out and put 20% of the number of the training samples into a validation set.
<br/>
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 20, which I found out after finalizing my model and workign through various failed implementations and roadblocks, as described below.

---

#### 2. Final Model Architecture

The final model architecture can be seen in lines **268-322** of `model.py` and was auto-generated by Keras with the following layers and layer parameter sizes:

<img src="https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/Network_Parameters.jpg" width="324" />

---

As noted above, the architecture presents a slightly modified version of the popular [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The following is a visualization of the original Nvidia model:
<img src="https://s3-us-west-1.amazonaws.com/sdc-gpu/examples/cnn-architecture.jpg" width="324" />

To finish off, the final architecture advantageouesly used an Adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image7]

---

#### 3. Creation of the Training Set & Training Process

- For training the model, the first step was to download and extract the dataset from an Amazon S3 bin and extract the dataset into the appropriate directories for training.

---
![alt text][image6]

Once the dataset was extracted, I began the preprocessing and image augmentation techniques described earlier.

---
![alt Preprocessing][image2]

---
![alt Image Augmentation Functions][image3]

#### 3. Creation of the Training Set & Training Process - Cont

* To capture good driving behavior in the simulator, I first recorded three laps (on track one) and combined it with the Udacity provided dataset, using a random selection of the three cameras in use while driving.
* I also reversed direction and drove the course in reverse. This was used to help the code generalize the steering angles irrespective of the direction on the course.
* To augment the data correctly, I used a scaled steering angle and either added or subtracted an offset value.
* I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to veer back towards the center while driving.
* To further augment the data set, I also flipped images and angles thinking that this would mimic the actual data set results of driving the course in reverse.
* I use a convolution neural network model similar to the end-to-end Nvidia architecture, but I initially started out with the commai AI model, I had many initial problems with that model, but it was likely due to other issues instead of the model's architecture itself. I thought this model might be appropriate because many SDC students had successful results training with the Nvidia model. I personally had the goal of creating and training a model that was robust enough to drive on multiple tracks.
* In order to gauge how well the model was working, I split my image and steering angle data into a training and validation sets and trained the model using a generator function:

![alt Image Augmentation Functions][image3]

---

* The final step was to run the simulator to see how well the car was driving around track one. I found that my first Comma AI would successfully train until there was a low mean squared error on the training set and on the validation set, but the resulting weights file was enormous! (over 800 MB). While I wasn't sure, this implied that I ws doing something very wrong.
* Somewhat frustrated, I then set out to discover more advanced image augmentation techniques and studied the advanced lane line tracking concepts before revisiting my model. This turned out to be a good decision because I had a much better understanding of the image augmentation techniques and quickly found my errors in my augmentation methods. For example, the biggest problem with my model before was that I wasn't scaling the image or the steering angles correctly. As a result, the car would either not drive at all, or behave erratically on the track. The issue with image scaling forced me to eliminate any resizing, which resulted in much longer training time. I was faced with a problem of not being able to continue progress, as the training time tooks simply too long to effectively test and debug.
* To overcome this obstacle, I wrote a [Docker script](https://medium.com/@deanofthewebb/dockerized-installation-of-tensorflow-1-0-from-source-with-gpu-support-77646cd25f92#.58prkfhl0) to run a container on AWS. At the time of completion, my script correctly compiled and installed Tensorflow 1.0 (with GPU) from source. However, there are currently some minor breaking changes introduced from AWS updating the Nvidia drivers for g2.8xlarge instances (from 367.XX to 375.XX). This upgrade broke support for nvidia-driver, however, the p2 instances seemed unaffected.
* Finally, after revisiting the project with GPU support a better understanding, I was able to successfully correct my errors and complete the project, the vehicle is able to drive autonomously **indefinitely** around the track without leaving the road.
