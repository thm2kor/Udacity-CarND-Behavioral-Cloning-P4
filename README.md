# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![Project Graphic](http://img.youtube.com/vi/9sm-gIwpDpQ/0.jpg)](http://www.youtube.com/watch?v=9sm-gIwpDpQ "Behavioral Cloning")

[//]: # (Image References)

[image1]: ./images/histogram_20210203-132347-112087.png "steering angle distribution from default dataset"
[image2]: ./images/histogram_20210203-132347-721190.png "steering angle distribution after distribution correction"
[image3]: ./images/carnd-using-multiple-cameras.png "multiple cameras"
Overview
---
The objective of this project is to clone a driving behavior of a human driver by training a deep neural network to model the steering angle of a car for a given driving situation, which enables a car to drive autonomously. Udacity provided a [simulator](https://github.com/udacity/self-driving-car-sim) which supports two modes:
- A **training mode** where a human can steer a car around a pre-defined track. During this phase, an image frame representing the current driving environment and the respective steering position is continuously recorded.
- An **autonomous mode** which uses a given deep neural network model to autonomously steer the vehicle.

**Note:** *Though the simulator records steering, throttle and brake positions, the current version of the model returns only the steering position. throttle position and brake requests are internally handled with the simulator*

---
## Overview of Files
| File| Description | Supported flags |
| ------ | ------ | ----- |
| [model.py](./model.py) | Module containing the functional code for model creation and training | --batch BATCH --epochs EPOCHS --lr LR --trim_straight --data DATA_PATHS |
| [drive.py](./drive.py) | Module provided by Udacity for driving the car in autonomous mode. **File is adapted to pre-process the simulator images**. | *python drive.py 'relative_path_2_model_file'* |
| [video.py](./video.py) | Module provided by Udacity to prepare videos. **No adaptations** | *python video.py folder_path_to_images --fps = FPS* |
| [README.md](./README.md) | Writeup for this project. | |
| [./models](./models) | Folder which contain the trained convolution neural network  |  |
| [./results](./results) | Folder which contain the result videos/gifs |  |
| [./logs](./logs) | Folder which contain the diagnostics logs |  |

### Model creation
The [model.py](./model.py) file contains the code for training and saving the convolutional neural network. The software pipeline in this file are organized as follows:
1. Prepare and load the dataset
The array `path_data_folders` holds the relative paths of all the datasets used in the project.  The default dataset is `./data/`.
```python
# Minimal dataset for the model
path_data_folders = ['./data/']
```
To generalize the network for additional tracks, new training data was generated using the simulator which were then stored in separate folders. As recommended in the project instructions, I collected training data of the tracks in stages. I appended the new folders to the above array.
```python
path_data_folders = ['./data/', './dataset_track1_2_mixed/', './dataset_t2_stage1/', \
'./dataset_t2_stage2/', './dataset_t2_stage3/', './dataset_t2_stage4/']
```
Alternatively the datasets can be added as a command line argument.
```sh
python model.py --PATH './dataset_t2_stage3/' --PATH './dataset_track1_2_mixed/'
```
2. At this stage, data is still *lean*. Each data is a row from a CSV file with 7 columns containing three file paths to images (from center, left and right cameras), the actual vehicle motion parameters like steering angle, throttle position, brake position and the speed of the vehicle.

3. The data lines are split into training and validation sets with a ratio of 80:20
```python
# prepare the training and validation samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
```   

4. In the next stage, memory intensive loading of images takes place. To efficiently handle the loading of large set of images, pre-processing and image augmentation, `generator functions` are used to load and return a required `batch_size` of images. Generator functions allows the function to behaves like an iterator with a lower memory usage.
```python
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
```   

5. Instantiate a CNN Model (more details later), configure the model and train the model with the data.
```python
## Prepare the model
from keras.optimizers import Adam
model = prepare_model()
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
## fit the model
history_object = model.fit_generator(train_generator, steps_per_epoch=6*ceil(len(train_samples)/batch_size),
        validation_data=validation_generator, validation_steps=6*ceil(len(validation_samples)/batch_size),
        epochs=epochs_count, verbose=1, callbacks=[checkpointer, logger])
```

6. After the training is complete, the model is saved in the TensorFlow SavedModel (.h5) format. The function will serializes the CNN architecture(layers and their connections), the weights values , the optimizer and the metrices.
```python
#Save the model
filename = './models/model_{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
model.save(filename)
print ('Model created at :' + filename)
```

### Running the Simulator
Using the Udacity provided simulator and the [drive.py](./drive.py) file provided by Udacity, the car can be driven autonomously around the track by executing :
```sh
python drive.py models/model.h5
```
The `drive.py` module was adapted to include the pre-processing of images and to set the speed of the vehicle. This ensures that the simulator and the model uses the same data format.
```python
from model import pre_process
...
set_speed = 20 # Set this value to 10 for Track 2
...
# Preprocess the image as defined in model.py
image_array = pre_process(image_array)
```

## Training Strategy
- Udacity provided a [default dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) which is taken as the starting point for training the model. It contained **8036** lines of data covering all possible driving scenarios on the 1st track.
- A distribution of the steering angles on this track is shown below:
![steering angle distribution before augmentation][image1]
- A quick look at the plot shows that the steering angles corresponding to straight line driving is over represented. This could potentially bias my model towards steering angles around 0. I had two options to solve this problem:
  - correct the distribution by randomly removing the data which are over-represented.
  - augment the data with additional scenarios which would increase the count of under-represented data.

### Distribution corrections

![steering angle distribution after augmentation][image2]

### Augmentation
The data provided by Udacity contains only steering angles for the center image, so in order to effectively use the left and right images during training, I added an offset of .15 to the left images and subtracted .15 from the right images. This is because an angle of 0 corresponds with the car going straight, left turns are negative, and right turns are positive.
![camera-positions][image3]
 I artificially increased my dataset using a couple of proven image augmentation techniques. One method I used was randomly adjusting the brightness of the images. This is done by converting the image to HSV color space, scaling up or down the V channel by a random factor, and converting the image back to RGB. Another technique I used was flipping the image about the vertical axis and negating the steering angle. The idea here is to attempt to get an equal number of left and right steering angles in the training data to reduce any possible bias of left turns vs right turns or vice versa.
- As recommended in the project, an adapted [NVIDIA model](https://arxiv.org/abs/1604.07316) is used as
In the training mode, the simulator generates a set of information which represents the current driving environment. For a given time, the following information are recorded:
1. Path to a set of 3 images taken the centre camera, left and right side of the vehicle respectively
2. Steering angles
3. Throttle position
4. Brake position
As seen in the below picture, the steering angles had to adapted incase the left and right camera images are used for augmentation.

Data normalization

Carried out in the first layer of the Keras model and simply scales the RGB values to be between -1.0 and 1.0.

### Network Architecture
The network consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully-connected layers. The first layer accepts an RGB image of size 66x200x3 and performs image normalization, resulting in features ranging from values -1.0 to 1.0. The first convolutional layer accepts input of 3x66x200, has a filter of size 5x5 and stride of 2x2, resulting in output of 24x31x98. The second convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in and output of 36x14x47. The third convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in output of 48x5x22. The fourth convolutional layer applies a filter of 3x3 with stride of 1x1 (no stride), resulting in output of 64x3x20. The fifth and final convolutional layer then applies a filter of 3x3 with no stride, resulting in output of 64x1x18. The output is then flattened to produce a layer of 1164 neurons. The first fully-connected layer then results in output of 100 neurons, followed by 50 neurons, 10 neurons, and finally produces an output representing the steering angle. A detailed image and summary of the network can be found below. I use dropout after each layer with drop probabilities ranging from 0.1 after the first convolutional layer to 0.5 after the final fully-connected layer. In addition, I use l2 weight regularization of 0.001. The activation function used is the exponential linear unit (ELU), and an adaptive learning rate is used via the Adam optimizer. The weights of the network are trained to minimize the mean squared error between the steering command output by the network and the steering angles of the images from the sample dataset. This architecture contains about 27 million connections and 252 thousand parameters.
