# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![Project Graphic](http://img.youtube.com/vi/9sm-gIwpDpQ/0.jpg)](http://www.youtube.com/watch?v=9sm-gIwpDpQ "Behavioral Cloning")

[//]: # (Image References)

[image1]: ./images/histogram_20210203-132347-112087.png "steering angle distribution from default dataset"
[image2]: ./images/histogram_20210203-132347-721190.png "steering angle distribution after distribution correction"
[image3]: ./images/carnd-using-multiple-cameras.png "multiple cameras"
[image4]: ./images/training_stats_20210203-143847-243593.png "accuracy"
[image5]: ./images/sample_augmented_image.png "image augmentation"
[image6]: ./images/sample_preprocessing.png "image preprocessing"
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
- Udacity provided a [default dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) which is taken as the starting point for training the model. It contained **8036** lines of data covering all possible driving scenarios on the Track-1.
- Based on the project instructions and an interesting [discussion note](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Behavioral+Cloning+Cheatsheet+-+CarND.pdf) from Pauly Heraty, [NVIDIA model](https://arxiv.org/pdf/1604.07316.pdf) was chosen as the starting point for the network model.
- Without dropout layers and a slightly adapted version of NVIDIA model (a normalization layer, 5 convolutional layers, and 4 fully-connected layers), Track-1 was successfully driven. To ensure that the model is generalized well enough, it is expected that the car should negotiate at least some part of Track-2. But the car could not negotiate even the first ramp. It was clear that the network was not trained for the new features like new landscape, varying slope, and shorter radius of curvature of Track-2.
- New training data was derived by running the simulator in training mode. Instead of large training datasets covering the entire track, shorter tracks were trained and added to the default dataset. Since it was advised not to store the data in the project workspace, the additional data is archived [here](https://github.com/thm2kor/Udacity-CarND-Behavioral-Cloning-P4/tree/main/training).
- Based on the actual performance in the autonomous mode, the network had to be expanded with an additional fully connected layer, 2 dropout layers and a shorter learning rate.
- The Track-1 was successfully run, but due to the fact that the Track-2 was too complicated to train with my keyboard, I did not completely train the Track-2. But my method of training in stages is a scalable solution.
- As summary, the amount and diversity of training data was the most influencing factor when it comes to generalization of the network. Tuning of model parameters had little impact on the overall performance, but it is all about data.

## Data Analysis
Since the data was generated automatically from a simulator, data sanity checks were kept at the minimum. With Microsoft Excel, the following offline checks were done:
1. Blank image paths - Result: No blank paths
2. Blank values for steering, throttle, brake and speed - Result: No blank values.

### Data Distribution
In the next step, an explorative study of steering angle, which is the key prediction value, was performed. A distribution of the steering angles of Track-1 is shown below:
![steering angle distribution before correction][image1]
A quick look at the plot shows that the steering angles corresponding to straight line driving is over represented. This could potentially bias my model towards steering angles around 0. I had two options to solve this problem:
1. Correct the distribution by randomly removing the data which are over-represented.
2. Augment the data with additional scenarios which would increase the count of under-represented data.

#### Distribution correction
- The steering angles were allocated to 21 uniformly divided bins.
- An ideal count of samples for each bin is defined. This had to be defined based on trial and error.
- Based on the degree of deviation from the ideal value, a probability factor for removing data from each bin is derived.
- The data from each bin is randomly removed based on the above calculated probability.
- The distribution of steering angles after distribution correction is shown below
![steering angle distribution after augmentation][image2]

*The performance after distribution correction did not show any performance improvement for a wide range of ideal count of bins.* Therefore, this feature was **disabled**. But, Performance improvement was seen after artificially increasing the data count using data augmentation.

#### Augmentation
The data provided by Udacity contains only steering angles for the center image. In order to effectively use the left and right images during training, an **offset of 0.15 is added to the left camera images and subtracted 0.15 from the right camera images** was added. The offset of 0.15 was derived based on trial-and-error method. With higher offset values (0.2 to 0.4), though the performance was good around the curves, the car wobbles more during straight line drive.
In addition, the images are flipped about their vertical axis with a negative steering angle. This would reduce the bias of left turns or right turns.
```python
# flip the images with a -ve angle
images.append(np.fliplr(image))
angles.append(-angle)
```
A set of augmented images for a sample image is shown below:

![augmented_images][image5]

### Pre-processing
The following steps are performed on the images to efficiently train the model.
1. The top portion of the images (approx. 140 pixels high) does not have any features which are necessary for the estimating the steering angle. Similarly the bottom frames (approx. 20 pixels high) of all the images capture the car hood, which again is not relevant for steering angle calculation. Both these portion of the images were cropped.
2. NVIDIA model expects the input shape of the image to be 3x66x200. The images are resized to the target shape.
3. Similar to the NVIDIA network architecture, the input image is converted to YUV color space.

```python
def pre_process(image):
    # crop the unwanted portions of the image
    result = image[range(20, 140), :, :]
    # resize image to fit the input shape of the model
    result = cv2.resize(result, dsize=(200, 66))
    # return the result compatible to the nvidia model
    result = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    return result.astype('float32')
```

The following set of images shows the effect of pre-processing on the images before they are fed to the model.
![preprocessed_images][image6]

### Network Architecture

#### Normalization
For a faster convergence of a neural network, the images needs to be normalized. The images are normalized in the first layer of the model where the RGB values are scaled to be between -1.0 and 1.0.
```python
model = Sequential()
#Normalization
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))
```
#### Model
The network starts with a normalization layer followed by 5 convolutional layers. The first convolutional layer accepts an input of 3x66x200, has a filter of size 5x5 and stride of 2x2, resulting in output of 24x31x98. The second convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in and output of 36x14x47. The third convolutional layer then applies a filter of size 5x5 with stride of 2x2, resulting in output of 48x5x22. The fourth convolutional layer applies a filter of 3x3 with stride of 1x1 (no stride), resulting in output of 64x3x20. The fifth and final convolutional layer then applies a filter of 3x3 with no stride, resulting in output of 64x1x18. The output is then flattened to produce a layer of 1152 neurons. The first fully-connected layer then results in output of 1164 neurons, followed by 100 neurons, 50 neurons, 10 neurons, and finally produces an output representing the steering angle. Two dropout layers with 50% probability is added after the first and second fully connected layers to avoid overfitting of the model.

The model can be summarized as below:

| Layer                   | Filter | Stride  | Output Shape  | Parameters   |
| :-------------          | :----- | :-----  | :----------   |   ---------: |
| Normalization (Lambda)  |        |         | (66, 200, 3)  |          0   |
| 1st Convolutional/ReLU  | (5,5)  | (2,2)   | (31, 98, 24)  |      1,824   |
| 2nd Convolutional/ReLU  | (5,5)  | (2,2)   | (14, 47, 36)  |     21,636   |
| 3rd Convolutional/ReLU  | (5,5)  | (2,2)   | (5, 22, 48)   |     43,248   |
| 4th Convolutional/ReLU  | (3,3)  | (1,1)   | (3, 20, 64)   |     27,712   |
| 5th Convolutional/ReLU  | (3,3)  | (1,1)   | (1, 18, 64)   |     36,928   |
| Flatten                 |        |         | (1152)        |          0   |
| 1st Fully Connected/ReLU|        |         | (1164)        |  1,342,092   |
| Dropout                 |        |         | (1164)        |          0   |
| 2nd Fully Connected/ReLU|        |         | (100)         |    116,500   |
| Dropout                 |        |         | (100)         |          0   |
| 3rd Fully Connected/ReLU|        |         | (50)          |       5050   |
| 4th Fully Connected/ReLU|        |         | (10)          |        510   |
| Final Fully Connected   |        |         | (1)           |         11   |
| **Total Parameters**    |        |         |               |  1,595,511   |

ReLU is used as the activation function to improve the non-linearity of the model. The weights of the network are trained to minimize the *mean squared error* between the predicted angle output and the steering angles from the dataset.

 Additional model parameters are summarized in the below table:

| Parameter               | Value             | Description                              |
|:---                     |:---               |:---                                      |
| Weights Initialization  | he_uniform        | Most recommended in the discussion forums|
| Optimizer               | Adam              | Most recommended in the discussion forums|
| Learning Rate           | 0.0001            | No adaptive learning rate                |
| Batch size              | 32                |                                          |
| Epochs                  | 5                 | Accuracy was stable within 3 EPOCH       |

The accuracy statistics of the network.

![accuracy_statistics][image4]
