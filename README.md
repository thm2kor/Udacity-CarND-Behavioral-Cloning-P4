# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./images/carnd-using-multiple-cameras.png "multiple cameras"

Overview
---
The objective of this project is to clone a driving behavior of a human driver by deriving a deep neural network to model the steering angle of a car for a given driving situation, which enables a car to drive autonomously. Udacity provided a [simulator](https://github.com/udacity/self-driving-car-sim) which supports two modes:
- A **training mode** where a human can steer a car around a pre-defined track. During this phase, an image frame representing the current driving environment and the respective steering position is continuously recorded.
- An **autonomous mode** which uses a given deep neural network model to autonomously steer the vehicle.

**Note:** *Though the simulator records steering, throttle and brake positions, the current version of the model returns only the steering position. throttle position and brake requests are internally handled with the simulator*

---
## Overview of Files
| File| Description | Supported flags |
| ------ | ------ | ----- |
| [model.py](./model.py) | Module containing the functions for model creation and training | No flags |
| [drive.py](./drive.py) | Module provided by Udacity for communicating with the simulator. **File is adapted to pre-process the simulator images**. | *python drive.py 'relative_path_2_model_file'* |
| [video.py](./video.py) | Module provided by Udacity to prepare videos. **No adaptations** | *python video.py folder_path_to_images --fps = FPS* |
| [README.md](./README.md) | Writeup for this project. | |
| [./models](./models) | Folder which contain the model files which are generated after running model.py  |  |
| [./results](./results) | Folder which contain the result videos |  |
| [./logs](./logs) | Folder which contain the diagnostics logs |  |
| [./checkpoints](./checkpoints) | Folder which contain the checkpoints of the weights |  |

## Training Strategy
In the training mode, the simulator generates a set of information which represents the current driving environment. For a given time, the following information are recorded:
1. Path to a set of 3 images taken the centre camera, left and right side of the vehicle respectively
2. Steering angles
3. Throttle position
4. Brake position
![camera-positions][image1]

### Dataset
Udacity provides a [default dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) which is taken as the starting point for training the model. Depending on the performance of the model, additional training images would be collected.
