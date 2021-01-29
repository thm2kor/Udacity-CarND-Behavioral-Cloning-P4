import csv
import cv2
import numpy as np
import math
import random

#global parameters
path_data_folder = './data/' ## <-- uses the old data in the opt folder
                             ##TODO Look for a better folder.
batch_size = 16

def get_lines ( path = path_data_folder ):
    # Read the given file and return the lines as an array
    lines = []
    # each line is expected be in the following format
    # line[0] - path to the center image
    # line[1] - path to the center image
    # line[2] - path to the right image
    # line[3] - float value of 'steering'
    # line[4] - float value of 'braking'
    # line[5] - float value of 'throttle'
    #open csv database
    with open(path_data_folder +'driving_log.csv') as csvfile:
        #skip header
        next(csvfile)
        reader =csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def prepare_model():
    #prepare model based on NVIDIA paper https://arxiv.org/pdf/1604.07316.pdf
    #TODO : Dropout layers ??
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda
    

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation = 'relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation = 'relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


    return model

def pre_process(image):
    # crop image (remove the frame which has sky, trees, etc
    # and the bottom frame which shows the hood )
    result = image[range(20, 140), :, :]
    # resize image
    result = cv2.resize(result, dsize=(200, 66))
    # return the result compatible to the nvidia model
    return result.astype('float32')

def generator(samples, batch_size=32):
    from sklearn.utils import shuffle
    # 'coroutine for generating 'batch_size' of training samples
    # code as provided in the class notes 'Generators' Project: Behavioral Cloining
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # randomly select a camera position (left , center or right )
                # load the respective picture and adjust the steering angle
                # only for left and right. Center camera angle will not be adjusted.
                angle_correction = 0.2
                camera_pos = random.randint(1,3)
                if camera_pos == 1: # augment the steering angle as if the image is from the left camera
                    name = path_data_folder + 'IMG/' + batch_sample[1].split('/')[-1]
                    image = pre_process (cv2.imread(name))
                    angle = float(batch_sample[3]) + angle_correction
                elif camera_pos == 2: # No augmentation of steering angle
                    name = path_data_folder + 'IMG/' + batch_sample[0].split('/')[-1]
                    image = pre_process (cv2.imread(name))
                    angle = float(batch_sample[3])
                elif camera_pos == 3: # augment the steering angle as if the image is from the right camera
                    name = path_data_folder + 'IMG/' + batch_sample[2].split('/')[-1]
                    image = pre_process (cv2.imread(name))
                    angle = float(batch_sample[3]) - angle_correction
                
                ## additional data augmentation
                # mirror images with chance=0.5
                #if random.choice([True, False]):
                #    image = image[:, ::-1, :]
                #    angle *= -1.

                # perturb slightly steering direction
                #angle += np.random.normal(loc=0, scale=0.1)
                
                images.append(image)
                angles.append(angle)

                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def main():
    lines = get_lines()

    # images , measurements = load_data (lines )
    ## Assert if the lines are properly parsed and loaded
    # assert (len(lines) == len(images))
    # assert (len(lines) == len(measurements))
    ## Prepare the training data and the respective labels
    # X_train = np.array(images)
    # y_train = np.array(measurements)

    from keras.callbacks import ModelCheckpoint, CSVLogger
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    ## Prepare the model
    model = prepare_model()
    model.compile(loss='mse', optimizer='adam')

    # callback to save the accuracy data in a log file
    logger = CSVLogger(filename='logs/logs.csv')

    # fit the model
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=8, verbose=1, callbacks=[logger])

    model.summary()
    #Save the model
    model.save('model.h5')

if __name__ == '__main__':
    main()
