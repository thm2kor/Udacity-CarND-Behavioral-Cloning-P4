import csv
import cv2
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from math import ceil
from datetime import datetime

# global application parameters

# List of folders where data is stored
# path_append_root = [True, True, False, False, False, False] 
# default data set has relativ path, own dataset has full paths
path_append_root = [True]
# Minimal dataset for the model
path_data_folders = ['./data/']
# Additional dataset for generalising track 1 and track 2
#  path_data_folders = ['./data/', './dataset_track1_2_mixed/', './dataset_t2_stage1/', 
#                     './dataset_t2_stage2/', './dataset_t2_stage3/', './dataset_t2_stage4/'] 
## Additonal paths can also be appended via command line --data

# Flag to balance the distribution
distribution_correction = False ## Can be set via program argument --trim_straight 

# Column index in the CSV file
col_path_image_center   = 0
col_path_image_left     = 1
col_path_image_right    = 2
col_angle_steer         = 3
col_pos_throttle        = 4
col_pos_brake           = 5
col_vehicle_speed       = 6

# hyper parameters
batch_size              = 32 ## Can be set via program argument -- batch
epochs_count            = 3 ## Can be set via program argument -- epochs
learning_rate           = 0.0001 ## Can be set via program argument -- lr
angle_correction        = 0.15 # correction angle for the left and right cameras

def plot_histogram(num_bins, angles, title):
    """
    Histogram of the steering angles. train_data is the lines array
    The histogram is time-stamped and stored in the images folder
    """
    plt.title(title)
    
    hist, bins = np.histogram(angles, num_bins)
    width = (bins[1] - bins[0])*0.8
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    filename = './images/histogram_{}.png'.format(datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    
    plt.savefig(filename)
    plt.close()
    return hist, bins
    

def plot_training_stats(history_object):
    """ 
    Line graph showing the training and validation loss
    The plot is timestamped and saved in the images folder
    """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model Training Statistics')
    plt.ylabel('Mean Squared Error loss')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    filename = './images/training_stats_{}.png'.format(datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    plt.savefig(filename)
    plt.close()
    
def get_lines (path = path_data_folders):
    """
    Read the given file and return the lines as an array
    """
    lines = []
    # each line is expected be in the following format
    # line[0] - path to the center image
    # line[1] - path to the left image
    # line[2] - path to the right image
    # line[3] - float value of 'steering angle'
    # line[4] - float value of 'braking'
    # line[5] - float value of 'throttle'
    for i in range(len(path_data_folders)):        
        with open(path_data_folders[i] +'driving_log.csv') as csvfile:
            print ('Loading data from {} ...'.format(path_data_folders[i] +'driving_log.csv'))
            #skip header
            next(csvfile)
            reader =csv.reader(csvfile)
            for path_center, path_left, path_right, angle, throttle, brake, speed in reader:
                if path_append_root[i]:
                    path_center = path_data_folders[i] + path_center.strip()
                    path_left = path_data_folders[i] + path_left.strip()
                    path_right = path_data_folders[i] + path_right.strip()
                line = [ path_center.strip(), path_left.strip(), path_right.strip(), angle.strip(), throttle, brake, speed ]
           
                lines.append(line)

    return lines

def prepare_model():
    """
    prepare model based on NVIDIA paper https://arxiv.org/pdf/1604.07316.pdf
    """
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, ReLU, ELU, Lambda
    #from keras.utils.vis_utils import plot_model
    model = Sequential()
    #Normalization
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))
    # Conv Layer 1
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_initializer='he_uniform'))
    model.add(ReLU())
    # Conv Layer 2
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_initializer='he_uniform'))
    model.add(ReLU())
    # Conv Layer 3
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', kernel_initializer='he_uniform'))
    model.add(ReLU())
    # Conv Layer 4
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(ReLU())
    # Conv Layer 5
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(ReLU())
    
    model.add(Flatten())
    # dense 1164
    model.add(Dense(1164))
    model.add(ReLU())
    model.add(Dropout(0.5)) 

    model.add(Dense(100))
    model.add(ReLU())
    model.add(Dropout(0.5)) 

    model.add(Dense(50))
    model.add(ReLU())
    
    model.add(Dense(10))
    model.add(ReLU())
    model.add(Dense(1))

    #plot_model(model, to_file='.\images\model.png')
    #project sandbox does not support py.dot
    #TODO: Investigate the problem

    return model

def pre_process(image):
    """
    This function will be called by the drive.py module as well.
    
    crop image (remove the frame which has sky, trees, etc on the top
    and the bottom frame which shows the hood )
    """
    result = image[range(20, 140), :, :]
    # resize image to fit the input shape of the model
    result = cv2.resize(result, dsize=(200, 66))
    # return the result compatible to the nvidia model
    result = cv2.cvtColor(result, cv2.COLOR_RGB2YUV)
    return result.astype('float32')

def get_image(image_path):
    """
    load the image from the given path
    """
    result = cv2.imread(image_path.strip())
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def augument_images(images, angles, batch_sample):
    """
    for the line (batch sample) load the center, left
    and right images. Flip each of them.
    In total 6 images will be returned.
    """
    # add data of center camera
    image = pre_process(get_image(batch_sample[col_path_image_center]))
    angle = float(batch_sample[col_angle_steer])
    # add center image to the 'master' array
    images.append(image)
    angles.append(angle)
    # flip the images with a -ve angle
    images.append(np.fliplr(image))
    angles.append(-angle)
    
    ## add data of left camera
    image = pre_process(get_image(batch_sample[col_path_image_left]))
    angle = float(batch_sample[col_angle_steer]) + angle_correction
    # augment left image and the corresponding corrected angle to the 'master' array
    images.append(image)
    angles.append(angle)
    # flip the images with a -ve angle
    images.append(np.fliplr(image))
    angles.append(-angle)
    
    ## add data of right camera
    image = pre_process(get_image(batch_sample[col_path_image_right]))
    angle = float(batch_sample[col_angle_steer]) - angle_correction
    # augment right image and the corresponding corrected angle to the 'master' array
    images.append(image)
    angles.append(angle)
    # flip the images with a -ve angle
    images.append(np.fliplr(image))
    angles.append(-angle)
    
    return images, angles
    
def generator(samples, batch_size=32):
    """
    - coroutine for generating 'batch_size' of training samples
    - code based on class notes 'Generators' Project: Behavioral Cloining
    """
    from sklearn.utils import shuffle
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # for every lines, 3 images * 2 (flipped) = 6 images will be returned
                images, angles = augument_images (images, angles, batch_sample)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
def correct_distribution (lines):
    """
    Balance the distribution of angles
    Define an ideal value of samples per bin. If the count per bin is greated
    than the the average, then randomly remove the items only for that bin
    """
    angles = np.float32(np.array(lines)[:, 3])
    num_bins = 21
    
    hist, bins = plot_histogram( num_bins, angles, 'Histogram - before distribution correction')
    #correct the distribution
    ideal_samples = len(angles)/num_bins * 1.5
        
    keep_prob = [1 if hist[i] < ideal_samples else ideal_samples/hist[i] for i in range(num_bins) ]
    remove_list = []

    for x, y in ((i,j) for i in range(len(angles)) for j in range(num_bins)):
        if angles[x] > bins[y] and angles[x] <= bins[y+1]:
            if np.random.rand() > keep_prob[y]:
                remove_list.append(x)
    
    lines = np.delete(lines, remove_list, axis=0)  
    # check if distribution is ok  
    angles = np.float32(np.array(lines)[:, 3])
    hist = plot_histogram(num_bins , angles, 'Histogram - after distribution correction')
    
    return lines
    
def main():
    """
    main routine
    """
    print('Training with Batch size of {:d}, Epoch count of {:d}, Learning rate of {:.2e} and Distribution correction set to {}' .format(batch_size, epochs_count, learning_rate, distribution_correction ))
    
    lines = get_lines()
    print('Line count {:d}'.format(len(lines)))
     
    if distribution_correction:
        lines = correct_distribution ( lines )
        print('Lines after distribution correction {:d}'.format(len(lines)))

    # prepare the training and validation samples
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    print('Training samples after splitting {:d}'.format(len(train_samples)))
    print('Training samples after splitting {:d}'.format(len(validation_samples)))
    
    # call generator functions
    # compile and train the model using the generator functions
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    ## Prepare the model
    from keras.optimizers import Adam
    model = prepare_model()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    # callback to save the accuracy and checkpoints data in a log file
    # TODO: Try Tensorboard to show the statistics
    from keras.callbacks import ModelCheckpoint, CSVLogger
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    filename = './logs/logs_{:d}_epochs_{:d}.csv'.format(batch_size, epochs_count)
    logger = CSVLogger(filename)

    ## fit the model
    history_object = model.fit_generator(train_generator, steps_per_epoch=6*ceil(len(train_samples)/batch_size),
            validation_data=validation_generator, validation_steps=6*ceil(len(validation_samples)/batch_size),
            epochs=epochs_count, verbose=1, callbacks=[checkpointer, logger])

    ## print the summary for documentation
    model.summary()

    ## plot the training and validation loss for each epoch
    plot_training_stats(history_object)
        
    #Save the model
    filename = './models/model_{}.h5'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.save(filename)
    print ('Model created at :' + filename)
    
    ############## Validation steps . Only for debugging ###############
    # for line in lines:
    #    image = pre_process(get_image(line[0]))
    #    steering_angle_pred = float(model.predict(image[None, :, :, :], batch_size=1))
    #    print (steering_angle_pred) 
    ############## Validation steps . Only for debugging ###############
    
    # End tensorflow session
    from keras import backend as K 
    K.clear_session()
    
if __name__ == '__main__':
    # Read the program arguments and prepare the global parameters
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--batch', type=int, default=32, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--trim_straight', dest='trim_straight', action='store_true', help='Skip distribution correction')
    parser.add_argument('--data', dest='data_paths', action='append', help='Path to the data files')

    parser.set_defaults(trim_straight=False)
    args = parser.parse_args()

    batch_size = args.batch
    epochs_count = args.epochs
    learning_rate = args.lr
    distribution_correction = args.trim_straight
    if args.data_paths:
        for path in args.data_paths:
            path_append_root.append (False) # Latest simulator version have full paths
            path_data_folders.append (path)

    # Call main function
    main()
