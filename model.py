import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    #ignoring the header row
    next(reader)
    for line in reader:
        lines.append(line)
        
print ('Number of Images {:d}'.format(len(lines)))
print (lines[1])
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement =float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print ('Length of X_train {}'.format(X_train.shape)) 
print ('Length of y_train {}'.format(y_train.shape)) 
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=6)
model.save('model.h5')
print ('Model saved ...')