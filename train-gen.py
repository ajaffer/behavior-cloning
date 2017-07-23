import os
import csv

DATA_FOLDER = '../data/'

samples = []
with open(DATA_FOLDER + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def get_image(source_path):
   filename = source_path.split('/')[-1]
   path = DATA_FOLDER  + filename
   return cv2.imread(path)

def shuffle(array):
    import random
    return random.shuffle(array)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                center_image = get_image(batch_sample[0])
                left_image = get_image(batch_sample[1])
                right_image = get_image(batch_sample[2])
                center_angle = float(batch_sample[3])
                throttle = float(batch_sample[4])
                brake = float(batch_sample[5])
                speed = float(batch_sample[6])
                correction = 0.2
                left_angle = steering_center + correction
                right_angle = steering_center - correction

                images.extend([center_image, left_image, right_image])
                measurements.append([center_angle, throttle, brake, speed])
                measurements.append([left_angle, throttle, brake, speed])
                measurements.append([right_angle, throttle, brake, speed])

            #augment images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append([measurement[0] * - 1.0, measurement[1], measurement[2], measurement[3]])

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# ch, row, col = 3, 80, 320  # Trimmed image format
ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
        
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))??
model.add(Convolution2D(36, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(38, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)