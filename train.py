import csv
import cv2
import numpy as np

def get_image(source_path):
   filename = source_path.split('/')[-1]
   current_path = '../data/IMG/' + filename
   #print("readin path {}".format(current_path))
   return cv2.imread(current_path)

lines = []
with open('../data/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
      lines.append(line)

car_images = []
steering_angles = []
for line in lines:
   steering_center = float(line[3])
   correction = 0.2
   steering_left = steering_center + correction
   steering_right = steering_center - correction
   
   img_center = get_image(line[0])
   img_left = get_image(line[1])
   img_right = get_image(line[2])

   #car_images.append(img_center)
   #steering_angles.append(steering_center)

   car_images.extend([img_center, img_left, img_right])
   steering_angles.extend([steering_center, steering_left, steering_right])

augmented_car_images, augmented_steering_angles = [], []
for car_image, steering_angle in zip(car_images, steering_angles):
   augmented_car_images.append(car_image)
   augmented_steering_angles.append(steering_angle)
   augmented_car_images.append(cv2.flip(car_image,1))
   augmented_steering_angles.append(steering_angle*-1.0)


X_train = np.array(augmented_car_images)
y_train = np.array(augmented_steering_angles)

#X_train = np.array(car_images)
#y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))


#model.add(Conv2D(24, 5, 2, input_shape=(160, 320, 3)))
##model.add(MaxPooling2D((2, 2)))
##model.add(Dropout(0.5))

#model.add(Conv2D(36, 5, 2))
#model.add(Conv2D(48, 5, 2))
#model.add(Conv2D(64, 3, 1))
#model.add(Conv2D(64, 3, 1))


##model.add(Conv2D(filters=24,kernel_size=5,strides=2, padding='same'))
##model.add(Conv2D(filters=36,kernel_size=5,strides=2, padding='same'))
##model.add(Conv2D(filters=48,kernel_size=5,strides=2, padding='same'))
##model.add(Conv2D(filters=64,kernel_size=3,strides=1, padding='same'))
##model.add(Conv2D(filters=64,kernel_size=3,strides=1, padding='same'))
##model.add(Flatten(input_shape=(160,320,3)))
##model.add(Flatten(input_shape=(31,71,64)))

#model.add(Activation('relu'))

#model.add(Flatten())

#model.add(Dense(100))
#model.add(Activation('relu'))

#model.add(Dense(50))
#model.add(Activation('relu'))

#model.add(Dense(10))
#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam')
##model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)



#model = Sequential()
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
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

## preprocess data
##X_normalized = np.array(X_train / 255.0 - 0.5 )

#from sklearn.preprocessing import LabelBinarizer
#label_binarizer = LabelBinarizer()
#y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'mse')
history = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2)





model.save('model.h5')
