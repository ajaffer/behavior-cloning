from keras.models import Model
import matplotlib.pyplot as plt

history = {'val_loss': [0.057678838493302464, 0.050279253240053855, 0.045697620914628111], 'loss': [0.11398382629565101, 0.04890348554212362, 0.052330473279382318]}

### print the keys contained in the history object
print(history.keys())

### plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()