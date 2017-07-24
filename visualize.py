from keras.models import load_model
model = load_model('combined-w-side.h5')

from keras.utils import plot_model
plot_model(model, to_file='model.png')