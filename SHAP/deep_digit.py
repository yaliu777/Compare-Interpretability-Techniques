from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# load json and create model
json_file = open('hand_digits_model_config.json', 'r')
model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("hand_digits_model_weights.h5")

# evaluate loaded model on test data
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


import shap
import numpy as np

# select a set of background examples to take an expectation over
# randomly select 100 images from train dataset as background
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

# explain predictions of the model on three images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)

# get 0-9 x
x_0to9 = x_test[3:4]
list_1to9 = [5,43,18,4,8,11,0,61,62]
for i in range(9):
    idx = list_1to9[i]
    x_0to9 = np.concatenate((x_0to9, x_test[idx:idx+1]), axis=0)

# 0 -> y_test[3]
# 1 -> y_test[5]
# 2 -> y_test[43]
# 3 -> y_test[18]
# 4 -> y_test[4]
# 5 -> y_test[8]
# 6 -> y_test[11]
# 7 -> y_test[0]
# 8 -> y_test[61]
# 9 -> y_test[62]

shap_values = e.shap_values(x_0to9)

# plot the feature attributions
shap.image_plot(shap_values, -x_0to9)