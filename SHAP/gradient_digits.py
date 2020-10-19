import tensorflow as tf
import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
import shap
from keras.datasets import mnist
from keras import backend as K
import numpy as np

# load data and model
batch_size = 128
num_classes = 10
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

# since we have two inputs we pass a list of inputs to the explainer
explainer = shap.GradientExplainer(model, x_train)

# get 0-9 x
x_0to9 = x_test[3:4]
list_1to9 = [5,43,18,4,8,11,0,61,62]
for i in range(9):
    idx = list_1to9[i]
    x_0to9 = np.concatenate((x_0to9, x_test[idx:idx+1]), axis=0)

# we explain the model's predictions on the first three samples of the test set
# shap_values = explainer.shap_values(x_test[:3])

shap_values = explainer.shap_values(x_0to9)

# plot the feature attributions
shap.image_plot(shap_values, -x_0to9)


# since the model has 10 outputs we get a list of 10 explanations (one for each output)
# print(len(shap_values))
# since the model has 2 inputs we get a list of 2 explanations (one for each input) for each output
# print(len(shap_values[0]))

# here we plot the explanations for all classes for the first input (this is the feed forward input)
# shap.image_plot([shap_values[i][0] for i in range(10)], x_test[:3])
# shap.image_plot(shap_values, -x_test[:3])