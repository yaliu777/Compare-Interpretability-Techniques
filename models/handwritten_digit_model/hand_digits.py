import tensorflow as tf
import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
import shap
from keras.datasets import mnist
from keras import backend as K
import numpy as np
# from __future__ import division, print_function


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


# Model conversion
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
revealcancel_model = kc.convert_model_from_saved_files(
                            h5_file="hand_digits_model_weights.h5",
                            json_file='hand_digits_model_config.json',
                            nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)

# Sanity checks
from deeplift.util import compile_func
import numpy as np
from keras import backend as K

deeplift_model = revealcancel_model
deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
                                       deeplift_model.get_layers()[-1].get_activation_vars())
original_model_predictions = model.predict(x_test, batch_size=200)
converted_model_predictions = deeplift.util.run_function_in_batches(
                                input_data_list=[x_test],
                                func=deeplift_prediction_func,
                                batch_size=200,
                                progress_update=None)
print("difference in predictions:",np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)))
assert np.max(np.array(converted_model_predictions)-np.array(original_model_predictions)) < 10**-5
predictions = converted_model_predictions

# Compile various scoring functions
from keras import backend as K
import deeplift
from deeplift.util import get_integrated_gradients_function

revealcancel_func = revealcancel_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)

# Call scoring functions on the data
from collections import OrderedDict
method_to_task_to_scores = OrderedDict()
for method_name, score_func in [('revealcancel', revealcancel_func)]:
    method_to_task_to_scores[method_name] = {}
    for task_idx in range(10):
        print("\tComputing scores for task: "+str(task_idx))
        scores = np.array(score_func(
                    task_idx=task_idx,
                    input_data_list=[x_test],
                    input_references_list=[np.zeros_like(x_test)],
                    batch_size=1000,
                    progress_update=None))
        method_to_task_to_scores[method_name][task_idx] = scores

# Prepare various functions to aid in plotting
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from keras import backend as K


#Function to plot scores of an MNIST figure
def viz_scores(scores,ax):
    reshaped_scores = scores.reshape(28,28)
    the_min = np.min(reshaped_scores)
    the_max = np.max(reshaped_scores)
    center = 0.0
    negative_vals = (reshaped_scores < 0.0)*reshaped_scores/(the_min + 10**-7)
    positive_vals = (reshaped_scores > 0.0)*reshaped_scores/float(the_max)
    reshaped_scores = -negative_vals + positive_vals
    ax.imshow(-reshaped_scores, cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])


#Function to plot the result of masking on a single example, for converting
#from task1->task2 and task1->task3
def plot_two_way_figures(idx, task_1, method_names):
    print("example index: "+str(idx))
    print("Order of columns is:","task "+str(task_1)+" scores")
    print("Order of the methods is: "+", ".join(str(x) for x in method_names))
    for method_name in method_names:
        scores = method_to_task_to_scores[method_name]
        mean_scores_over_all_tasks = np.mean(np.array([scores[i][idx] for i in range(10)]), axis=0)
        f, axarr = plt.subplots(1, 2, sharey=False, figsize=(15,10))
        viz_scores(x_test[idx], axarr[0])
        viz_scores(scores[task_1][idx] - mean_scores_over_all_tasks, axarr[1])
    plt.show()

# Plot scores and result of masking on a single example
method_names = ['revealcancel']
list_1to9 = [5,43,18,4,8,11,0,61,62]
for i in range(len(list_1to9)):
    # plot_two_way_figures(5838,0,method_names)
    plot_two_way_figures(list_1to9[i], i, method_names)

