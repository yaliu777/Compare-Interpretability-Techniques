# %tensorflow_version 1.x

from __future__ import division, print_function

# # use tensorflow v1 instead of v2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Load the keras model
import keras
# saved_model_file = "keras2_mnist_cnn_allconv.h5"
saved_model_file = "hand_digits_model_weights.h5"
# load json and create model
json_file = open('hand_digits_model_config.json', 'r')
model_json = json_file.read()
json_file.close()
keras_model = keras.models.model_from_json(model_json)
# load weights into new model
keras_model.load_weights("hand_digits_model_weights.h5")
keras_model.summary()

# Load the data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test[:,:,:,None]

# Model conversion
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
revealcancel_model = kc.convert_model_from_saved_files(
                            h5_file=saved_model_file,
                            json_file='hand_digits_model_config.json',
                            nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)

# Sanity checks
from deeplift.util import compile_func
import numpy as np
from keras import backend as K

deeplift_model = revealcancel_model
deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
                                       deeplift_model.get_layers()[-1].get_activation_vars())
original_model_predictions = keras_model.predict(X_test, batch_size=200)
converted_model_predictions = deeplift.util.run_function_in_batches(
                                input_data_list=[X_test],
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
                    input_data_list=[X_test],
                    input_references_list=[np.zeros_like(X_test)],
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
        viz_scores(X_test[idx], axarr[0])
        viz_scores(scores[task_1][idx] - mean_scores_over_all_tasks, axarr[1])
    plt.show()

# Plot scores and result of masking on a single example
method_names = ['revealcancel']

list_1to9 = [5,43,18,4,8,11,0,61,62]
for i in range(len(list_1to9)):
    # plot_two_way_figures(5838,0,method_names)
    plot_two_way_figures(list_1to9[i], i, method_names)
