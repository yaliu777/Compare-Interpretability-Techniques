from __future__ import division, print_function
from tqdm import tqdm
from keras.preprocessing import image
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential, load_model, Model
from keras.optimizers import SGD, Adam
import time

input_tensor = Input(shape=(224,224,3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
# for layer in base_model.layers[:15]:
for layer in base_model.layers:
  layer.trainable = False
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(5, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
# compile model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the keras model
import keras
# saved_model_file = "keras2_mnist_cnn_allconv.h5"
saved_model_file = "fashion_5class_vgg_weights.h5"
model_json_file = 'fashion_5class_vgg_seq_config.json'
# load json and create model
json_file = open(model_json_file, 'r')
model_json = json_file.read()
json_file.close()
# keras_model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights(saved_model_file)

# Load the data
DATA_PATH = '/home/ubuntu/explainML_deeplift/fashion_data/'
IMAGE_PATH = DATA_PATH + 'images/'
IMAGE_SIZE = 224
LIMIT_IMAGES = 300

def load_images(names, articletype):
    image_array = []
    for image_name in tqdm(names, desc = 'reading images for ' + articletype):
        img_path = IMAGE_PATH + image_name
        try:
            img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        except:
            continue
        img = image.img_to_array(img)
        image_array.append(img)
    return np.array(image_array)

dfstyles = pd.read_csv('/home/ubuntu/explainML_deeplift/fashion_data/styles.csv', error_bad_lines=False, warn_bad_lines=False)
dfstyles['image'] = dfstyles['id'].map(lambda x: str(x) + '.jpg')
dfstyles.columns = dfstyles.columns.str.lower()
print(dfstyles.shape)
dfstyles.head()

dfstyles['articletype'].nunique() # 143 classes
dfstyles['articletype'].value_counts().head()
dfstyles['articletype'].value_counts().tail()

dfstyles['cntarticle'] = dfstyles.groupby('articletype')['id'].transform('count')
dfdata = dfstyles[dfstyles['cntarticle'] > 500]
print(dfdata.shape, dfdata['articletype'].nunique())
dfarticles =dfdata.groupby('articletype',as_index=False)['id'].count()

image_list = []
article_list = []
for index, grouprow in dfarticles.iterrows():
    if index > 4:
        continue
    image_names = dfdata[dfdata['articletype'] == grouprow['articletype']]['image'].values
    if len(image_names) > LIMIT_IMAGES:
        image_names = image_names[:LIMIT_IMAGES]
    image_list.extend(load_images(image_names, grouprow['articletype']))
    article_list.extend(len(image_names) * [grouprow['articletype']])

X = np.array(image_list) / 255.0
X = X.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
y_encoded = LabelEncoder().fit_transform(article_list)
print("Number of classes : ",np.unique(y_encoded, return_counts=True))
y = to_categorical(y_encoded, num_classes = len(np.unique(article_list)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
print(X_train.shape, X_test.shape)

# Model conversion
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
revealcancel_model = kc.convert_model_from_saved_files(
                            h5_file=saved_model_file,
                            json_file=model_json_file,
                            nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)

# Sanity checks
from deeplift.util import compile_func
import numpy as np
from keras import backend as K

deeplift_model = revealcancel_model
deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
                                       deeplift_model.get_layers()[-1].get_activation_vars())
original_model_predictions = model.predict(X_test, batch_size=100)
converted_model_predictions = deeplift.util.run_function_in_batches(
                                input_data_list=[X_test],
                                func=deeplift_prediction_func,
                                batch_size=100,
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
t0 = time.time()
for method_name, score_func in [('revealcancel', revealcancel_func)]:
    method_to_task_to_scores[method_name] = {}
    for task_idx in range(5):
        print("\tComputing scores for task: "+str(task_idx))
        scores = np.array(score_func(
                    task_idx=task_idx,
                    input_data_list=[X_test],
                    input_references_list=[np.zeros_like(X_test)],
                    batch_size=10,
                    progress_update=None))
        method_to_task_to_scores[method_name][task_idx] = scores
t1 = time.time()
total = t1-t0

#Function to plot scores of a figure
def viz_scores(scores,ax):
    reshaped_scores = scores.reshape(224,224,3)
    the_min = np.min(reshaped_scores)
    the_max = np.max(reshaped_scores)
    center = 0.0
    negative_vals = (reshaped_scores < 0.0)*reshaped_scores/(the_min + 10**-7)
    positive_vals = (reshaped_scores > 0.0)*reshaped_scores/float(the_max)
    reshaped_scores = -negative_vals + positive_vals
    # ax.imshow(-reshaped_scores, cmap="Greys")
    # reshaped_scores[np.where((reshaped_scores == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    ax.imshow(-reshaped_scores)
    ax.set_xticks([])
    ax.set_yticks([])


#Function to plot the result of masking on a single example, for converting
def plot_two_way_figures(idx, task_1, method_names):
    print("example index: "+str(idx))
    print("Order of columns is:","task "+str(task_1)+" scores")
    print("Order of the methods is: "+", ".join(str(x) for x in method_names))
    for method_name in method_names:
        scores = method_to_task_to_scores[method_name]
        mean_scores_over_all_tasks = np.mean(np.array([scores[i][idx] for i in range(5)]), axis=0)
        f, axarr = plt.subplots(1, 2, sharey=False, figsize=(15,10))
        viz_scores(X_test[idx], axarr[0])
        viz_scores(scores[task_1][idx] - mean_scores_over_all_tasks, axarr[1])
    plt.show()

# Plot scores and result of masking on a single example
method_names = ['revealcancel']

# backpacks, belts, briefs, casual shoes, flip flops

x_explain = [5,6]
# x_explain = flip_toexplain[:5]
for i in x_explain:
    # plot_two_way_figures(5838,0,method_names)
    plot_two_way_figures(i, 3, method_names)


# plot raw pixel data
# plt.imshow(X_test[i])
# plt.show()