from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.segmentation import felzenszwalb, slic, quickshift
# felzenszwalb, quickshift not good

num_classes = 10
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

# x_0to9 = x_test[3:4]
x_0to9 = x_test[5:6]
# list_1to9 = [5,43,18,4,8,11,0,61,62]
# for i in range(9):
#     idx = list_1to9[i]
#     x_0to9 = np.concatenate((x_0to9, x_test[idx:idx+1]), axis=0)

def new_predict_fn(images):
    images = rgb2gray(images).reshape(10,28,28,1)
    return model.predict(images)

def segment_fn(images):
    return slic(images)


from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

list_1to9 = [3,5,43,18,4,8,11,0,61,62]
for num in range(len(list_1to9)):
    x = x_test[list_1to9[num]:list_1to9[num]+1] # (1,28,28,1)
    explanation = explainer.explain_instance(x[0][:,:,0].astype('double'), new_predict_fn, top_labels=5,
                                             hide_color=0, num_samples=1000, segmentation_fn=segment_fn)

    # plot explanation weights onto a heatmap visualization
    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.show()


# # see the top 5 superpixels that are most positive towards the class with the rest of the image hidden
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
# plt.imshow(mark_boundaries(temp, mask))
# plt.show()
# # with the rest of the image present
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
# plt.imshow(mark_boundaries(temp, mask))
# plt.show()
# # see the 'pros and cons' (pros in green, cons in red)
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# plt.imshow(mark_boundaries(temp, mask))
# plt.show()
# # the pros and cons that have weight at least 0.1
# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
# plt.imshow(mark_boundaries(temp, mask))
# plt.show()

