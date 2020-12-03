from tqdm import tqdm
from keras.preprocessing import image
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import time

DATA_PATH = '/home/ubuntu/fashion-dataset/'
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

dfstyles = pd.read_csv('/home/yaliu20160710/styles.csv', error_bad_lines=False, warn_bad_lines=False)
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
fashion_dict = {}
for index, grouprow in dfarticles.iterrows():
    if index > 4:
        continue
    image_names = dfdata[dfdata['articletype'] == grouprow['articletype']]['image'].values
    if len(image_names) > LIMIT_IMAGES:
        image_names = image_names[:LIMIT_IMAGES]
    image_list.extend(load_images(image_names, grouprow['articletype']))
    article_list.extend(len(image_names) * [grouprow['articletype']])
    fashion_dict[grouprow['articletype']] = image_names

X = np.array(image_list) / 255.0
X = X.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
y_encoded = LabelEncoder().fit_transform(article_list)
print("Number of classes : ",np.unique(y_encoded, return_counts=True))
y = to_categorical(y_encoded, num_classes = len(np.unique(article_list)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
print(X_train.shape, X_test.shape)

# load json and create model
json_file = open('fashion_5class_vgg_config.json', 'r')
model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("fashion_5class_vgg_weights.h5")

import shap
import numpy as np

background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
e = shap.DeepExplainer(model, background)

# for Inception model
# shap.explainers.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

# plot first few images
# for j in range(20):
#     for i in range(1):
#         # define subplot
#         pyplot.subplot(330 + 1 + i)
#         # plot raw pixel data
#         pyplot.imshow(X_test[j])
#     pyplot.show()
x_toexplain = [0,1,2,5,6,11,13,15,19,23,29]
# find x in X_test that are flips and casual shoes
flip_toexplain, shoes_toexplain = [], []
for i in range(len(y_test)):
    if sum(y_test[i] == np.array([0,0,0,1,0])) == 5:
        shoes_toexplain.append(i)
    elif sum(y_test[i] == np.array([0,0,0,0,1])) == 5:
        flip_toexplain.append(i)
    else:
        continue

for num in flip_toexplain[:5]:
    plt.imshow(X_test[num])
    plt.show()

# show the figure

for i in range(2):
    x = X_test[i:i+1]
    shap_values = e.shap_values(x)
    # plot the feature attributions
    shap.image_plot(shap_values, -x)


t0 = time.time()
for num in flip_toexplain[:5]:
    x = X_test[num:num+1]
    shap_values = e.shap_values(x)
    # plot the feature attributions
    shap.image_plot(shap_values, -x)
t1 = time.time()
total = t1-t0

# backpacks, belts, briefs, casual shoes, flip flops

# rotate an image and see its explanation
import cv2
flip = X_test[1]
flip_rotate = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
# cv2.rotate(a, cv2.ROTATE_180)
# cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE)
plt.imshow(flip_rotate)
plt.show()
flip_rotate = np.expand_dims(flip_rotate, axis=0)
shap_values = e.shap_values(flip_rotate)
shap.image_plot(shap_values, -flip_rotate)
print(model.predict(flip_rotate))
