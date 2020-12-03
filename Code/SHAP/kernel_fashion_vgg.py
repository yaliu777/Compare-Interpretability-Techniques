from tqdm import tqdm
from keras.preprocessing import image
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
import matplotlib.pylab as pl
import shap
import numpy as np
import time

DATA_PATH = '/home/ubuntu/fashion-dataset/'
IMAGE_PATH = DATA_PATH + 'images/'
IMAGE_SIZE = 224
# LIMIT_IMAGES = 1000

# def load_images(names, articletype):
#     image_array = []
#     for image_name in tqdm(names, desc = 'reading images for ' + articletype):
#         img_path = IMAGE_PATH + image_name
#         try:
#             img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
#         except:
#             continue
#         img = image.img_to_array(img)
#         image_array.append(img)
#     return np.array(image_array)
#
# dfstyles = pd.read_csv('/home/yaliu20160710/styles.csv', error_bad_lines=False, warn_bad_lines=False)
# dfstyles['image'] = dfstyles['id'].map(lambda x: str(x) + '.jpg')
# dfstyles.columns = dfstyles.columns.str.lower()
# print(dfstyles.shape)
# dfstyles.head()
#
# dfstyles['articletype'].nunique() # 143 classes
# dfstyles['articletype'].value_counts().head()
# dfstyles['articletype'].value_counts().tail()
#
# dfstyles['cntarticle'] = dfstyles.groupby('articletype')['id'].transform('count')
# dfdata = dfstyles[dfstyles['cntarticle'] > 500]
# print(dfdata.shape, dfdata['articletype'].nunique())
# dfarticles =dfdata.groupby('articletype',as_index=False)['id'].count()

# image_list = []
# article_list = []
# for index, grouprow in dfarticles.iterrows():
#     if index > 4:
#         continue
#     image_names = dfdata[dfdata['articletype'] == grouprow['articletype']]['image'].values
#     if len(image_names) > LIMIT_IMAGES:
#         image_names = image_names[:LIMIT_IMAGES]
#     image_list.extend(load_images(image_names, grouprow['articletype']))
#     article_list.extend(len(image_names) * [grouprow['articletype']])

# X = np.array(image_list) / 255.0
# X = X.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
# y_encoded = LabelEncoder().fit_transform(article_list)
# print("Number of classes : ",np.unique(y_encoded, return_counts=True))
# y = to_categorical(y_encoded, num_classes = len(np.unique(article_list)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
# print(X_train.shape, X_test.shape)

# load json and create model
json_file = open('fashion_5class_vgg_config.json', 'r')
model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("fashion_5class_vgg_weights.h5")

# idx = 1
# # plot
# img_orig = X_test[idx]
# plt.imshow(img_orig, interpolation='nearest')
# plt.show()

# img = X_test[idx]
# img[0:256, 0:256] = [255, 0, 0]
# img = Image.fromarray(img, 'RGB')

# segment the image so we don't have to explain every pixel
# segments_slic = slic(img_orig, n_segments=50, compactness=30, sigma=3)

# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out
def f(z):
    return model.predict(mask_image(z, segments_slic, img_orig, 255))

# make a color map
from matplotlib.colors import LinearSegmentedColormap
colors = []
for l in np.linspace(1,0,100):
    colors.append((245/255,39/255,87/255,l))
for l in np.linspace(0,1,100):
    colors.append((24/255,196/255,93/255,l))
cm = LinearSegmentedColormap.from_list("shap", colors)


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

# plot our explanations


file = "2608.jpg"
files = ['46108.jpg', '10746.jpg', '49831.jpg', '26677.jpg', '20948.jpg']
shoes = ['34861.jpg', '29576.jpg', '12506.jpg', '13646.jpg', '6810.jpg']
t0 = time.time()
for file in shoes:
    img = image.load_img(IMAGE_PATH + file, target_size=(224, 224))
    img_orig = image.img_to_array(img)

    segments_slic = slic(img, n_segments=30, compactness=30, sigma=3)
    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000)

    # get the top predictions from the model
    preds = model.predict(np.expand_dims(img_orig.copy(), axis=0))
    top_preds = np.argsort(-preds)

    fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12,4))
    inds = top_preds[0]
    axes[0].imshow(img)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    for i in range(3):
        m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
        axes[i+1].set_title(str(inds[i]))
        axes[i+1].imshow(img.convert('LA'), alpha=0.15)
        im = axes[i+1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        axes[i+1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    pl.show()
t1 = time.time()
total = t1-t0
print(total)
