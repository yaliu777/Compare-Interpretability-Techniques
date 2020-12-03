from tqdm import tqdm
from keras.preprocessing import image
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.segmentation import felzenszwalb, slic, quickshift
# felzenszwalb, quickshift not good

DATA_PATH = '/home/ubuntu/fashion-dataset/'
IMAGE_PATH = DATA_PATH + 'images/'
IMAGE_SIZE = 224
LIMIT_IMAGES = 1000

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)

# load json and create model
json_file = open('fashion_5class_vgg_config.json', 'r')
model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("fashion_5class_vgg_weights.h5")



# x_0to9 = x_test[3:4]
x_0to9 = X_test[5:6]
# list_1to9 = [5,43,18,4,8,11,0,61,62]
# for i in range(9):
#     idx = list_1to9[i]
#     x_0to9 = np.concatenate((x_0to9, x_test[idx:idx+1]), axis=0)

# def new_predict_fn(images):
#     images = rgb2gray(images).reshape(224,224,3)
#     return model.predict(images)

def segment_fn(images):
    return slic(images)


from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

x_explain = [1,4,5,8,9,10,12,16,17,18,19]
t0 = time.time()
for num in range(len(x_explain)):
    # x = X_test[x_explain[num]:x_explain[num]+1]
    x = X_test[x_explain[num]]
    explanation = explainer.explain_instance(x.astype('double'), model.predict, top_labels=5,
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
t1 = time.time()
total = t1-t0

for num in x_explain:
    plt.imshow(X_test[num])
    plt.show()