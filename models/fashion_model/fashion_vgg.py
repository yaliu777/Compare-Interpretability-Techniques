# https://www.kaggle.com/ilkdem/image-classification-and-similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

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

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred,axis = 1)
    y_test_classes = np.argmax(y_test, axis = 1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    sns.heatmap(cm, annot = True,fmt='.0f')
    plt.show()

DATA_PATH = '/home/ubuntu/fashion-dataset/'
IMAGE_PATH = DATA_PATH + 'images/'
IMAGE_SIZE = 224
LIMIT_IMAGES = 1000

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

# show some images
# imglist = [IMAGE_PATH + x for x in dfdata['image'].sample(10).values]
# fig,ax = plt.subplots(2,5,figsize=(18,10))
# for index, img_file in enumerate(imglist):
#     img = plt.imread(img_file)
#     x = int(index / 5)
#     y = index % 5
#     ax[x,y].imshow(img)
# plt.show()

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
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)


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

# save model and configuration
model_json = model.to_json()
with open("fashion_5class_vgg_config.json", "w") as json_file:
    json_file.write(model_json)
# model.save_weights("cifar_vgg_weights.h5")

# fit model
checkpoint = ModelCheckpoint("fashion_5class_vgg_weights.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint])
# evaluate model
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plot_confusion_matrix(model, X_test, y_test)

