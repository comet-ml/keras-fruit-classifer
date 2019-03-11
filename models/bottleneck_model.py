# get code dependencies
import numpy as np
import pandas as pd
from comet_ml import Experiment  # must be imported before keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
import t4


# get data dependencies
img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'
open_fruits = t4.Package.browse('quilt/open_fruit', 's3://quilt-example')
open_fruits['training_data/X_meta.csv'].fetch(metadata_filepath)
open_fruits['images_cropped'].fetch(img_dir)


# set up experiment logging
# set COMET_API_KEY in your environment variables
experiment = Experiment(workspace="ceceshao1", project_name="aleksey-open-fruits")


# get X and y values for flow_from_dataframe
X_meta = pd.read_csv(metadata_filepath)
X = X_meta[['CroppedImageURL']].values
y = X_meta['LabelName'].values


# randomly partition data into train and test
np.random.seed(42)
idxs = np.arange(len(X))
np.random.shuffle(idxs)
split_ratio = 0.8
n_samples = len(X)
split_idx = int(split_ratio * n_samples)
X_train, X_test = X[idxs[:split_idx]], X[idxs[split_idx:]]
y_train, y_test = y[idxs[:split_idx]], y[idxs[split_idx:]]
X_train_df = pd.DataFrame().assign(ImagePath=X_train[:, 0], ImageClass=y_train)
X_test_df = pd.DataFrame().assign(ImagePath=X_test[:, 0], ImageClass=y_test)


# define data generators
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1/255
)
train_generator = train_datagen.flow_from_dataframe(
    X_train_df,
    directory=img_dir,
    x_col='ImagePath',
    y_col='ImageClass',
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_dataframe(
    X_test_df,
    directory=img_dir,
    x_col='ImagePath',
    y_col='ImageClass',
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical'
)


# define the model
batch_size = 512
model = keras.applications.VGG16(include_top=False, weights='imagenet')
prior = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=(48, 48, 3)
)
model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.2, name='Dropout_Regularization'))
model.add(Dense(16, activation='sigmoid', name='Output'))
model.layers[0].trainable = False
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fit the model
from keras.callbacks import EarlyStopping
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=20,
    callbacks=[EarlyStopping(patience=2)]
)


# save model artifact
model.save('/opt/ml/model/model.h5')
# model.save('model.h5')
