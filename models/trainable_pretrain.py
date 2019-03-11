# get code dependencies
from comet_ml import Experiment  # must be imported before keras
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
import t4


# get data dependencies
img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'
# open_fruits = t4.Package.browse('quilt/open_fruit', 's3://quilt-example')
# open_fruits['training_data/X_meta.csv'].fetch(metadata_filepath)
# open_fruits['images_cropped'].fetch(img_dir)


# set up experiment logging
# set COMET_API_KEY in your environment variables
# or pass it as the first value in the Experiment object
# experiment = Experiment(
#     "...",
#     workspace="ceceshao1", project_name="aleksey-open-fruits"
# )


# get X and y values for flow_from_directory
X_meta = pd.read_csv(metadata_filepath)
X = X_meta[['CroppedImageURL']].values
y = X_meta['LabelName'].values


# define data generators
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)
train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical',
    subset='validation'
)


# load the pretrained model
# import t4
# t4.Package.browse('quilt/open_fruit_models', 's3://quilt-example')['bottleneck_model.h5']\
#     .fetch('bottleneck_model.h5')
# pretrained_model = keras.models.load_model('bottleneck_model.h5')


# define the model
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD

batch_size = 512
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


# set pretrained weights
for new_layer, old_layer in zip(model.layers[-4:], pretrained_model.layers[-4:]):
    new_layer.set_weights(old_layer.get_weights())


# leave the outermost convblock trainable, but freeze all other layers
for cnn_block_layer in model.layers[0].layers[:-4]:
    cnn_block_layer.trainable = False

    
# compile the model
model.compile(
    # one-tenth the standard SGD learning rate w/ some momentum
    optimizer=SGD(lr=1e-4, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# fit the model
from keras.callbacks import EarlyStopping
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    callbacks=[EarlyStopping(patience=2)]
)


# save model artifact
# model.save('/opt/ml/model/model.h5')
model.save('model.h5')

# experiment.end()