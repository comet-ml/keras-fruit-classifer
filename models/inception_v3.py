# import dependencies 
from comet_ml import Experiment 
import t4
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping



# set up experiment logging
# set COMET_API_KEY in your environment variables
# or pass it as the first value in the Experiment object
experiment = Experiment("...",workspace="WORKSPACE_NAME", project_name="PROJECT_NAME")


# Define parameters
# dimensions of our images
img_width, img_height = 150, 150

batch_size = 16
num_classes = 16
epochs = 50
activation = 'relu'
min_delta=0
patience=4
dropout=0.2
lr=0.0001

train_samples = 27593
validation_samples = 6889


params={'batch_size':batch_size,
        'num_classes':num_classes,
        'epochs':epochs,
        'min_delta':min_delta,
        'patience':patience,
        'learning_rate':lr,
        'dropout':dropout
}
experiment.log_parameters(params)




# Define data generators
from keras.applications.inception_v3 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 #set the validation split 
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

train_generator = train_datagen.flow_from_directory(
    './data/quilt/open_fruit/images_cropped',
    target_size=(150, 150),
    shuffle=True,
    seed=20,
    batch_size = batch_size,
    class_mode='categorical',
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    './data/quilt/open_fruit/images_cropped',
    target_size=(150, 150),
    seed=20,
    batch_size=batch_size,
    class_mode='categorical',
    subset = "validation"
)



# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(150,150))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 16 classes for the fruits 
predictions = Dense(16, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


for i,layer in enumerate(base_model.layers):
    print(i,layer.name)

import pathlib
sample_size = len(list(pathlib.Path('./data/quilt/open_fruit/images_cropped').rglob('./*')))



# train the model on the new data for a few epochs
model.fit_generator(
    train_generator,
    steps_per_epoch=sample_size // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)]
)


# freeze the first 249 layers and unfreeze the rest:
for layer in model.layers[:310]:
   layer.trainable = False
for layer in model.layers[310:]:
   layer.trainable = True



# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate so we don't lose the value of the pre-trained model
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
    train_generator,1
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples// batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)]
)


#save locally
model.save_weights('inceptionv3_tuned.h5') 

#save to Comet Asset Tab
# you can retrieve these weights later via the REST API 
experiment.log_asset(file_path='./inceptionv3_tuned.h5', file_name='inceptionv3_tuned.h5') 