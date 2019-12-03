import os
import shutil
import numpy as np
import pandas as pd
import cv2
import imageio
from datetime import datetime as dt
import matplotlib.pyplot as plt
import random
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.optimizers import *
from keras import regularizers as reg
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from keras import optimizers
from keras.callbacks import History
from sklearn.preprocessing import LabelEncoder


source = r'D:\Deep Learning\DL Datasets\cell_images'
destination = r'D:\Deep Learning\DL Datasets\cell_images\data'
DATADIR= r'D:\Deep Learning\DL Datasets\cell_images'

def plt_train_val_acc(x,val_acc,train_acc,colors=['b']):
    plt.figure(figsize=(12,6))
    plt.grid()
    plt.plot(x,val_acc,'b', label = 'Validation Accuracy')
    plt.plot(x, train_acc, 'r', label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy after each epoch')
    plt.legend()

def plt_train_val_loss(x,val_loss,train_loss,colors=['b']):
    plt.figure(figsize=(12,6))
    plt.grid()
    plt.plot(x,val_loss,'b', label = 'Validation Loss')
    plt.plot(x, train_loss, 'r', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Cross-entropy loss')
    plt.legend()

train_file_names = [DATADIR+r'\data\train\Parasitized' +'\\'+ fname for fname in os.listdir(DATADIR+r"\data\train\Parasitized")] + [DATADIR+r'\data\train\Uninfected\\'+fname for fname in os.listdir(DATADIR+r'\data\train\Uninfected')]
X_train = [imageio.imread(image) for image in train_file_names]
X_train = np.array(X_train)
y_train = ["Parasitized"]*8000 + ["Uninfected"]*8000

#Validation files
val_file_names = [DATADIR+r"\data\validation\Parasitized\\"+fname for fname in os.listdir(DATADIR+r"\data\validation\Parasitized")] + [DATADIR+r"\data\validation\Uninfected\\"+fname for fname in os.listdir(DATADIR+r"\data\validation\Uninfected")]
X_val = [imageio.imread(image) for image in val_file_names]
X_val = np.array(X_val)
y_val = ["Parasitized"]*3000 + ["Uninfected"]*3000

#Test files
test_file_names = [DATADIR+r"\data\test\Parasitized\\"+fname for fname in os.listdir(DATADIR+r"\data\test\Parasitized")] + [DATADIR+r"\data\test\Uninfected\\"+fname for fname in os.listdir(DATADIR+r"\data\test\Uninfected")]
X_test = [imageio.imread(image) for image in test_file_names]
X_test = np.array(X_test)
y_test = ["Parasitized"]*3000 + ["Uninfected"]*3000

#Scaling all the values between 0 and 1
X_train = X_train.astype('float32')
X_val  = X_val.astype('float32')
X_test  = X_test.astype('float32')

X_train /= 255
X_val /= 255
X_test /= 255

print("Shape of the train dataset: ",X_train.shape)
print("Shape of the validation dataset: ",X_val.shape)
print("Shape of the test dataset: ",X_test.shape)

#Store the datasets in pickle files
os.mkdir(r"D:\Deep Learning\DL Datasets\cell_images\\normalized_data") if not os.path.isdir(r"D:\Deep Learning\DL Datasets\cell_images\normalized_data") else None

infected_image_dir = r'D:\Deep Learning\DL Datasets\cell_images\data\train\Parasitized\\'
filenames=random.sample(os.listdir(infected_image_dir),26)

#Display 25 images from Parasitized cells
plt.figure(figsize=(15,15))
for i in range(1,len(filenames)):
    row = i
    image = imageio.imread(infected_image_dir+filenames[i]) #Image(filename=image_dir+filenames[i])
    plt.subplot(5,5,row)
    plt.imshow(image)
plt.show()

uninfected_image_dir =r'D:\Deep Learning\DL Datasets\cell_images\data\train\Uninfected\\'
filenames=random.sample(os.listdir(uninfected_image_dir),26)

#Display 25 images from Uninfected cells
plt.figure(figsize=(15,15))
for i in range(1,len(filenames)):
    row = i
    image = imageio.imread(uninfected_image_dir+filenames[i]) #Image(filename=image_dir+filenames[i])
    plt.subplot(5,5,row)
    plt.imshow(image)
plt.show()

#Dimensions of our flicker images is 256 X 256
img_width, img_height = 128, 128

#Declaration of parameters needed for training and validation
train_data_dir = r'D:\Deep Learning\DL Datasets\cell_images\data\train'
validation_data_dir = r'D:\Deep Learning\DL Datasets\cell_images\data\validation'
nb_train_samples = 16000 #8000 training samples for each class
nb_validation_samples = 6000 #3000 validation samples for each class
epochs = 50
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

encoder=LabelEncoder()
encoder.fit(y_train)

y_train_enc=encoder.transform(y_train)
y_val_enc=encoder.transform(y_val)

#Declaring the model architecture.
sgd = optimizers.SGD(lr=0.01, decay=1e-6)

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.summary()

#Train the model
#We will use the below code snippet for aumenting the training data
train_datagen = ImageDataGenerator(rescale=1./255)

#Only rescale the test images, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(DATADIR+"/data/train/",
                                                    target_size=(128, 128),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(DATADIR+"/data/validation/",
                                                       target_size=(128, 128),
                                                       batch_size=batch_size,
                                                       class_mode='binary')


hist=History()
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[hist])

#Get Train loss vs validation loss

#Get model history
history=model.history

"""Plot train vs test loss"""
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

#List of epoch numbers
x = list(range(1,epochs+1))

#Display the loss
val_loss = history.history['val_loss'] #Validation Loss
train_loss = history.history['loss'] #Training Loss
plt_train_val_loss(x, val_loss, train_loss, ax)

"""Plot train vs validation accuracy"""
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Accuracy for each epochs')

#List of epoch numbers
x = list(range(1,epochs+1))

#Display the loss
val_acc = history.history['val_acc'] #Validation Accuracy
train_acc = history.history['acc'] #Training Accuracy
plt_train_val_acc(x, val_acc, train_acc, ax)

model.save(r'D:\Deep Learning\models\cell_classification_model_trained.h5')