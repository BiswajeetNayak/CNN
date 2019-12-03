import os
import imageio
import matplotlib.pyplot as plt
import random
from datetime import datetime as dt
import cv2

#A utility function to resize a given input image
def resize_image(image):
    resized_image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA) #Resize all the images to 128X128 dimensions
    return resized_image

source = r'D:\Deep Learning\DL Datasets\cell_images'
destination = r'D:\Deep Learning\DL Datasets\cell_images\data'
DATADIR= r'D:\Deep Learning\DL Datasets\cell_images'

#We will check if the folder 'data','train','test','validation' exists. If its not, we will create a folder with the same name
os.mkdir(destination) if not os.path.isdir(destination) else None
os.mkdir(destination+"/train") if not os.path.isdir(destination+"/train") else None
os.mkdir(destination+"/validation") if not os.path.isdir(destination+"/validation") else None
os.mkdir(destination+"/test") if not os.path.isdir(destination+"/test") else None

#We will check if the folder 'Parasitized' and 'Uninfected' exists. If its not, we will create a folder with the same name
os.mkdir(destination+"/train"+"/Parasitized") if not os.path.isdir(destination+"/train"+"/Parasitized") else None
os.mkdir(destination+"/train"+"/Uninfected") if not os.path.isdir(destination+"/train"+"/Uninfected") else None

os.mkdir(destination+"/validation"+"/Parasitized") if not os.path.isdir(destination+"/validation"+"/Parasitized") else None
os.mkdir(destination+"/validation"+"/Uninfected") if not os.path.isdir(destination+"/validation"+"/Uninfected") else None

os.mkdir(destination+"/test"+"/Parasitized") if not os.path.isdir(destination+"/test"+"/Parasitized") else None
os.mkdir(destination+"/test"+"/Uninfected") if not os.path.isdir(destination+"/test"+"/Uninfected") else None

#Get all the filenames from the original "cell_images" data folder
par_filenames=os.listdir(source+"/Parasitized")
un_filenames=os.listdir(source+"/Uninfected")

#Get details about the number of images present
print("Number of images of type 'Parasitized': ",len(par_filenames))
print("Number of images of type 'Uninfected': ",len(un_filenames))

st = dt.now()

# First, the Parasitized images
par_train_images = random.sample(par_filenames, 8000)
par_val_images = random.sample(list(set(par_filenames) - set(par_train_images)), 3000)
par_test_images = list(set(par_filenames) - set(par_train_images) - set(par_val_images))

for file in par_train_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Parasitized" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/train" + "/Parasitized/" + file, resized_image)
print("Train folder created for Parasitized images...")

for file in par_val_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Parasitized" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/validation" + "/Parasitized/" + file, resized_image)
print("Validation folder created for Parasitized images...")

for file in par_test_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Parasitized" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/test" + "/Parasitized/" + file, resized_image)
print("Test folder created for Parasitized images...")

# Now, the uninfected files
un_train_images = random.sample(un_filenames, 8000)
un_val_images = random.sample(list(set(un_filenames) - set(un_train_images)), 3000)
un_test_images = list(set(un_filenames) - set(un_train_images) - set(un_val_images))

for file in un_train_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Uninfected" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/train" + "/Uninfected/" + file, resized_image)
print("Train folder created for Uninfected images...")

for file in un_val_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Uninfected" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/validation" + "/Uninfected/" + file, resized_image)
print("Validation folder created for Uninfected images...")

for file in un_test_images:
    if (file.endswith("png")):
        image = imageio.imread(source + "/Uninfected" + "/" + file)
        resized_image = resize_image(image)
        imageio.imsave(destination + "/test" + "/Uninfected/" + file, resized_image)
print("Test folder created for Uninfected images...")

print("\nTotal time taken to resize the images and create the dataset: ", dt.now() - st)


'''orig_input_dataset = r'D:\Deep Learning\DL Datasets\cell_images'
base_path = r'D:\Deep Learning\DL Datasets'

train_path = os.path.sep.join([base_path,'training'])
val_path = os.path.sep.join([base_path,'validation'])
test_path = os.path.sep.join([base_path,'testing'])

train_split = 0.8
val_split = 0.1
'''