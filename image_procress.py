import numpy as np

import os
import sys
from PIL import Image
import matplotlib.pyplot as ppl
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
father_folder = '128128/'
downscale = 4
image_shape = (128,128,3)
def hr_images(images):
    images_hr = np.array(images)
    return images_hr


def lr_images(images_real ):
    global downscale
    images = []
    for img in  images_real:
        term = np.array(Image.fromarray(np.array(img)).resize(size = (image_shape[0]//downscale, image_shape[1]//downscale))) 
        images.append(np.array(Image.fromarray(np.array(term)).resize(size = (image_shape[0],image_shape[1]))))
    images_lr = np.array(images)
    
    return images_lr

def normalize(input_data):

    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def load_data(pathfile,files):
    global father_folder
    path = father_folder + pathfile
    path = path+'/'
    
    
    for file in os.listdir(path):
        
        image = Image.open(path+file)
               
        files.append(image)
    
    return files

def load_training_data(files):
    image = []
    for file in files:
        Image.Image.load(file)
        term = np.array(file)
        image.append(term)
    
    image_train_hr = hr_images(image)
    image_train_hr = normalize(image_train_hr)
    image_train_lr = lr_images(image)
    image_train_lr = normalize(image_train_lr)
    return image_train_hr, image_train_lr

def file_close(files):
    for file in files:
        Image.Image.close(file)


def local_downscale(img_array):
    global downscale
    global image_shape
    term = np.array(Image.fromarray(img_array).resize(size = (image_shape[0]//downscale, image_shape[1]//downscale)))
    term = np.array(Image.fromarray(np.array(term)).resize(size = (image_shape[0],image_shape[1])))
    return term