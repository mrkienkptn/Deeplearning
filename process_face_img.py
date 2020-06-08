

from network import generator


import numpy as np
from tensorflow.keras.optimizers import Adam

import keras
from keras.applications import VGG16

from PIL import Image
import image_procress
import os
import time # tinh toan thoi gian chay het 1 batch
from math import ceil
import tensorflow as tf



#np.random.seed(10)
image_shape = (128,128,3) ## change to (128,128,3) later for testing
def generate(path_file):
    global image_shape

    gener = generator(image_shape)
    gener.load_weights('with_only_gener_savepoint_5.h5') # load weight  

   
    
    file = Image.open(path_file)
    file.load()
    array= np.array(file).shape
    if(array[2] == 4):
        rgb_image = Image.new('RGB',file.size,(255,255,255))
        rgb_image.paste(file,mask = file.split()[3])
        file = rgb_image
    print("file size", array)
    file = file.resize((128,128))
    
    file = np.array(file)
    print(file.shape)
    file = image_procress.normalize(file)
    img = np.array([file])
    generated_img = gener(img)
    generated_img = np.array(generated_img)
    
    term = image_procress.denormalize(generated_img[0])
    im = Image.fromarray(term)
    return im



