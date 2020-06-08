import numpy as np
import os
import glob
import math
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.models import load_model


Image.MAX_IMAGE_PIXELS = None
LR_SIZE = 24
HR_SIZE = 96


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def load_image(path):
    return np.array(Image.open(path))



def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
# Network
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)

def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve_and_plot(model_fine_tuned, lr_image):
    
    sr_ft = resolve_single(model_fine_tuned, lr_image)
    x = sr_ft.numpy()
    im = Image.fromarray(x, 'RGB')
    return im


# ---------------------------------------
# Split image to 128 * 128 part
# ---------------------------------------


start_pos = start_x, start_y = (0, 0)
cropped_image_size = w, h = (128, 128)

def get_crop_images(file):
    img = Image.open(file)
    width, height = img.size
    n_cols = math.ceil(width/w)
    if width%w == 0:
        n_cols = int(width/w)
    n_rows = math.ceil(height/h)
    if (height%h == 0):
        n_rows = int(height/h)
    frame_num = 1
    croplist = []
    for col_i in range(0, width, w):
        for row_i in range(0, height, h):
            if col_i + w <= width and row_i+ h <= height:
                crop = img.crop((col_i, row_i, col_i+w, row_i + h))
            elif col_i + w <= width:
                crop = img.crop((col_i, row_i, col_i + w, height))
            elif row_i + h <= height:
                crop = img.crop((col_i, row_i, width, row_i+h))
            else:
                crop = img.crop((col_i, row_i, width, height))
            croplist.append(crop)
    return croplist, n_rows, n_cols, width, height

# ---------------------------------------
# Merge all generated parts into a big image
# ---------------------------------------

big_scale_image = new_w, new_h = (w*4, h*4)

# load model

generator = sr_resnet()
generator.load_weights('gan_generator.h5')

def make_full_image(image_name):
    
    
    croplist, n_rows, n_cols, old_width, old_height = get_crop_images(image_name)
    full_image = Image.new('RGB', (old_width*4, old_height*4))
    

    for c in range(n_cols):
        for r in range(n_rows):
            k = (c) * n_rows + r
            print(k)
            
            im = croplist[k]

            im = np.array(im)
#             im = normalize(im)
#             gen_im_normal = G(im)
#             im_denormalize = denormalize(gen_im_normal)
#             im_denormalize = Image.fromarray(im_denormalize.numpy(), 'RGB')
            im_denormalize = resolve_and_plot(generator, im)
            full_image.paste(im_denormalize, (c*new_w, r*new_h))
            
    return full_image





