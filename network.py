import keras
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.models import Input
from keras.layers import add
from keras.models import Model
from keras.layers import Conv2DTranspose
import numpy as np
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import UpSampling2D
from keras.models import load_model

class Generator():
    def __init__(self, noise_shape):
            self.noise_shape = noise_shape
          
    

    def generator(self):
            def residual_block(model,kernel_size, num_filter, stride):
                ip = model
                model = Conv2D(filters= num_filter,kernel_size = kernel_size,strides = stride,padding = 'same')(model)
                model = BatchNormalization()(model)
                model = PReLU(shared_axes = [1,2])(model)
        
                model = Conv2D(filters=  num_filter,kernel_size = kernel_size,strides= stride,padding = 'same')(model)
        
                model = add([ip,model])
                model = BatchNormalization()(model)
                return model

            gen_input = Input(self.noise_shape)
            model = Conv2D(filters= 32,kernel_size =3, strides = (2,2),padding = 'same')(gen_input)
            model = PReLU(shared_axes = [1,2])(model)
            model = Conv2D(filters= 32,kernel_size =3, strides = (1,1),padding = 'same')(model)
            model = BatchNormalization()(model)

            model = Conv2D(filters= 32,kernel_size =3, strides = (1,1),padding = 'same')(model)
            model = BatchNormalization()(model)


            model = Conv2D(filters= 64,kernel_size =3, strides = (2,2),padding = 'same')(model)
            model = BatchNormalization()(model)
            for i in range(5):
                model = PReLU(shared_axes = [1,2])(model)
                model = residual_block(model, kernel_size =3,num_filter= 64,stride = 1)
            model = PReLU(shared_axes = [1,2])(model)
            

            model = Conv2D(filters= 128,kernel_size =3, strides = (2,2),padding = 'same')(model)
            model = BatchNormalization()(model)
            for i in range(5):
                model = PReLU(shared_axes = [1,2])(model)
                model = residual_block(model, kernel_size =3,num_filter= 128,stride = 1)
            model = PReLU(shared_axes = [1,2])(model)


         
           

            model = Conv2D(filters= 128,kernel_size =3, strides = (1,1),padding = 'same')(model)
            model = UpSampling2D(size = 2)(model)
            model = PReLU(shared_axes = [1,2])(model)

            model = Conv2D(filters= 64,kernel_size =3, strides = (1,1),padding = 'same')(model)
            model = UpSampling2D(size = 2)(model)
            model = PReLU(shared_axes = [1,2])(model)

            model = Conv2D(filters= 32,kernel_size =3, strides = (1,1),padding = 'same')(model)
            model = UpSampling2D(size = 2)(model)
            model = PReLU(shared_axes = [1,2])(model)


            model = Conv2D(filters= 3, kernel_size =3, strides = (1,1), padding = 'same')(model)
            model = Activation('tanh')(model)
            return Model(inputs = gen_input, outputs = model )



class Discrimator():
    def __init__(self,noise_shape):
        self.noise_shape = noise_shape

    def discrimator(self):
        dis_input = Input(self.noise_shape)
       
        model = Conv2D(filters = 32, kernel_size = 3, strides = (2,2), padding = 'same')(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
   
        
        
        
      
        model = Conv2D(filters = 64, kernel_size = 3, strides = (2,2), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)      

        model = Conv2D(filters = 64, kernel_size = 3, strides = (1,1), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)

       
  

        model = Conv2D(filters = 128, kernel_size = 3, strides = (2,2), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)      

        model = Conv2D(filters = 128, kernel_size = 3, strides = (1,1), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)






        model = Conv2D(filters = 256, kernel_size = 3, strides = (2,2), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)  

        model = Conv2D(filters = 256, kernel_size = 3, strides = (1,1), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)
      


        
        model = Conv2D(filters = 512, kernel_size = 3, strides = (2,2), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)  

        model = Conv2D(filters = 512, kernel_size = 3, strides = (1,1), padding = 'same')(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha = 0.2)(model)
        model = Flatten()(model)
        
        model = Dense(units = 1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(units = 1)(model)
        model = Activation('sigmoid')(model)
        
        return Model(inputs = dis_input, outputs = model)



