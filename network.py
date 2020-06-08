import keras
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, Activation, UpSampling2D



from tensorflow.keras.models import Model

import numpy as np






    

def generator(noise_shape):
        def residual_block(model,kernel_size, num_filter, stride):
            ip = model
            model = Conv2D(filters= num_filter,kernel_size = kernel_size,strides = stride,padding = 'same')(model)
            model = BatchNormalization()(model)
            model = PReLU(shared_axes = [1,2])(model)
    
            model = Conv2D(filters=  num_filter,kernel_size = kernel_size,strides= stride,padding = 'same')(model)
    
            model = Add()([ip, model])
            model = BatchNormalization()(model)
            return model

        gen_input = Input(noise_shape)
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





