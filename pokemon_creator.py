from __future__ import print_function
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras import utils
import numpy as np
from PIL import Image, ImageOps
import argparse
import math
import sys
import os
import glob
import argparse
from keras import regularizers
import warnings; warnings.filterwarnings('ignore')

def generate(BATCH_SIZE):
    generator = generator_model()
    #generator.compile(loss='binary_crossentropy', optimizer=Adam())
    generator.load_weights('./Checkpoints/g_mod_71.h5')
    noise = np.zeros((BATCH_SIZE, 100))
    a = np.random.uniform(-1, 1, 100)
    b = np.random.uniform(-1, 1, 100)
    grad = (b - a) / BATCH_SIZE
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    #image = combine_images(generated_images)
    print(generated_images.shape)
    for image in generated_images:
        image = image[0]
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save("dirty1.png")
        #Image.fromarray(image.astype(np.uint8)).show()
        '''
        clean(image)
        image = Image.fromarray(image.astype(np.uint8))
        image.show()        
        image.save("clean1.png")
        '''
        
def generator_model():
	model = Sequential()
	model.add(Dense(input_dim=100,output_dim=1024))
	model.add(Activation('tanh'))
	model.add(Dense(128*8*8))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((128,8,8),input_shape=(128*8*8,)))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(64,5,5,border_mode='same'))
	model.add(Activation('tanh'))
	model.add(BatchNormalization())
	model.add(Convolution2D(128,5,5,border_mode='same'))
	model.add(Activation('tanh'))
	model.add(BatchNormalization())
	model.add(Convolution2D(256,5,5,border_mode='same'))
	model.add(Activation('tanh'))
	model.add(BatchNormalization())
	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(1,5,5,border_mode='same'))
	model.add(Activation('tanh'))
	model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001))
	print(model.summary())
	
	return model

if __name__=='__main__':
	generate(1)	
	
