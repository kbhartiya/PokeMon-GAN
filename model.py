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
from keras.optimizers import SGD, Adam, RMSprop
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
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
import warnings; warnings.filterwarnings("ignore")


#parser = ArgumentParser()
checkpoint_dir = './Checkpoints'

data = './resizedData'
if not os.path.exists(checkpoint_dir):
	os.mkdir(checkpoint_dir)

def generator_model():
	model = Sequential()
	model.add(Dense(input_dim=100,output_dim=1024))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(128*8*8))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((128,8,8),input_shape=(128*8*8,)))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(64,5,5,border_mode='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Convolution2D(128,5,5,border_mode='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(Convolution2D(256,5,5,border_mode='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(1,5,5,border_mode='same'))
	model.add(Activation('tanh'))
	model.compile(loss='mse',optimizer=RMSprop(0.001,0.5))
	print(model.summary())
	
	return model

def discriminator_model():
	model = Sequential()
	model.add(Convolution2D(64,5,5,border_mode='same',input_shape=(1,32,32)))
	model.add(LeakyReLU(alpha=0.2))
	model.add(AveragePooling2D(pool_size=(4,4)))
	model.add(Convolution2D(128,5,5))
	model.add(LeakyReLU(alpha=0.2))
	model.add(BatchNormalization())
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))	
	model.compile(loss='mse',optimizer=RMSprop(0.001,0.5),metrics=['accuracy'])
	print(model.summary())
	
	return model
	
	
def whole_model(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable=False
	model.add(discriminator)
	model.compile(loss='binary_crossentropy',optimizer=RMSprop(0.001,0.5))
	
	return model
	
def train_model(data,n_epochs=100,batch_size=96,weights=False):
	X_train = data
	X_train = (X_train.astype(np.float32)-127.5)/127.5
	#X_train = np.expand_dims(X_train, axis=3)
	X_train = X_train.reshape((X_train.shape[0],1) + X_train.shape[1:])
	
	discriminator = discriminator_model()
	generator = generator_model()
	
	if weights==True:
		generator.load_weights('g_mod.h5')
		discriminator.load_weights('d_mod.h5')
	
	adversarial_mod = whole_model(generator, discriminator)
	discriminator.trainable = True
	stoc_batch = np.zeros((batch_size,100))
	
	num_iterations = (X_train.shape[0] // batch_size)
	
	for epoch in range(n_epochs):
		print("Epoch : {}/{}".format(epoch+1, n_epochs))
		for iter in range(num_iterations):
			for i in range(batch_size):
				stoc_batch[i, :] = np.random.uniform(-1,1,100)
			
			real_batch = X_train[iter*batch_size:(iter+1)*batch_size]
			generated_image = generator.predict(stoc_batch,verbose=1)
			#print(real_batch.shape)
			#print(generated_image.shape)
			domain = np.concatenate((real_batch,generated_image))
			codomain = [1]*batch_size + [0]*batch_size
			d_loss = discriminator.train_on_batch(domain, codomain)
			print("Discriminator Loss(Batch: {}): {}".format(iter, d_loss))
			
			for i in range(batch_size):
				stoc_batch[i,:] = np.random.uniform(-1,1,100)
			
			discriminator.trainable = False
			g_loss = adversarial_mod.train_on_batch(stoc_batch,[1]*batch_size)
			discriminator.trainable = True
			print("Generator Loss(Batch: {}): {}".format(iter, g_loss))
			
			if epoch%10==0:
				generator.save_weights(checkpoint_dir+'/g_mod_'+str(epoch+1)+'.h5',True)
				discriminator.save_weights(checkpoint_dir+'/d_mod_'+str(epoch+1)+'.h5',True)
					 
def main():
	X_train = []
	images = os.listdir(data)

	for image in images:
		im = Image.open(os.path.join(data+"/"+image))
		im = ImageOps.fit(im, (32, 32), Image.ANTIALIAS)
		im = ImageOps.grayscale(im)
		im = np.asarray(im)
		X_train.append(im)
	X_train = np.array(X_train)
	#print(X_train.shape)
	train_model(X_train,n_epochs=400,batch_size=32,weights=False)
			
	
	
if __name__=='__main__':			
	main() 			
			
		
	
		
		
	
	
		
		
	
		
