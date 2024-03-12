# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:38:38 2020

@author: Kirtan Patel
"""


#import os
#import pickle
#from keras.preprocessing import image
##from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
##from keras.applications.vgg16 import preprocess_input
#from keras.applications.inception_v3 import preprocess_input
#import numpy as np
##from keras.layers import merge, Input
#from keras.layers import Input
#from matplotlib import pyplot
#
#image_input = Input(shape=(224,224,3))
#model = InceptionV3(include_top=False,weights="imagenet",input_tensor=image_input)
#model.summary()
#
#vgg16_feature_list=[]
#img_path ="998845445.jpg"
#img = image.load_img(img_path, target_size=(224, 224))
#img_data = image.img_to_array(img)
#img_data = np.expand_dims(img_data, axis=0)
#img_data = preprocess_input(img_data)

#vgg16_feature = model.predict(img_data)
#vgg16_feature_np = np.array(vgg16_feature)
#vgg16_feature_list.append(vgg16_feature_np.flatten())

#vgg16_feature_list_np = np.array(vgg16_feature_list)

#vgg16_feature_list_np.shape
#print("Feature Vector: ",vgg16_feature_list_np)
#size=8
#for fmap in vgg16_feature:
#    #plot output from each block
#    ix=1
#    for _ in range(size):
#        for _ in range(size):
#            #specify subplot and turn of axis
#            ax=pyplot.subplot(size,size,ix)
#            ax.set_xticks([])
#            ax.set_yticks([])
#            #plot filter channel in grayscale
#            pyplot.imshow(fmap[0,:,:,ix-1],cmap="gray")
#            ix+=1
#    #show the figure
#    pyplot.show()



from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = InceptionV3()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(299, 299))#vgg16-shape=(224,224)
		#print(image)
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		#print(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = "C:\\Users\\admin\\Desktop\\Image Caption\\Flicker8k_Dataset"
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features_v3.pkl', 'wb'))
