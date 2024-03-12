# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:34:45 2020

@author: Kirtan
"""

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, obj, tok, tokenizer, photo, max_length, max_length_obj):
	in_text = 'startseq'
	seq_obj=tok.texts_to_sequences([obj])[0]
	#seq_obj2=tokenizer.texts_to_sequences([scene])[0]
	seq_obj=pad_sequences([seq_obj],maxlen=max_length_obj)
    #obj = seq_obj = desc_list[0][0]
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
       	# seed the generation process
		#print(in_text)
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,seq_obj,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		#print(word)
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	#print("caption:",in_text)
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
tok = load(open('tok.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
max_length_obj=49
obj='person,person'
#scene='archaelogical_excavation,badlands,burial_chamber'
# load the model
model = load_model('model-ep004-loss3.868-val_loss4.273.h5')#11epochs
# load and prepare the photograph
photo = extract_features('F://MTECH//Dataset//Flickr8k//Test_Images//452419961_6d42ab7000.jpg')
# generate description
description = generate_desc(model, obj, tok, tokenizer, photo, max_length, max_length_obj)
print('objects: ',obj)
#print('scene: ',scene)
print('Caption: ',description)