# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:13:00 2020

@author: TEQIP
"""

from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

model = load_model('model.h5')
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy','mse'])

filename = 'Flicker8k_text\Flickr_8k.testImages - Copy.txt'
train = load_set(filename)

for img in train:
	test='C:\\Users\\TEQIP\\Desktop\\Image Caption\\Flicker8k_Dataset\\'+img+'.jpg'
	test_image= image.load_img(test)
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
#test_image = test_image.reshape(img_width, img_height*3)
	result = model.predict(test_image)
	print(result)