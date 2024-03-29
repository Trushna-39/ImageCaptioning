# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:03:30 2020

@author: Trushna
"""

from numpy import argmax
from pickle import load,dump
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return list(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

def load_object(doc,train):
    descriptions = dict()
    #print(train)
    doc=load_doc(doc)
    #process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        #print(image_id,image_desc)
        # remove filename from image id
        image_id = image_id.split('.')[0]
        if image_id in train:
            descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
    return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def max_length_Obj(objects):
    lines = to_lines(objects)
    l=[]
    for i in range(len(lines)):
        l.append(len(lines[i]))
    return max(l)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, obj, scene, tok, tokenizer, photo, max_length, max_length_obj):
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

# evaluate the skill of the model
def evaluate_model(model, objects, descriptions, photos, tok, tokenizer, max_length, max_length_obj):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc_list in objects.items():
		# generate description
		#print(key, desc_list)
		yhat = generate_desc(model, desc_list[0][0], desc_list[0][1], tok, tokenizer, photos[key], max_length, max_length_obj)
		#print(key, yhat)
        # store actual and predicted
		#print(desc_list)
		references = [d.split() for d in desc_list[0]]
		actual.append(references)
		predicted.append(yhat.split())
	print("P: ",predicted)
	# calculate BLEU score
	method=SmoothingFunction()
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, smoothing_function=method.method2))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, smoothing_function=method.method3))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, smoothing_function=method.method4))

# prepare tokenizer on train set

# load training dataset (6K)
filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

objects = load_object('Objects.txt', train)
print('Full Descriptions: train=',len(objects))

file = load_object('‪Output.txt', train)
print('Full Descriptions: train=',len(file))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
tok=create_tokenizer(file)
vocab=len(tok.word_index)+1
print("Objects Vocabulary:%d" % vocab)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
dump(tok, open('tok.pkl', 'wb'))
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

max_length_obj = max_length_Obj(file)
print('Objects Length: %d' % max_length_obj)

# prepare test set

# load test set
filename = 'Flicker8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features

objects1 = load_object('Test_doc.txt', test)
print('Full Descriptions: train=',len(objects1))

test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model-ep004-loss3.868-val_loss4.273.h5'
model = load_model(filename)
# evaluate model
evaluate_model(model, objects1, test_descriptions, test_features, tok, tokenizer, max_length, max_length_obj)
