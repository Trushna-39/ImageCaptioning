# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:49:05 2020

@author: Trushna
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:44:39 2020

@author: Trushna Patel
"""

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense,Flatten
from keras.layers import LSTM,Reshape
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras_radam import RAdam
import pydot
import graphviz
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
  result = session.run(c)
  print(result)
# Runs the op.
#print(sess.run(c))

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
    return list(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        #print(line[0],line[1],line[2:])
        tokens = line.split()
        #print(tokens[1:])
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
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

def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    #print("features",features)
    return features

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def max_length_Obj(objects):
    lines = to_lines(objects)
    l=[]
    for i in range(len(lines)):
        l.append(len(lines[i]))
    return max(l)
    #return max(len(d[0].split()) for d in lines)

def create_sequences(tokenizer, max_length, max_length_obj, descriptions, photos, objects):
    X1, X2, X3, y = list(), list(), list(), list()
    for key, desc_list in objects.items():
        #print(key,desc_list)
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            seq_obj = tokenizer.texts_to_sequences([desc_list[0][0]])[0]
            seq_obj2 = tokenizer.texts_to_sequences([desc_list[0][1]])[0]
            #print(len(desc))
            for i in range(2, len(seq)):
                obj_seq = pad_sequences([seq_obj,seq_obj2], maxlen=max_length_obj)[0]
                #print("dsf",obj_seq)
                in_seq, out_seq = seq[:i], seq[i]
                #print(in_seq, out_seq)
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                #in_seq = pad_sequences([in_seq], maxlen=4096)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                #print(out_seq)
                X1.append(photos[key][0])
                X3.append(in_seq)
                y.append(out_seq)
                X2.append(obj_seq)
    print(len(X1))
    print(len(X2))
    print(len(X3))
    print(len(y))
    return array(X1), array(X2), array(X3), array(y)

def define_model(vocab_size, max_length, max_length_obj):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2=Input(shape=(max_length_obj,))
    #in1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    in2=Dropout(0.5)(inputs2)
    in3 = Dense(256, activation='relu')(in2)
    inputs3 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs3)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, in3, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train = list(load_set(filename))
print('Dataset: %d' % len(train))

#train_descriptions = load_clean_descriptions('Objects.txt', train)
#print('Descriptions: train=%d' % len(train_descriptions))

train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

objects = load_object('Train.txt', train)
print('Full Descriptions: train=',len(objects))

train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

tokenizer = create_tokenizer(objects)
#tok=create_tokenizer(objects)
#obj_vocab=len(tok.word_index)+1
#print("Objects Vocabulary:%d"%obj_vocab)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

max_length_obj = max_length_Obj(objects)
print('Objects Length: %d' % max_length_obj)

X1train, X2train, X3train, ytrain = create_sequences(tokenizer, max_length, max_length_obj, train_descriptions, train_features, objects)

filename = 'Flicker8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))

test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

objects_test = load_object('Val.txt', test)
print('Objects: test=',len(objects_test))
 
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

X1test, X2test, X3test, ytest = create_sequences(tokenizer, max_length, max_length_obj, test_descriptions, test_features, objects_test)

#X2train=X2train.reshape(max_length_obj,1).T
model = define_model(vocab_size, max_length, max_length_obj)
#model.save('model.h5')
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit([X1train, X2train, X3train], ytrain, epochs=21, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test, X3test], ytest))
