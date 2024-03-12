# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:24:27 2020

@author: Kirtan Patel
"""

import cv2
import numpy as np

import os

# file = os.listdir('D:\\MTECH\\Dataset\\Flickr8k\\img')
# images=[]

#def load_doc(filename):
#    file=open(filename)
#    #print(file)
#    text=file.read()
#    file.close()
#    return text
#
#def load_file(filename):
#   image_id=[]
#   for line in filename.split('\n'): 
#       tokens=line.split()
#       image_id.append(tokens[0])
#   return image_id
#
##filename1 = 'F:\MTECH\Dataset\‪Output.txt'
##file = load_doc(filename1)
##train=load_file(file)
## print('Dataset: %d' % len(train))
#images=[]
#filename="C:\\Users\\admin\\Desktop\\testing"
#for i in os.listdir(filename):
#    path="C:\\Users\\admin\\Desktop\\testing\\"+i
#    images.append(path)
#    
#    
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#import os
#dirname = 'Output'
#os.mkdir(dirname)


#for jpg in images:
#    #if jpg not in train:
#    # Loading image
img = cv2.imread('F:\\MTECH\\Dataset\\Flickr8k\\Test_Images\\2675685200_0913d84d9b.jpg')
#print(jpg)
#img=np.expand_dims(img,axis=0)
img = cv2.resize(img, None, fx=0.9, fy=0.9)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
#fp=open("testing_objects.txt","a")
#fp.write("\n"+jpg+" ")
#
#for i in range(len(class_ids)):
#    #labels=(str(classes[class_ids[i]]))
#    #print(jpg,labels)
#    fp.write(""+str(classes[class_ids[i]])+",")
#fp.close()
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        labels="{}: {:.4f}".format(str(classes[class_ids[i]]), confidences[i])
        color = colors[i]
        print(labels)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 12), font, 1.5, color, 2)
    
    
cv2.imshow("Image", img)
    
    
    #print("svckjuavc jamkv")
    #path=os.path.join(dirname , jpg)
    #print(path)
    #path="D:\\Data\\Output\\"
    #cv2.imwrite(path,img)
cv2.waitKey(0)
cv2.destroyAllWindows()


