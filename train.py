#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:07:48 2019

@author: mohamed
"""

from models.traffic_nn import Traffic_Classifier

import matplotlib 
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os


def load_split(base_path,csvpath):
    data = []
    labels = []
    
    rows = open(csvpath).read().strip().split("\n")[1:]
    
    random.shuffle(rows)
    
    for (i,row) in enumerate(rows):
        if i > 0 and i%1000==0 : 
            print("processed {} images ".format(i))
            
        img_id , image_path = row.strip().split(",")[-2:]
        
        image_path = os.path.sep.join([base_path,image_path])
        image = io.imread(image_path)
        
        image = transform.resize(image,(32,32))
        image = exposure.equalize_adapthist(image,clip_limit=0.1)
        
        data.append(image)
        labels.append(img_id)
        
    data = np.array(data)
    labels = np.array(labels)
    
    return data , labels
    
    


 

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path input to the dataset")
ap.add_argument("-p","--plot",required=True,help="path to output the plot")
ap.add_argument("-m","--model",required=True,help="path to output model")
args = vars(ap.parse_args())

NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64
 
# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]


train_path = os.path.sep.join([args["dataset"],"Train.csv"])

test_path = os.path.sep.join([args["dataset"],"Test.csv"])

print("Loading the training and testing data ")

(X_train,y_train)=load_split(args["dataset"],train_path)

(X_test,y_test)=load_split(args["dataset"],test_path)


X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0


# one-hot encode the training and testing labels

numLabels = len(np.unique(y_train))
y_train = to_categorical(y_train, numLabels)
y_test = to_categorical(y_test, numLabels)
 
# account for skew in the labeled data
classTotals = y_train.sum(axis=0)
classWeight = classTotals.max() / classTotals

aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")
optimizer = Adam(lr=INIT_LR,decay = INIT_LR/(NUM_EPOCHS*0.5))

model = Traffic_Classifier().build(32,32,3,numLabels)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

H=model.fit_generator(aug.flow(X_train,y_train,batch_size=BS),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=X_train.shape[0] // BS,
                    epochs=NUM_EPOCHS,
                    class_weight=classWeight,
                    verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=BS)
print(classification_report(y_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
 
# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

import json

with open('history.json', 'w') as f:
    json.dump(H.history, f)


N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
