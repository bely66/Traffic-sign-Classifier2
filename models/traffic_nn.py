#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:38:31 2019

@author: mohamed
"""
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dropout , Dense , Flatten , BatchNormalization
from tensorflow.keras.layers import Activation
class Traffic_Classifier():
    @staticmethod
    def build(height,width,depth,classes):
        model = Sequential()
        chanDim = -1
        model.add(Conv2D(8,(5,5),padding="same",input_shape=(height, width, depth)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #first set of (CONV ==> RELU ==> BN)*2 ==> MaxPooling
        model.add(Conv2D(16,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #second set of (CONV ==> RELU ==> BN)*2 ==> MaxPooling

        model.add(Conv2D(32,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        # the MLP 
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model



        