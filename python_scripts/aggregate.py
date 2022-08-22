# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:30:34 2021

@author: kanan
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import time 
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def fl_average(model_directory):
    tf_model_directory = model_directory+"/TF_Models/"
    client_models_path = os.listdir(tf_model_directory)
       
    scaled_local_weight_list = []
    
    for path in client_models_path:
        print("Loading Model...")
        print(path)
       
        local_model = tf.keras.models.load_model(tf_model_directory + path)
        local_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        #test_model(local_model,test_dataset_directory)
        local_weights= local_model.get_weights()
        
        #scale the model weights and add to list
        scaling_factor = 1/3
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        
        print(len(local_weights))
        scaled_local_weight_list.append(scaled_weights)
        
    return scaled_local_weight_list

def build_model(weights):
    
    mobile = tf.keras.applications.mobilenet.MobileNet()   
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    global_model = Model(inputs=mobile.input, outputs=output)
    
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(weights)
    print('Total models:',len(weights))
    print(len(average_weights))
    #update global model 
    global_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    global_model.set_weights(average_weights)
    return global_model

def save_agg_model(model, model_directory):
    model_path = model_directory+"/Aggregated_Model/agg_model.h5"
    model.save(model_path)
    print("Model written to storage!")
    return model_path

#def test_Tfmodel(model, test_path):
def test_Tfmodel(model, test_path):
    print(test_path)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
    #Testing the Local Model Tensorflow Model
    test_labels = test_batches.classes
    print("Test Labels",test_labels)
    print(test_batches.class_indices)
    predictions = model.predict(test_batches,steps=len(test_batches),verbose=0)
    acc = 0
    answer = []
    for i in range(len(test_labels)):
        actual_class = test_labels[i]
        if predictions[i][actual_class] > 0.7 : 
            acc += 1
            answer.append(actual_class)
        else:
            answer.append(abs(1-actual_class))
    print("Accuarcy:",(acc/len(test_labels))*100,"%")
    accuracy = (acc/len(test_labels))*100
    return accuracy

'''weights = fl_average()
model = build_model(weights)
save_agg_model(model)'''