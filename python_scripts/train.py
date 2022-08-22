# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:27:25 2021

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

def train_model(client_no):
    client = str(client_no)
    
    #Paths to Dataset Directory 
    Directory_Path = 'C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning' 
    
    path_to_tensorflow = Directory_Path+"/Models/Tensorflow_Models/client"+client+".h5"
    
    #Paths for Training and Validation
    train_path = Directory_Path+'/Images/client_'+client+'/train'
    valid_path = Directory_Path+'/Images/client_'+client+'/valid'
    
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=train_path, target_size=(224,224), class_mode='categorical', batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=valid_path, target_size=(224,224), class_mode='categorical', batch_size=10)
    
    # transfer learning - MobileNet
    
    #Loading the MobileNet model through API
    mobile = tf.keras.applications.mobilenet.MobileNet()
    
    #Fine Tuning Model Using Transfer Learning
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)
    
    #model.summary()
    
    for layer in model.layers[:-5]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches,
              steps_per_epoch=len(train_batches),
              validation_data=valid_batches,
              validation_steps=len(valid_batches),
              epochs=10,
              verbose=1,use_multiprocessing = False
    )
    model.save(path_to_tensorflow)
    
    return path_to_tensorflow

def test_Tfmodel(client):
    model = keras.models.load_model('C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Tensorflow_Models/client'+client+'.h5')
    test_path = 'C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Images/client_'+client+'/test'
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
    #Testing the Local Model Tensorflow Model
    test_labels = test_batches.classes
    #print("Test Labels",test_labels)
    #print(test_batches.class_indices)
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
    #print(answer)

def convert_to_tflite(client):
    model = keras.models.load_model('C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Tensorflow_Models/client'+client+'.h5')
    path_to_tflite = "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Tensorflow_Lite_Models/client"+client+".tflite"
    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TF Lite model.
    with tf.io.gfile.GFile(path_to_tflite, 'wb') as f:
      f.write(tflite_model)
    #Create Label.txt for TFlite Interpreter
    with open("C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Tensorflow_Lite_Models/labelmap.txt","w") as f :
      f.write("face \nnot_face")

    
#Function to test TFlite Model
def test_TfLite(client):
      
    #Object Detection for Tensorflow Lite Model (Divded into respective  Code Blocks)
    
    MODEL_NAME = "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Models/Tensorflow_Lite_Models/"
    GRAPH_NAME = "client"+client+".tflite"
    LABELMAP_NAME = "labelmap.txt"
    min_conf_threshold = 0.7
    IM_DIR = "C:/Users/kanan/Desktop/BlockChain/BFLC/Federated_Learning/Images/server/"
    
    images = sorted(glob.glob(IM_DIR+'*.jpg'))
    
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(MODEL_NAME,GRAPH_NAME)
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(MODEL_NAME,LABELMAP_NAME)
    
    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    #Load the TFLite Interpreter
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()
      
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5
    count = 0
    x = 0
    #Perform the testing on TFlite Model over the loop of images
    for image_path in images:
        # Load image and resize to expected shape [1xHxWx3]
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
      
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
        predict_label = np.argmax(output()[0])
        
        if output()[0][predict_label] > min_conf_threshold:
            count += 1
        else:
            #If detection is wrong made it will indicate "wrong detection"
            x +=1
  
    #TFLite Model Accuracy for 100 images (50-Faces and 50-Non_Faces)        
    print("\n Total TFLite Model Accuracy(in %) for 100 images : ", count)   

#train_model("1")
#test_tfmodel("1")
#convert_to_tflite("1")
#test_TfLite("1")


