# from pyexpat import model
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import  tensorflow.python.trackable.data_structures
import numpy as np
import os
import h5py

st.header("Driver Drowsiness Detection")
    
        
def predicted_class(img):
    model = tf.keras.models.load_model(r'D:\sem V\DL\Proj\model\model_resnet.h5')
    # shape()
    # model = tf.keras.Sequential(hub[hub.KerasLayer(classifier,(224,224,3))])
    test = img.resize((224,224))
    test = preprocessing.img.img_to_array(test)
    test = test/255.0
    test = np.expand_dims(test,axis=0)
    class_name = ['Open Eyes', 'Close Eyes' ]
    predictions = model.predict(test)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_name[np.argmax(scores)]
    result = "The Image is {}".format(image_class)
    return result
    
file_uploaded = st.file_uploader("Choose the file")
if file_uploaded is not None:
        img = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(img)
        plt.axis('off')
        result = predicted_class(img)
        st.write(result)
        st.pyplot(figure)