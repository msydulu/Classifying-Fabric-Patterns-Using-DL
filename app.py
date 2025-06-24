import os
import keras 
from keras.models import load_model
import streamlit as st
import tensorflow as tf

st.header('fabric classification cnn model')

Fabric_name = ['chequered', 'paisley', 'plain', 'polka-dotted', 'striped', 'zigzagged']

model = load_model(r'C:\Users\sydul\Downloads\Flower Dataset\Flower_Recog_Model_cnn.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The image belongs to ' +  Fabric_name[np.argmax(result)] + ' with a score of ' + str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('upload an image')
if uploaded_file is not None:
    with open(os.path.join('upload',uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())





