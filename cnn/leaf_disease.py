import streamlit as st 
import tensorflow as tf 
import numpy as np 
import os 
from PIL import Image

st.set_page_config(page_title = 'Tomato disease Classification',page_icon = '🍅🌿', layout = 'wide')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'C:\Users\POOJA\Downloads\datascience\deep_learnig\cnn\tomato_leaf_disease_detection.keras')

model = load_model()

class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

st.sidebar.title('tomato Leaf disease')

page = st.sidebar.radio('Select Page',['Prediction'])

if page == 'Prediction':
    st.title('tomato leaf Disease Prediction')
    
    uploaded_file = st.file_uploader('Upload a tomato leaf image', type =['jpg','jpeg','png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption = 'Uploaded tomato Leaf', width = 400)
        
        if st.button('Predict Disease'):
            image = image.resize((128,128))
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis = 0)
            predictions = model.predict(img_array)
            score = predictions[0]
            predicted_class = np.argmax(score)
            predicted_name = class_names[predicted_class]
            confidence = score[predicted_class]*100
            
            st.success(f'Prediction:{predicted_name}')
            st.info(f'Confidence:{confidence:.2f}')