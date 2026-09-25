import streamlit as st
import tensorflow as tf 
from tensorflow.keras.preprocessing import image # Preparing images for a TensorFlow/Keras model
from PIL import Image # pillow - PIL - Opening/manipulating image files
import numpy as np 
import os 

# set page confurigation
st.set_page_config(page_title = 'cat vs Dog Classification', page_icon = '😺🐶', layout = 'wide')
# st - streamlit, page_title - title of the browser tab., layout='centered' ➡️ Content stays in a narrower, centered area.,layout='wide'➡️ Content uses more of the screen width.(make the webpage use the full available width)

# side bar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go To', ['Home', 'Prediction']) # radio() creates radio-button choices where the user can select one option.

# model function
@st.cache_resource # Load this model once and reuse it instead of loading it again every time the app changes
def load_model():
    model_path = r'C:\Users\POOJA\Downloads\datascience\deep_learnig\cnn\animal_cat_dog.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None
model = load_model()

if page == "Home":
    st.title('😺🐶Cat vs Dog Classifier')
    st.markdown('''
    ### Welcome to the Cat Vs Dog Classification App!. 
    This application uses a Convolutional Neural Network(CNN) built with Tensorflowand Keras to classify images of Ctas and Dogs.
    ### navigation Guide: 
    - **Home** : Overview of the Application
    - **Prediction** : Upload a single image and let the trained model predict whether it's a Cat or a Dog.
    
    Use the side bar on the left to navigate through the app.''')

elif page == 'Prediction': # if the user click prediction button then
    st.title('Prediction')
    st.write('Upload an image to predict whether it is a cat or a Dog.')
    
    if model is None:
        st.warning('**Model not Found!** Please ensure you have run the jupyter Notebook to train and save the animal_cat_dog.keras model in this directory')
    else:
        uploaded_file = st.file_uploader('Upload an image please....', type = ['jpg', 'png','jpeg'])
        
        if uploaded_file is not None:
            # display the uploaded image
            img = Image.open(uploaded_file) # here is pillow used
            st.image(img, caption = 'Your Uploaded Image', width = 300)
        
            if st.button('Predict'): # Wait for the user to click Predict
                with st.spinner('Predicting.....'):
                    img_resized = img.resize((150,150)) # the model already expects 150*150, but the user can upload any images, so we need to reshape it to 150*150
                    img_array = image.img_to_array(img_resized) # the computers represent the pixels as num.
                    img_array = np.expand_dims(img_array, axis = 0) # it changes to 1*150*150*3
                    
                    # make predictions
                    
                    predictions = model.predict(img_array)
                    
                    # matching the notebook implemetation
                    score = tf.nn.softmax(predictions[0])
                    # Raw output: [2.1, 0.7], softmax changes to [0.80, 0.20]
                    # This converts the raw scores into values that behave like probabilities.
                    class_names = ['Cat', 'Dog'] 
                                                  
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100* np.max(score)
                    
                    st.success(f'### This ia **{predicted_class}**!')
                    st.info(f'**Confidence:**{confidence:.2f}%')

