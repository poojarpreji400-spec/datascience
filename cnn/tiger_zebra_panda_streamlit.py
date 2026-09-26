import streamlit as st 
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np 
import os 

st.set_page_config(page_title = 'Classification of Images of Tiger, Zebra and Pandas', page_icon ='🐅🦓🐼', layout = 'wide')

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go To', ['Home', 'Prediction'])

@st.cache_resource
def load_model():
    model_path = r'C:\Users\POOJA\Downloads\datascience\deep_learnig\cnn\tiger_zebra_pands.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None
model = load_model()

if page == 'Home': # if the person selects home
    st.title('🐅🦓🐼 Classifier')
    st.markdown('''Upload an image and the CNN model will predict whether the image is a Cat or a Dog.''')
    
elif page == 'Prediction':
    st.title('Predict')
    st.write('Upload an image')
    
    if model is None:
        st.warning('Model not Found')
        
    else:
        uploaded_file = st.file_uploader('Upload a image...', type =['jpg','png', 'jpeg'])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption = 'Your Uploaded Image', width = 300)
            
            if st.button('Predict'):
                with st.spinner('Predicting...'):
                    img_resized = img.resize((150,150))
                    img_array = image.img_to_array(img_resized) # from tensorflow.keras.preprocessing import image
                    img_array = np.expand_dims(img_array, axis = 0)
                    
                    predictions = model.predict(img_array)
                    
                    score = tf.nn.softmax(predictions[0])
                    
                    class_names = ['Panda', 'Tiger', 'Zebra']
                    
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100*np.max(score)
                    
                    st.success(f'This is {predicted_class}')
                    st.info(f'confidenc:{confidence:.2f}')
                    