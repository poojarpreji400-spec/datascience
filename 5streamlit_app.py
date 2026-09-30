import streamlit as st # streamlit - library to create webpage
import base64 # We need this because we want to put our local image into the webpage background, The image is converted into a format that the webpage can display.
import os 
import librosa # reads the uploaded audio and extracts audio features.
import numpy as np 
import joblib # loads your trained Random Forest model and LabelEncoder.
import tempfile # temporarily saves the uploaded audio so librosa can read it.

def set_background(image_file): # creating our own fn called set_background
    with open(image_file, 'rb') as file:
        # Images are binary files, so we read the image as binary data. r= read, b= binary
        image = base64.b64encode(file.read()).decode()
        # This converts the image into a form that can be inserted into the webpage.
        
        # CSS = controls how the webapage looks
        # triple quotes helps to write the code in seperate line(if single quotes is used we need\n)
        # <style> - used to add css styling.
        # .stApp - to identify which part of the page change is needed(st - streamlit, App - application)
        # inside f string double curly brackets is used (meaning - I actually want a normal { character.). If not used f string considers that as something to print and causes error when variable name not found.
        # url('data:image/png;base64,{image}') - Use this PNG image as the background of my webpage
        # data: - The image data is being provided directly here, instead of giving a normal file location.
        # url() tells CSS:Here is where the image comes from.
        # base64: The image data has been converted into Base64 format.
        page_background = f'''  
        <style>
        .stApp{{
            background-image: url('data:image/png;base64,{image}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            }}
        </style>
        '''
        # background-image - put the image behind the webpage.
        # background-size: cover- Makes the image cover the entire screen.
        # background-position: center - Keeps the important part of the image centered.
        # background-attachment: fixed - Keeps the background fixed while you scroll.
        
        st.markdown(page_background, unsafe_allow_html = True) # Streamlit, display the HTML/CSS stored inside page_background on my page.
        # markdown - Display this content on the webpage.
        # unsafe_allow_html - Allow HTML/CSS code inside this content to actually work.
        
st.set_page_config(page_title = 'Bioacoustic Threat Detection', page_icon = '🔊🔊', layout = 'wide')
image_path = os.path.join(os.path.dirname(__file__), 'background.png')
set_background(image_path)

# load trained model
model_path = os.path.join(os.path.dirname(__file__), 'rf_model1.pkl')
encoder_path = os.path.join(os.path.dirname(__file__),'label_encoder.pkl')

rf_model = joblib.load(model_path)
le = joblib.load(encoder_path)

# feature extraction
def extract_features(audio_file):
    y,sr = librosa.load(audio_file, sr = 22050, mono = True)
    
    spectral_cen = librosa.feature.spectral_centroid(y=y, sr = sr)
    spectral_mean = np.mean(spectral_cen)
    
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    rms = librosa.feature.rms(y = y)
    rms_mean = np.mean(rms)
    
    mfcc = librosa.feature.mfcc(y = y, sr= sr, n_mfcc = 20)
    mfcc_mean = np.mean(mfcc, axis = 1)
    
    new_features = np.concatenate([[spectral_mean], [zcr_mean],[rms_mean],mfcc_mean])
    return new_features # Total 23 features

# sidebar
st.sidebar.title('🔊 Bioacoustic Detection')
page = st.sidebar.radio(
    'Go to',
    ['🏠 Home','🌳 Prediction','ℹ️ About']
    )

# home page
if page == '🏠 Home':
    st.title('🔊 Bioacoustic Threat & Poaching Detection 🔊')
    st.write('This application uses machine Learning to classify audio ' 
                'into different sound Categories')# st.write() displays text on the webpage.
    st.subheader('🌳 About the Project')
    st.write('This application analyses bioacoustic recordings'
                'and predict the type of sound using a Random Forest Model')

elif page == '🌳 Prediction':
    st.title('🎶Audio Prediction')
    st.header('Upload an Audio File Please')
    uploaded_file = st.file_uploader('Upload an audio file', type = ['wav', 'mp3','ogg'])

    if uploaded_file is not None:
        st.audio(uploaded_file)
    
        st.success('🥳 Audio file uploaded successfully!')
    
        predict_button = st.button('Predict sound❓')
    
        if predict_button:
            # creating a temporaray file - Because the user uploads the audio through Streamlit, but we want to give the audio file to: librosa.load()
            with tempfile.NamedTemporaryFile(delete = False, suffix = '.wav') as temp_file: # we create a temp file
                temp_file.write(uploaded_file.read()) # reads the uploaded file and writes it into temporrary file.
                temp_path = temp_file.name # python create a location in system to save the temporary path
        
    # show a loading message
            with st.spinner('🤖 Analysing the Audio...'):
                # sxtract 23 features
                new_features = extract_features(temp_path)
                new_features = new_features.reshape(1,-1) # befor - (23,), After - (1, 23)
        
                prediction = rf_model.predict(new_features) 
                predicted_class = le.inverse_transform(prediction)[0]
        
            st.subheader('🐢 Prediction')
            st.success(f'Predicted Sound: **{predicted_class}**')
        
            os.unlink(temp_path) # delete the temp file we dont need this audio after prediction

elif page == 'ℹ️ About':
    st.title('ℹ️ About the Project')
    st.write('This project uses machine learning to classify different evnironmental and threat-related audio signals')
    st.subheader('Sound Categories trained in the project.')
    st.write('The model classifies audio into 5 categories:')
    st.markdown('''
                1) Animal
                2) Car Sound
                3) Gunshot
                4) Natural Sound
                5) Timber Cutting
                ''')
    st.subheader('The Audio Features')
    st.write('The model uses the following audio features to predict:')
    st.markdown(''' 
                1) Spectral Centroid
                2) Zero Crossing Rate
                3) RMS Energy
                4) 20 MFCC Features
                ''')
    
        