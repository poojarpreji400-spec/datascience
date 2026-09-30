# converting audio files into features
# Audio file
#   ↓
# Read audio
#   ↓
# Extract characteristics of the sound
#    ↓
# Convert characteristics into numbers
#    ↓
# Create a CSV table
# MFCC describes the characteristics/pattern of a sound in numerical form.

# first coverting  a single audio

import os
import librosa # librosa - used to load audio, extract MFCC, calculate spectral centroid, calculate zero crossing rate, calculate RMS
import numpy as np
import pandas as pd # used to create final rows and columns
import pickle #pickle allows Python to save a Python object into a file.

dataset_path = r'C:\Users\POOJA\Downloads\FSM5'
for label in os.listdir(dataset_path): #os.listdir - Give me everything inside the dataset folder.
    print(label)
class_path = os.path.join(dataset_path, label)
file_path = r'C:\Users\POOJA\Downloads\FSM5\Timber_Cutting\1766-17-2.wav' # we are not taking the entire audio, instead we are first just checking on any one single audio file.
y, sr = librosa.load(file_path, sr = 22050, mono = True)
# here librosa is the libraray coverting the audio, the file _path - is the location of one single audio inside timber_cutting, 
#sr - sampling rate - how many samples is needed per second, here it is given as 22050 for 1 second, (22,050 samples are used to represent 1 second of audio.)
# 22050 - we want all audio files to be processed at the same sampling rate, so we are setting it to 22050. If the audio file is already at 22050, then it will not change anything. If it is at a different sampling rate, then it will convert it to 22050.
# mono - means there is left and right audio, so here using mono it will make it one audio channel
# y - tells us the sound values, sr - tells us how quickly those values occur.
# y = raw audio signal represented by numbers, 1 sec - 220250 samples are created(y = [-0.0041, -0.0057, -0.0014, ...] - like this seperated by commas) then, 5 seconds × 22,050 samples/second = 110,250 samples

print('Number of Samples:', len(y)) # len(y) - tells us how many samples are there in the audio file.
print('Sampling Rate:', sr)
print('Duration:', len(y)/sr, 'seconds')
# The audio contains 110250 samples and we are using 22050 samples per second, so the duration of the audio is 5 seconds.

mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20) # use the mfcc feature inside librosa
print('MFCC Shape', mfcc.shape)
# mfcc - Mel- Frequency Ceptral Coefficients. - A numerical representation of important characteristics of the sound.
#For example, different sounds such as: Animal ,Gunshot, Car ,Timber cutting ,Natural sound have different acoustic characteristics. MFCC helps us represent those characteristics using numbers.
# y - contains all the actual sound signal, mfcc - helps to find the specific characteristics extracted from that sound.
#n_mfcc = 20 - give only 20 MFCC Coefficients 
# shape = (20,216), here 216 means - 216 small time windows/frames of this one audio file. ans 20 means - 20 columns

mfcc_mean = np.mean(mfcc, axis = 1)
print('MFCC Mean Shape:', mfcc_mean.shape)
print('MFCC Mean:', mfcc_mean)
# earlier we got 20 mfcc features(rows) and 216(columns) small time frame of the audio
# axis =1(mean across the columns of each row.), for each row calculate the mean across all 216 time frames(columns), s0 the 20 rows*216 columns(time frames) becomes 20 values.
# then mfcc.mean - this will have 20 values, this is the one that will later become 20 columns.

# MFCC can produce many different coefficients:, that is why we are giving n_mfcc = 20, whereas spectral , zero, rms has only one feature, that is why it cannot be given the num(eg:shape of rms - (1,216))

# Extract Spectral Centroid
# Spectral centroid tells us whether a sound has more low-frequency or high-frequency energy.
# Deep/low sound → lower spectral centroid, Sharp/high sound → higher spectral centroid
spectral_centroid = librosa.feature.spectral_centroid(y = y, sr = sr)
print('Spectral Centroid Shape:', spectral_centroid.shape) # shape = (1,216)
spectral_centroid_mean = np.mean(spectral_centroid)
print('spectral centroid mean:',spectral_centroid_mean ) # the mean of 216 values are taken that is = 3135.662184..
# this mean will become one column

# spectral centroid and MFCC is related to frequency, so sr is used, whereas zcr and rms is not so no sr is needed

#ZCR - Zero CRossing Line
#        /\       /\
#       /  \     /  \
#------/----\---/----\------  Zero line
#     /      \ /      \
    
# Every time the waveform moves from positive → negative or negative → positive, that is a zero crossing.
#Different sounds have different patterns.
# For example: Low/deep sounds → generally fewer zero crossings, Sharp/noisy sounds → generally more zero crossings, So ZCR gives our ML model another useful clue about the type of sound.
zcr = librosa.feature.zero_crossing_rate(y)
print('ZCR Shape:', zcr.shape) #(1, 216)
print('ZCR Mean:', np.mean(zcr)) # 0.16255922670717593 -  finally this value becomes the column

# RMS - Root Mean Square
# RMS - tells us how strong the sound is
#Quiet sound → lower RMS, Strong/loud sound → higher RMS
rms = librosa.feature.rms(y = y)
print('rms Shape:', rms.shape) #(1,216)
print('rms mean:', np.mean(rms)) # 0.09279777

# creating the loop for entire 2,258 audio files

all_data = []
for label in os.listdir(dataset_path):
    class_path_ = os.path.join(dataset_path, label) # Eg: joining animal with dataset
    
    for file_name in os.listdir(class_path_):
        file_path_ = os.path.join(class_path_, file_name) # Eg: joing each audio name with above.
        
        y_, sr_ = librosa.load(file_path_, sr = 22050, mono = True)
        
        # MFCC
        mfcc_ = librosa.feature.mfcc(y = y_, sr = sr_, n_mfcc = 20)
        mfcc_mean_ = np.mean(mfcc_, axis = 1) # axis=1 means you are computing the mean of each row.
        
        # spectral
        spectral_centroid_ = librosa.feature.spectral_centroid(y =y_, sr = sr_)
        spectral_centroid_mean_ = np.mean(spectral_centroid_)
        
        #ZCR
        zcr_ = librosa.feature.zero_crossing_rate(y_)
        zcr_mean_ = np.mean(zcr_)
        
        # Rms
        rms_ = librosa.feature.rms(y = y_)
        rms_mean_ = np.mean(rms_)
    
        all_data.append({'filepath': file_path_, 'label': label, 'mfcc': mfcc_mean_, 'spectral_centroid': spectral_centroid_mean_,'zcr': zcr_mean_,'rms': rms_mean_ ,}) #Each audio file becomes one dictionary/record inside the list.

print('Total Files:', len(all_data))
print(all_data[0]) # show the first item stored in all_data

# to store and save Python object to file, because dataset is created uisng this dataset and dataset is created in the next file
# import pickle
with open('all_data.pkl', 'wb') as file: # creating a file named all_data.pkl, wb - write binary
#w = write → we are creating/saving data into the file.
# b = binary → the data is saved in binary format, which is suitable for Python objects like your all_data list.
    pickle.dump(all_data, file) # takes all_data - which contains the features for all 2,258 audio files, and saves it into all_data.pkl.
print('Features data saved successfully')