import pickle
import pandas as pd 
import os 

with open('all_data.pkl', 'rb') as file:
    all_data = pickle.load(file)
print('Total records', len(all_data))
#pickle.load(file) - takes the saved data: all_data.pkl and brings it back into Python.So now: all_data is available inside create_dataset.py.

rows = []
for item in all_data:
    row = {'file_name': os.path.basename(item['filepath']),'label' :item['label'] , 'spectral_centroid': item['spectral_centroid'], 'zcr': item['zcr'], 'rms':item['rms']} # basepath(file_name) - 1766-0-0.wav
# label - This is very important because label is what our ML model will eventually predict. Possible labels:Animal, Car_Sound, Gunshot, Natural_Sound, Timber_Cutting.
# spectral, zcr, rms is calculated that is added as column, and values as rows.

    for i, value in enumerate(item['mfcc'], start =1):
        row[f'MFCC_{i}'] = value 
    rows.append(row)
# mff has 20 values, that 20 values must be converted to column names, so we are doing this
#MFCC = [-390.90, 26.15, -33.76, 9.77, ... , 7.60] to 
# MFCC_1  → -390.90
#MFCC_2  → 26.15
#MFCC_3  → -33.76
#MFCC_4  → 9.77
#...
#MFCC_20 → 7.60

# Creating the dataframe
data = pd.DataFrame(rows)
print('Data frame shape:' , data.shape)
print(data.head())

# save the csv
data.to_csv('Bioacoustic_threat_Detection.csv', index = False)
print('cvs created successfully')






