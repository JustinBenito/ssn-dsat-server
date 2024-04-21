
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation
import fleep
import shutil
import moviepy.editor as moviepy

app = FastAPI()
import os
from flask import Flask, abort,flash, render_template, request, send_from_directory
from fileinput import filename
import os
from werkzeug.wsgi import FileWrapper

from disvoice.phonation import Phonation
from werkzeug.utils import secure_filename
import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import opensmile
from Signal_Analysis.features.signal import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import matplotlib
import crepe
from sklearn.preprocessing import LabelEncoder

import torch
from torchvision import models
import os
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from PIL import Image
import webbrowser
import parselmouth
import seaborn as sns
import tempfile
from fastapi.middleware.cors import CORSMiddleware


import parselmouth
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
# audio_path = r"audios/Adithya-_a-Edited.wav"
# sound = parselmouth.Sound(audio_path)

# # Compute energy
# energy = sound.to_intensity()
# s={"e":list(np.array(energy.values)[0])}
# print(s)


# # Plot the waveform, F0 curve, and energy
# plt.figure(figsize=(10, 6))

# # Plot waveform
# plt.subplot(3, 1, 1)
# plt.plot(sound.xs(), sound.values.T)
# plt.title('Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# # Plot F0 curve
# plt.subplot(3, 1, 2)
# pitch = sound.to_pitch()
# print(list(np.array(pitch.selected_array['frequency'])))
# f0_values = pitch.selected_array['frequency']
# f0_values[f0_values == 0] = np.nan
# plt.scatter(pitch.xs(), f0_values, color='orange', marker='.')
# plt.title('F0 curve')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.ylim(0, 500)

# # Plot energy
# plt.subplot(3, 1, 3)
# plt.plot(energy.xs(), energy.values.T, color='green')
# plt.title('Energy')
# plt.xlabel('Time (s)')
# plt.ylabel('Energy (dB)')
# plt.ylim(0, np.nanmax(energy.values))  # Find the maximum value excluding NaNs

# plt.tight_layout()
# plt.show()




basedir = os.path.abspath(os.path.dirname(__file__))
file_name='/Users/justinbenito/Downloads/webapp/audios/Adithya-_a-Edited.wav'

# with open(file_name,'r') as f:
d = []
phonation = Phonation()
articu = Articulation()
print("PH is coming")
ph=phonation.extract_features_file(file_name)
print(ph)

print("Arti is coming")
ah=articu.extract_features_file(file_name)
print(ah)

# matplotlib.pyplot.savefig(os.path.join(basedir, 'app/static/images', 'plot.jpg')) 
# dt = ph.to_dict('records')
# p=dt[0]
# avg_df0 = p['avg DF0']
# avg_ddf0 = p['avg DDF0']
# avg_jitter = p['avg Jitter']
# avg_shimmer = p['avg Shimmer']
# avg_apq = p['avg apq']
# avg_ppq = p['avg ppq']
# avg_logE = p['avg logE']
# phon = pd.DataFrame.from_dict(d, "columns")
# #mfcc (40) diemensional feature extraction
# y, sr = librosa.load(file_name)
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max),x_axis='time',y_axis='mel',fmax=8000)
# plt.savefig(os.path.join(basedir, 'app/static/images', 'mel_testanalyse.jpg'))

# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.ComParE_2016,
#     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
# )
# op = smile.process_file(file_name)
# s = op.to_dict('records')
# opsmile = s[0]

# hnr = get_HNR( y, sr )
# f0 =  get_F_0( y, sr )

# f0 = f0[0]    
# feat={
#         'avg DF0' :[ p['avg DF0']],
#         'avg DDF0' :[p['avg DDF0']],
#         'avg Jitter' : [p['avg Jitter']],
#         'avg Shimmer' : [p['avg Shimmer']],
#         'avg apq' :[p['avg apq']],
#         'avg ppq' : [p['avg ppq']],
#         'avg logE' : [p['avg logE']],
#         'pcm_fftMag_spectralHarmonicity_sma' :[opsmile['pcm_fftMag_spectralHarmonicity_sma']],
#         'pcm_fftMag_spectralVariance_sma' : [opsmile['pcm_fftMag_spectralVariance_sma']],
#         'pcm_fftMag_spectralCentroid_sma' : [opsmile['pcm_fftMag_spectralCentroid_sma']],
#         'filename': file_name
# } 
# print(feat)




# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:5173",
    
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# basedir = os.path.abspath(os.path.dirname(__file__))
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.post("/extract")
# async def extract_file(file: UploadFile):
#     try:
#         os.mkdir("audios")
#         print(os.getcwd())
#     except Exception as e:
#         print(e) 
#     file_name = os.getcwd()+"/audios/"+file.filename.replace(" ", "-")

#     with open(file_name,'wb+') as f:
#         f.write(file.file.read())
#         f.close()

#     phonation = Phonation()
#     ph=phonation.extract_features_file(file_name, static=True, plots=False, fmt="dataframe")
#     matplotlib.pyplot.savefig(os.path.join(basedir, 'app/static/images', 'plot.jpg')) 
#     dt = ph.to_dict('records')
#     p=dt[0]

#     y, sr = librosa.load(file_name)

#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
#     librosa.display.specshow(librosa.power_to_db(S, ref=np.max),x_axis='time',y_axis='mel',fmax=8000)
#     plt.savefig(os.path.join(basedir, 'app/static/images', 'mel_testanalyse.jpg'))

#     smile = opensmile.Smile(
#         feature_set=opensmile.FeatureSet.ComParE_2016,
#         feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
#     )
#     op = smile.process_file(file_name)
#     s = op.to_dict('records')
#     opsmile = s[0]

#     hnr = get_HNR( y, sr )
#     f0 =  get_F_0( y, sr )

#     f0 = f0[0]    
#     feat={
#             'avg DF0' :[ p['avg DF0']],
#             'avg DDF0' :[p['avg DDF0']],
#             'avg Jitter' : [p['avg Jitter']],
#             'avg Shimmer' : [p['avg Shimmer']],
#             'avg apq' :[p['avg apq']],
#             'avg ppq' : [p['avg ppq']],
#             'avg logE' : [p['avg logE']],
#             'pcm_fftMag_spectralHarmonicity_sma' :[opsmile['pcm_fftMag_spectralHarmonicity_sma']],
#             'pcm_fftMag_spectralVariance_sma' : [opsmile['pcm_fftMag_spectralVariance_sma']],
#             'pcm_fftMag_spectralCentroid_sma' : [opsmile['pcm_fftMag_spectralCentroid_sma']],
#             'filename': file_name
#     } 
#     return {"features":feat}
    


# @app.post("/analyse")
# async def create_upload_file(file: UploadFile):
#     try:
#         os.mkdir("audios")
#         print(os.getcwd())
#     except Exception as e:
#         print(e) 
#     file_name = os.getcwd()+"/audios/"+file.filename.replace(" ", "-")
#     with open(file_name,'wb+') as f:
#         f.write(file.file.read())
#         f.close()
#         resnet_model = models.resnet50()  # Change the model architecture if needed
#         model_path = os.path.join(basedir, 'app/static/model', 'resnet_model.pth')
#         # Load the model state dictionary
#         model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

#         # Create an instance of the ResNet model
#         model = models.resnet50(pretrained=True)  # Specify pretrained=False if your model was not pretrained
#         model.fc = torch.nn.Linear(in_features=2048, out_features=4)  # Adjust the number of output classes
#         model.load_state_dict(model_state_dict)
#         model.eval()

#         y, sr = librosa.load(file_name)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
#         img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),x_axis='time',y_axis='mel',fmax=8000)
#         plt.savefig(os.path.join(basedir, 'app/static/images', 'mel1.jpg'))

#         def preprocess_image(image_path):
#             transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])
#             image = Image.open(image_path)
#             input_tensor = transform(image)
#             input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
#             return input_batch
#         # Mapping between class indices and labels (replace this with your actual mapping)
#         class_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Normal', 3: 'Severe'}

#         # Example of making predictions
#         image_path = os.path.join(basedir, 'app/static/images', 'mel1.jpg')
#         input_data = preprocess_image(image_path)
#         with torch.no_grad():
#             output = model(input_data)

#         # You may need to apply additional post-processing depending on your model's output
#         # For example, if it's a classification task, you might want to apply softmax
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)

#         # Get the predicted class index
#         predicted_class = torch.argmax(probabilities).item()
#          #Decode the class index to label
#         predicted_label = class_mapping.get(predicted_class, 'Unknown')
#     return {"message":f"Success 200 {predicted_class}"}
#     # try:
#     #     with open(file.filename, 'wb') as f:
#     #         shutil.copyfileobj(file.file, f)
#     # except Exception:
#     #     return {"message": "There was an error uploading the file"}
#     # finally:
#     #     file.file.close()
        
#     # return {"message": f"Successfully uploaded {file.filename}"}
#     # try:
#     #     with open(file.filename, 'wb') as f:
#     #         a= shutil.copyfileobj(file.file, f)
#     #         phonation = Phonation()
#     #         ph=phonation.extract_features_file(file, static=True, plots=False, fmt="dataframe")
#     #         return {"file": f"{file.filename}"}
#     # except Exception as e:
#     #     return {"message": f"There was an error uploading the file {e}"}
#     # finally:
#     #     file.file.close()
        
#     # return {"message": f"Successfully uploaded {file.filename}"}
#     # contents = myfile.file.read()
#     # with open(contents, "rb") as f:
#     #     print(f)
#     #     info = fleep.get(f)
#     # phonation = Phonation()
#     #ph=phonation.extract_features_file(file, static=True, plots=False, fmt="dataframe")
#     # return {"filename": info.type}