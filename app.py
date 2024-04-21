
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
#matplotlib.use('TkAgg') 

app = Flask(__name__,template_folder='app/templates',static_folder='app/static')
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'wav'}
model_folder = 'app/static/model'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['model_folder'] = model_folder

image_folder = 'app/static/images'
app.config['image_folder'] = image_folder
basedir = os.path.abspath(os.path.dirname(__file__))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config["CACHE_TYPE"] = "null"
@app.route('/')




def index():
    return render_template('index1.html')

@app.route('/output', methods=['POST'])
def upload_file():
   # matplotlib.use('agg')
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        f.save(f.filename)
        f.flush()
        f.close()
        d = []
        data = os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename)
        #extraction of phonation features
        phonation = Phonation()
        # ph=phonation.extract_features_file(data, static=True, plots=False, fmt="dataframe")
        # matplotlib.pyplot.savefig(os.path.join(basedir, app.config['image_folder'], 'plot.jpg')) 
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
        # y, sr = librosa.load(data)
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        # librosa.display.specshow(librosa.power_to_db(S, ref=np.max),x_axis='time',y_axis='mel',fmax=8000)
        # plt.savefig(os.path.join(basedir, app.config['image_folder'], 'mel.jpg'))
        # # m={}
        # # for i in range(0,len(mfcc)):
        # #     try:
        # #         m.update({"mfcc_"+str(i):[mfcc[0][i]]})
        # #     except IndexError:
        # #         pass
        # # mfc = pd.DataFrame.from_dict(m)
        # smile = opensmile.Smile(
        #     feature_set=opensmile.FeatureSet.ComParE_2016,
        #     feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        # )
        # op = smile.process_file(data)
        # s = op.to_dict('records')
        # opsmile = s[0]
        # pcm_fftMag_spectralHarmonicity_sma =opsmile['pcm_fftMag_spectralHarmonicity_sma']
        # pcm_fftMag_spectralVariance_sma = opsmile['pcm_fftMag_spectralVariance_sma']
        # pcm_fftMag_spectralCentroid_sma = opsmile['pcm_fftMag_spectralCentroid_sma']
    
        # hnr = get_HNR( y, sr )
        # f0 =  get_F_0( y, sr )
        # #print(f0)
        # f0 = f0[0]    
        # #print(opsmile)  
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
                

        # } 
        
        # # Assuming you have a ResNet model architecture
        # resnet_model = models.resnet50()  # Change the model architecture if needed
        # model_path = os.path.join(basedir, app.config['model_folder'], 'resnet_model.pth')
        # # Load the model state dictionary
        # model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # # Create an instance of the ResNet model
        # model = models.resnet50(pretrained=True)  # Specify pretrained=False if your model was not pretrained
        # model.fc = torch.nn.Linear(in_features=2048, out_features=4)  # Adjust the number of output classes
        # model.load_state_dict(model_state_dict)
        # model.eval()  # Set the model to evaluation mode

        # # Example of preprocessing for an image
        # def preprocess_image(image_path):
        #     transform = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        #     image = Image.open(image_path)
        #     input_tensor = transform(image)
        #     input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
        #     return input_batch
        # # Mapping between class indices and labels (replace this with your actual mapping)
        # class_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Normal', 3: 'Severe'}

        # # Example of making predictions
        # image_path = os.path.join(basedir, app.config['image_folder'], 'mel.jpg')
        # input_data = preprocess_image(image_path)

        # with torch.no_grad():
        #     output = model(input_data)

        # # You may need to apply additional post-processing depending on your model's output
        # # For example, if it's a classification task, you might want to apply softmax
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # # Get the predicted class index
        # predicted_class = torch.argmax(probabilities).item()
        #  #Decode the class index to label
        # predicted_label = class_mapping.get(predicted_class, 'Unknown')
        # print(f"Predicted class index: {predicted_class}")
        # print(f"Predicted label: {predicted_label}")
        return render_template('index1.html', name=f.filename,ph=p,mfc=m,hnr=hnr,f0=f0,res=predicted_label,variance= pcm_fftMag_spectralVariance_sma, centroid = pcm_fftMag_spectralCentroid_sma,harmonicity = pcm_fftMag_spectralHarmonicity_sma)

@app.route('/analysis', methods=['POST','GET'])
def analysis():
    print("hello")
    if request.method == 'POST':
        f = request.files['audio']
        
        filename = secure_filename(f.filename)
        #filename = f.filename
        f.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
        f.save(f.filename)
        f.flush()
        

        data = os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename)
        print('File uploaded successfully')
        sound = parselmouth.Sound(data)
  
  
        sns.set() # Use seaborn's default style to make attractive graphs
        plt.rcParams['figure.dpi'] = 100 # Show nicely large images in this notebook
        y, sr = librosa.load(data)
        #plt.figure(figsize=(12, 4))
        waveform =  librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.savefig(os.path.join(basedir, app.config['image_folder'], 'wav.jpg'))
        #plt.show()
        # Estimate pitch using crepe
        time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True)

        # Plot pitch contour
        plt.figure(figsize=(10, 4))
        plt.plot(time, frequency, color='b')
        plt.title('Pitch Contour')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.show()
        return render_template('analysis.html',res1=waveform)
    return render_template('analysis.html')
    
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print("An error occurred:", e)
    # Re-raise the exception to see the full traceback