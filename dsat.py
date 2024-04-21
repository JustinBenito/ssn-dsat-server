from disvoice.phonation import Phonation
import os
import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
#Phonation features
def phonation(file):
    d = []
    f = file
    phonation = Phonation()
    f=phonation.extract_features_file(f, static=True, plots=True, fmt="dataframe")
    p = f.to_dict('records')
    d.append(p[0])
    phon = pd.DataFrame.from_dict(d, "columns")
    phon.to_csv("phonation.csv",index=False)
#mfcc feature
def mfcc(file):
    d =[]
    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    data = {}
    for i in range(0,len(mfcc)):
        try:
            data.update({"mfcc_"+str(i):mfcc[0][i]})
        except IndexError:
            pass
    d.append(data)
    mfc = pd.DataFrame.from_dict(d, "columns")
    mfc.to_csv("mfcc40.csv",index = False)
#hnr feature


