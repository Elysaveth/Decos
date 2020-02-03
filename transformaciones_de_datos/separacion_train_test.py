import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models,initializers,regularizers
from tensorflow.keras.layers import BatchNormalization, Concatenate
import pickle
import matplotlib.pyplot as plt
from random import sample
#from scipy import stats

#PREPARACIÓN DATOS
infile = open("datosinterpoladosrecortados.matrix","rb")
data = pickle.load(infile)
CSV=pd.read_csv('ManoriUTCetiqueta.csv',sep=';',header=None)
labels=CSV[2].to_numpy()[0:24670]
patron_dia = CSV[4].to_numpy()[0:24670]

num_datos = len(labels)
train_last_index= int(0.7*num_datos)
train_data = data[0:train_last_index,:,:]
train_labels = labels[0:train_last_index]
train_patron_dia = patron_dia[0:train_last_index]

bloques = int((num_datos - int(0.7*num_datos))/72)
test_index = []
val_index = []

for bloq in range(bloques): 
    val_index = val_index + list(range(train_last_index,train_last_index+48))
    test_index = test_index + list(range(train_last_index+48, train_last_index+72))
    train_last_index= train_last_index+72

if train_last_index < num_datos:
    if train_last_index + 48 < num_datos:
        val_index = val_index+ list(range(train_last_index, train_last_index+48))
        test_index = test_index+ list(range(train_last_index+48, num_datos))
    else:
        val_index = val_index+ list(range(train_last_index, num_datos))
        

val_data= data[val_index,:,:]
val_labels=labels[val_index]
val_patron_dia = patron_dia[val_index]
test_data=data[test_index,:,:]
test_labels=labels[test_index]
test_patron_dia = patron_dia[test_index]
