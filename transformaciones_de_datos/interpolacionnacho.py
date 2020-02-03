# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2019

@author: marik
"""

import pandas as pd
import numpy as np
import pickle

df= pickle.load(open('data.matrix (8220,70,56)', 'rb'))
#data=np.array([[[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]],[[[13,14],[15,16]],[[17,18],[19,20]],[[21,22],[23,24]]],[[[25,26],[27,28]],[[29,30],[31,32]],[[33,34],[35,36]]]])
data0=df[0] #estos son los datos en las horas que conocemos, en todas las horas, latitudes, longitudes

obs_data=data0.shape[0] #esto es cuantas previsiones tenemos

aux1 = np.copy(data0[:obs_data-1,:,:,:]) #esto quita la última observación
aux2 = np.copy(data0[1:,:,:,:]) #esto quita la primera observación 
#Se tiene que aux1[i,:,:,:]=data0[i,:,:,:] y aux2[i,:,:,:]=data0[i+1,:,:,:]
data1=(2/3)*aux1 + (1/3)*aux2 #estos son los interpolados en las horas +1
data2=(1/3)*aux1 + (2/3)*aux2 #estos son los interpolados en las horas +2


#lo que viene a continuación es un truco basado en
#https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
nueva_data=np.empty([3*obs_data-2,data0.shape[1],data0.shape[2],2])
nueva_data[0::3,:,:,:]=data0
nueva_data[1::3,:,:,:]=data1
nueva_data[2::3,:,:,:]=data2

#lo que viene a continuación es un CHECK, se puede comentar tranquilamente.
#Quizás ayude a entender lo que se está haciendo.
print(nueva_data[12,10,10,1], data0[4,10,10,1])
print(nueva_data[13,10,10,1], (2/3)*data0[4,10,10,1]+ (1/3)*data0[5,10,10,1])
print(nueva_data[14,10,10,1], (1/3)*data0[4,10,10,1]+ (2/3)*data0[5,10,10,1])



f=open('datos_interpoladosnacho.matrix','wb')
pickle.dump(nueva_data,f)