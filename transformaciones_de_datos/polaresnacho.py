# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:57:06 2020

@author: marik
"""

import xarray as xr
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

import datetime

data=np.array([[[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]],[[[13,14],[15,16]],[[17,18],[19,20]],[[21,22],[23,24]]],[[[25,26],[27,28]],[[29,30],[31,32]],[[33,34],[35,36]]]])

#Esta es la forma que menos memoria ocupa, y que menos tarda
datapolares = np.empty((3,3,2,2))
datapolares[:,:,:,0]=data[:,:,:,0]**2 + data[:,:,:,1]**2
datapolares[:,:,:,1]=np.arctan2(data[:,:,:,1], data[:,:,:,0])

for obs in data:
    data_lon=[]
    for longitud in obs:
        ro=[]
        for coordenadas in longitud:
            mod= coordenadas[0]**2+coordenadas[1]**2
            ro.append(mod)
        ro= np.array(ro).reshape(1,1,2)
        #Hasta aquí saca lo que quiero
        try:
            data_lon=np.concatenate((data_lon,ro),axis=1) 
            pass
        except:
            data_lon=ro
    try:
        data_trans=np.concatenate((data_trans,data_lon),axis=0)
        pass
    except:
        data_trans=data_lon
print(data_trans.shape)

#Aquí a lo mejor queda un poco más claro lo que se hace. Es lo mismo
#dataU = data[:,:,:,0]
#dataV = data[:,:,:,1]
#datarho = np.sqrt(dataU**2+dataV**2)
#datatheta = np.arctan2(dataV, dataU)
#datapolares2 = np.empty((24670,70,56,2))
#datapolares2[:,:,:,0]=datarho[:,:,:]
#datapolares2[:,:,:,1]=datatheta[:,:,:]