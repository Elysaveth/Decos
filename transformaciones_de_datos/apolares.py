# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:58:08 2020

@author: user
"""

import xarray as xr
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
infile = open("datosinterpoladosrecortados.matrix","rb")
data = pickle.load(infile)
#Esta es la forma que menos memoria ocupa, y que menos tarda
datapolares = np.empty(data.shape)
datapolares[:,:,:,0]=np.sqrt(data[:,:,:,0]**2 + data[:,:,:,1]**2)
datapolares[:,:,:,1]=np.arctan2(data[:,:,:,1], data[:,:,:,0])


#Aquí a lo mejor queda un poco más claro lo que se hace. Es lo mismo
#dataU = data[:,:,:,0]
#dataV = data[:,:,:,1]
#datarho = np.sqrt(dataU**2+dataV**2)
#datatheta = np.arctan2(dataV, dataU)
#datapolares2 = np.empty((24670,70,56,2))
#datapolares2[:,:,:,0]=datarho[:,:,:]
#datapolares2[:,:,:,1]=datatheta[:,:,:]

f=open('mapasinterpoladosenpolares.matrix','wb')
pickle.dump(datapolares,f)