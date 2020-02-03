
import numpy as np
import pickle
from scipy import stats

#PREPARACIÓN DATOS
df= pickle.load(open('data.matrix (8220,70,56)', 'rb'))
data=df[0]
datacanal0 = data[:,:,:,0]
datacanal1 = data[:,:,:,1]
stdatacanal0 = stats.zscore(datacanal0, axis=None)
stdatacanal1 = stats.zscore(datacanal1, axis=None) #ver también sklearn StandardScaler 
#o similares
stand_data_nacho_canalesseparados = np.empty(data.shape)
stand_data_nacho_canalesseparados[:,:,:,0]= stdatacanal0   #check que esta linea y la siguiente son sintaxis equivalentes
stand_data_nacho_canalesseparados[:,:,:,1]= stdatacanal1[:,:,:]

f=open('stand_data_nacho_canalesseparados.matrix','wb')
pickle.dump(df,f)


# ver también _preprocess_numpy_input

#datapolares = np.empty(data.shape)
#datapolares[:,:,:,0]=np.sqrt(data[:,:,:,0]**2 + data[:,:,:,1]**2)
#datapolares[:,:,:,1]=np.arctan2(data[:,:,:,1], data[:,:,:,0])



#Aquí a lo mejor queda un poco más claro lo que se hace. Es lo mismo
#dataU = data[:,:,:,0]
#dataV = data[:,:,:,1]
#datarho = np.sqrt(dataU**2+dataV**2)
#datatheta = np.arctan2(dataV, dataU)
#datapolares2 = np.empty((24670,70,56,2))
#datapolares2[:,:,:,0]=datarho[:,:,:]
#datapolares2[:,:,:,1]=datatheta[:,:,:]
