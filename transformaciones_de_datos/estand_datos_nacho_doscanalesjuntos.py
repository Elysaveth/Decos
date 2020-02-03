
import numpy as np
import pickle
from scipy import stats

#PREPARACIÓN DATOS
df= pickle.load(open('data.matrix (8220,70,56)', 'rb'))
data=df[0]
stand_data = stats.zscore(data, axis=None)
f=open('stand_data_nacho_canalesjuntos.matrix','wb')
pickle.dump(df,f)


# ver también _preprocess_numpy_input