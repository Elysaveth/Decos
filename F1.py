#Entrenamiento con 100u y 100v 

#Entrenamos con MAE
#Indice de test random
#Sin etiquetas Noche/Dia
#No tocamos numero de neuronas original

#Entrenando con el MAE y 30 epocas se obtiene:
#El MAE del test es:
#76.43583193919697
#El MAE entre la potencia instalada es:
#0.021705228945062948

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from random import sample
from tensorflow.keras import layers, models, initializers


infile = open("../data/datosinterpoladosrecortados.matrix","rb")
data = pickle.load(infile)
labels=pd.read_csv('../data/ManoriUTC.csv',sep=';',header=None)[2]
labels=labels.to_numpy()

#DIVISION DE LOS DATOS DE MANERA RANDOM
msk = np.random.rand(len(data)) < 0.90
test_data= data[~msk]
test_labels= labels[~msk]
test_index = list(range(0,len(test_data)))

num_datos = len(labels)
data_index=[]
for i in range(len(msk)):
    if msk[i]==1:
        data_index.append(i)

#en los data estan los val y los train
val_index = sample(data_index,int(0.2*len(data_index)))
val_data= data[val_index,:,:]
val_labels=labels[val_index]

train_index=[]
for i in data_index:
    if i not in val_index:
        train_index.append(i)

train_data=data[train_index]
train_labels=labels[train_index]

##PARAMETROS DE RED

epochs=30
input_shape=(70,56,2)
optimizer='adam'
inter_layers=2
dense_layer_nodes=[248,16,1]


model = models.Sequential()
for i in range(inter_layers):
    if i == 0:
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    else:
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(layers.Flatten())
for nodes in dense_layer_nodes:
    model.add(layers.Dense(nodes, activation='relu', kernel_initializer = tf.keras.initializers.he_uniform(seed=None)))

model.summary()

def amse(y_pred, y_target):
  mse = tf.keras.losses.mean_absolute_error
  #mse = tf.keras.losses.MeanSquaredError()
  return abs(mse(y_pred, y_target))

model.compile(optimizer=optimizer,
              loss=amse)

history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(val_data, val_labels))

test_loss = model.evaluate(test_data, test_labels)
val_loss= model.evaluate(val_data,val_labels)
test_pred = model.predict(test_data, batch_size= None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers = 1, use_multiprocessing = False)

#MAE test
print('El MAE del test es:')
print(test_loss)
print('El MAE entre la potencia instalada es:')
print(test_loss/3521.54)


#pintar la grafica de train y validation
plt.plot(history.history['loss'])
#loss es la perdida de validacion en pada epoch
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train','Val'], loc='upper left')
plt.show()
