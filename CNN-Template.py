import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import pickle

data= pickle.load(open('data.matrix', 'rb'))
labels=pd.read_csv('produccion_nueva.csv',sep=';',header=None)[2]
print(type(labels[0]))
msk = np.random.rand(len(data)) < 0.95
train_data= data[msk,:,:]
train_labels=labels[msk]
test_data= data[~msk,:,:]
test_labels=labels[~msk]
epochs=500
input_shape=(35,28,2)
optimizer='adam'
inter_layers=2
dense_layer_nodes=[248,16,1]

model = models.Sequential()
for i in range(inter_layers):
    if i == 0:
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
    else:
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(layers.Flatten())
for nodes in dense_layer_nodes:
    model.add(layers.Dense(nodes, activation='relu'))

model.summary()

def amse(y_pred, y_target):
  mse = tf.keras.losses.MeanSquaredError()
  return abs(mse(y_pred, y_target))

model.compile(optimizer=optimizer,
              loss=amse)

model.fit(train_data, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print(test_loss)