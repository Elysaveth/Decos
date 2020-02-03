import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models,initializers,regularizers
import pickle
import matplotlib.pyplot as plt
from random import sample
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#PREPARACIÓN DATOS
df= pickle.load(open('global_stand_data.matrix', 'rb'))
data= df[0]
labels=df[1]
num_datos = len(labels)
test_index = list(range(0,num_datos,10))
data_index=list(range(num_datos))
for test_obs in test_index[::-1]:
    data_index.pop(test_obs)
val_index = sample(data_index,int(0.2*len(data_index)))
train_index=[]
for i in range(num_datos):
    if i not in val_index:
        train_index.append(i)

train_data= data[train_index,:,:]
train_labels=labels[train_index]
val_data= data[val_index,:,:]
val_labels=labels[val_index]
test_data=data[test_index,:,:]
test_labels=labels[test_index]

#PARAMENTROS DE LA RED
epochs=10
input_shape=(35,28,2)
#optimizer='adam'
inter_layers=2
dense_layer_nodes=[248,16,1]

def amse(y_pred, y_target):
  mse = tf.keras.losses.MeanSquaredError()
  return abs(mse(y_pred, y_target))

def create_model(kernel_regularizer=regularizers.l1(0.01)):
    #PARAMENTROS DE LA RED
    epochs=10
    input_shape=(35,28,2)
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
        model.add(layers.Dense(nodes, activation='relu', kernel_initializer = 'he_uniform',kernel_regularizer= kernel_regularizer))
    
    model.compile(optimizer='adam', loss = 'mean_absolute_error', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
kernel_regularizer= [regularizers.l1(0.01), regularizers.l1_l2(l1=0.01,l2=0.01),regularizers.l1(0.0)]
param_grid = dict(kernel_regularizer=kernel_regularizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_data, train_labels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

