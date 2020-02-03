# Use scikit-learn to grid search the number of neurons
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import layers, models,initializers,regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from random import sample
#from tensorflow.keras.constraints import maxnorm
import pickle

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
for i in data_index:
    if i not in val_index:
        train_index.append(i)

train_data= data[train_index,:,:]
train_labels=labels[train_index]
val_data= data[val_index,:,:]
val_labels=labels[val_index]
test_data=data[test_index,:,:]
test_labels=labels[test_index]

# Function to create model, required for KerasClassifier
def create_model(filter1=64,filter2=32):
    input_shape=(35,28,2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
    inter_layers=2

    model = models.Sequential()
    for i in range(inter_layers):
        if i == 0:
            model.add(layers.Conv2D(filters=filter1, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters=filter2, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=filter1, kernel_size=(3, 3), activation='relu'))
    dense_layer_nodes=[int(np.sqrt(model.output.shape[3])),1]
    model.add(layers.Flatten())
    for nodes in dense_layer_nodes:
        model.add(layers.Dense(nodes, activation='relu', kernel_initializer = 'he_uniform',kernel_regularizer= tf.keras.regularizers.l1(0.01)))
    model.compile(optimizer=optimizer, loss= 'mean_squared_error', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
# define the grid search parameters
filter1= [9,16,32,64]
filter2=[6,8,16,32]
param_grid = dict(filter1=filter1,filter2=filter2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_data, train_labels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))