import tensorflow as tf
import os
import sys
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualkeras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


dBCoilDatasetL = pd.read_csv("newCoilDataFrame_Q.csv", low_memory=False)

dBCoilDatasetL
dataset = dBCoilDatasetL.copy() # ╨Ч╨░╨│╤А╤Г╨╢╨░╨╡╨╝ ╨┤╨░╨╜╨╜╤Л╨╡
dataset.tail()

dataset.isna().sum() # ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╨╝ NaN ╨╕ ╨┐╤А╨╛╤З╨╡╨╡??
dataset = dataset.dropna() # ╨г╨┤╨░╨╗╤П╨╡╨╝ NaN ╨╕ ╨┐╤А╨╛╤З╨╡╨╡??
train_dataset = dataset.sample(frac=0.95, random_state=0) # ╨а╨░╨╖╨┤╨╡╨╗╨╡╨╜╨╕╨╡ ╨┤╨░╨╜╨╜╤Л╤Е
test_dataset = dataset.drop(train_dataset.index) # ╤В╨╡╤Б╤В╨╛╨▓╤Л╨╡ ╨╕ ╤В╤А╨╡╨╜╨╕╤А╨╛╨▓╨╛╤З╨╜╤Л╨╡ ╨┤╨░╨╜╨╜╤Л╤Е
train_dataset

#big Data Analis
import seaborn as sns
sns.pairplot(train_dataset[['nTurns', 'width', 'OD', 'Frequency', 'Quality']], diag_kind='kde')
#!!!!!
train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_features
test_features

train_labels = train_features.pop('Quality')
test_labels = test_features.pop('Quality')

# train_labels
train_labels

test_labels

train_dataset.describe().transpose()[['mean', 'std']]

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

train_features = np.array(train_features)
normalizer = preprocessing.Normalization(input_shape=[4,])
normalizer.adapt(train_features)

# Model description
Q_Inductor_model = tf.keras.Sequential([normalizer,
    layers.Dense(64, activation="tanh"),
    layers.Dense(64, activation="tanh"),
    layers.Dense(64, activation="tanh"),
    layers.Dropout(0.01, input_shape=(64,)),
    layers.Dense(36,  activation="tanh"),
    layers.Dense(36,  activation="tanh"),
    layers.Dense(36,  activation="tanh"),
    layers.Dense(1)])

Q_Inductor_model.predict(train_features[:10])
Q_Inductor_model.layers[1].kernel

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Inductance]')
    plt.legend()
    plt.grid(True)

rate = 0.01
batch = 250
split = 0.35
for i in range(1):    
    Q_Inductor_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate),
    loss='mse', metrics=['accuracy'])
    # %%time
    history = Q_Inductor_model.fit(train_features, train_labels, 
        epochs=10,
        batch_size = batch,
        # Calculate validation results on 20% of the training data
        validation_split = split)
    rate = rate - 0.001
    batch = batch - 10 * i
    print('Iteration ' + str(i))
    print('Rate ' + str(rate))
    

plot_loss(history)

Q_Inductor_model.save('Q_full_model.h5')

rate = 0.01
batch = 250
split = 0.35
epohi = 24
for i in range(5):    
    Q_Inductor_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=rate),
    loss='mse', metrics=['accuracy'])
    # %%time
    history = Q_Inductor_model.fit(train_features, train_labels, 
        epochs=epohi,
        batch_size = batch,
        # Calculate validation results on 20% of the training data
        validation_split = split)
    rate = rate - 0.001
    batch = batch - 10 * i
    print('Iteration ' + str(i))
    print('Rate ' + str(rate))
    Q_Inductor_model.save('QQQ_full_model.h5')

plot_loss(history)
Q_Inductor_model.save('QQQ_full_model.h5')

test_predictions = Q_Inductor_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Q]')
plt.ylabel('Predictions [Q]')
_ = plt.plot()

#!mkdir -p saved_model
Q_Inductor_model.save('saved_model/Q_Inductor_model')


#import tensorflow as tf
#import os
#import sys
#import seaborn as sns

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import visualkeras

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing


Q_Inductor_model = tf.keras.models.load_model('saved_model/Q_Inductor_model')



nt = 1
wd = 12
od = 300

real_C_data = pd.read_csv('q/n' + str(nt) + '_w' + str(wd) + '_OD' + str(od) + '_Qcurve.txt', sep=' ')
real_C_data.columns = ['nTurns','width', 'OD', 'Frequency','Quality', 'XXX']
real_C_data

freq = real_C_data['Frequency'].astype(float).tolist()
qqq = real_C_data['Quality'].astype(float).tolist()
num = 1e9
freq = [i / num for i in freq]

x = tf.linspace(0.0, 30, 100)
y = []
for i in np.arange(0.0, 30 , 0.3):
    q_val = float((Q_Inductor_model.predict([[nt,wd,od,i]]).astype(float)))
    y.append(q_val)

plt.plot(freq,qqq,'g*-', x, y,'r-')