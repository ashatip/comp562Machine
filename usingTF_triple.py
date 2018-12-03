import csv

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import scale
import numpy as np

print('opening file...')
with open('data/shapemap1000000.txt',"r") as f:
    # all_data=[x.split() for x in f.readlines()]
    all_data=[next(f).split() for i in range(12000)]

    myinputs = np.array([[float(i) for i in x] for x in all_data[::3]], dtype=float)
    myinputs = scale(myinputs, axis=0)

    coords = np.array([[float(i) for i in x] for x in all_data[2::3]], dtype=float)
    coords = scale(coords, axis=0)
print('finished opening')

lastcoord = coords[:,-3:] # get last three columns
lastX = lastcoord[:,:1]
lastY = lastcoord[:,:2]
lastZ = lastcoord[:,:3]

def split_data(data, perc): # perc is the percentage value of where the data will split. ie: .8 or .2
  splitind = int(len(data)*perc)
  return data[:splitind], data[splitind:]

train_data, test_data = split_data(myinputs, 0.8)
train_label_X, test_label_X = split_data(lastX, 0.8)
train_label_Y, test_label_Y = split_data(lastY, 0.8)
train_label_Z, test_label_Z = split_data(lastZ, 0.8)

print("Training set: {}".format(train_data.shape))  # xxx examples, 6 features
print("Testing set:  {}".format(test_data.shape))   # xxx examples, 6 features

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 1])
  plt.show()


# PREDICTION
def plot_prediction(data, labels):
  predictions = model.predict(data).flatten()

  plt.scatter(labels, predictions)
  plt.xlabel('True Values []')
  plt.ylabel('Predictions []')
  plt.axis('equal')
  plt.xlim(plt.xlim())
  plt.ylim(plt.ylim())
  plt.plot([-100, 100], [-100, 100])
  plt.show()

  error = predictions - labels
  plt.hist(error, bins = 50)
  plt.xlabel("Prediction Error []")
  plt.ylabel("Count")
  plt.show()

  return predictions

################################################################################


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model = build_model()
model.summary()

# Store training stats
history = model.fit(train_data, train_label_X, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)

# PERFORMANCE ON TEST SET
[loss, mae] = model.evaluate(test_data, test_label_X, verbose=0)
print("Testing set Mean Abs Error: ${:7.5f}".format(mae))
plot_prediction(test_data, test_label_X)

model = build_model()

# Store training stats
history = model.fit(train_data, train_label_Y, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)

# PERFORMANCE ON TEST SET
[loss, mae] = model.evaluate(test_data, test_label_Y, verbose=0)
print("Testing set Mean Abs Error: ${:7.5f}".format(mae))
plot_prediction(test_data, test_label_Y)

print("done")