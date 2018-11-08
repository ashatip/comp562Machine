import csv

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import scale
import numpy as np

print('opening file...')
with open('shapemap1000000.txt',"r") as f:
    # all_data=[x.split() for x in f.readlines()]
    all_data=[next(f).split() for i in range(12000)]

    myinputs = np.array([[float(i) for i in x] for x in all_data[::3]], dtype=float)
    myinputs = scale(myinputs, axis=0)

    coords = np.array([[float(i) for i in x] for x in all_data[2::3]], dtype=float)
    coords = scale(coords, axis=0)
print('finished opening')

lastcoord = coords[:,-3:] # get last three columns
lastcoordX = lastcoord[:,:1]

length = len(coords)
train_data = myinputs[:length//5*4] # 80%
train_labels = lastcoordX[:length//5*4] # 80%

test_data = myinputs[length//5*4:] # last 20%
test_labels = lastcoordX[length//5*4:] # last 20%

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

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

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

plot_history(history)


# PERFORMANCE ON TEST SET

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.5f}".format(mae))

# PREDICTION

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values []')
plt.ylabel('Predictions []')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error []")
plt.ylabel("Count")
plt.show()