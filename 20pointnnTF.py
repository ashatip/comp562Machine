import csv

import tensorflow as tf
import sys
from tensorflow import keras
from sklearn.preprocessing import scale
import numpy as np

print('opening file...')
with open('data/shapemap1000000.txt',"r") as f:
    myrange = 120000
    all_data=[next(f).split() for i in range(myrange)]

    # Creating input matrix
    myinputs = np.zeros((int(myrange/3)*20, 7))
    xcounter = 0
    for x in all_data[::3]:
        for y in range(0, 20):
            myinputs[xcounter][0] = y
            myinputs[xcounter][1:7] = [float(i) for i in x]
            xcounter += 1
    #myinputs = scale(myinputs, axis=0)

    coords = np.zeros((int(myrange/3)*20, 3))
    counter = 0
    for x in all_data[2::3]:
        for y in range(0, 20):
            coords[counter] = x[y * 3:(y + 1) * 3]
            counter += 1
    #coords = scale(coords, axis=0)
print('finished opening')

lastcoord = coords#[:,-3:] # get last three columns

def split_data(data, perc): # perc is the percentage value of where the data will split. ie: .8 or .2
  splitind = int(len(data)*perc)
  return data[:splitind], data[splitind:]

train_data, test_data = split_data(myinputs, 0.8)
train_labels, test_labels = split_data(lastcoord, 0.8)

print("Training set: {}".format(train_data.shape))  # xxx examples, 6 features
print("Testing set:  {}".format(test_data.shape))   # xxx examples, 6 features


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(3)
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
    plt.ylim([0, .1])
    plt.show()

plot_history(history)

# PERFORMANCE ON TEST SET

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.5f}".format(mae))

# PREDICTION

test_predictions = model.predict(test_data)#.flatten()

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
plt.legend(['X','Y','Z'])
plt.xlabel("Prediction Error []")
plt.ylabel("Count")
plt.show()

#for x in range(0, 2):
#    print(test_labels[x], test_predictions[x], sep=" : ")