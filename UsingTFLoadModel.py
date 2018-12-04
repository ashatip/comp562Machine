import csv

import tensorflow as tf
import sys
from tensorflow import keras
from sklearn.preprocessing import scale
import numpy as np

datapoints = 1000000

print('opening file...')
with open('shapemap1000000.txt',"r") as f:
    myrange = 3*datapoints
    all_data=[next(f).split() for i in range(myrange)]

    # Creating input matrix
    myinputs = np.zeros((int(myrange/3)*20, 7))
    xcounter = 0
    for x in all_data[::3]:
        for y in range(0, 20):
            myinputs[xcounter][0] = y
            myinputs[xcounter][1:7] = [float(i) for i in x]
            xcounter += 1
    myinputs = scale(myinputs, axis=0)

    coords = np.zeros((int(myrange/3)*20, 3))
    counter = 0
    for x in all_data[2::3]:
        for y in range(0, 20):
            coords[counter] = x[y * 3:(y + 1) * 3]
            counter += 1
    coords = scale(coords, axis=0)
print('finished opening')

lastcoord = coords#[:,-3:] # get last three columns


length = len(coords)

train_data = myinputs[:length//5*4] # 80%
train_labels = lastcoord[:length//5*4] # 80%

test_data = myinputs[length//5*4:] # last 20%
test_labels = lastcoord[length//5*4:] # last 20%

print("Training set: {}".format(train_data.shape))  # xxx examples, 6 features
print("Testing set:  {}".format(test_data.shape))   # xxx examples, 6 features


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


model = keras.models.load_model('model' + str(datapoints) + '.h5')
optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
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

#plot_history(history)

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
plt.xlabel("Prediction Error []")
plt.ylabel("Count")
plt.show()

#for x in range(0, 50):
#    print(test_labels[x], test_predictions[x], sep=" : ")