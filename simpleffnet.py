import numpy as np
import csv
from simpleneuralnet import Neural_Network

# # scale units
# X = X/np.amax(X, axis=0) # maximum of X array
# xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
# y = y/100 # max test score is 100

print('opening file...')
with open('shapemap1000000.txt',"r") as f:
    # all_data=[x.split() for x in f.readlines()]
    all_data=[next(f).split() for i in range(6000)]
    all_data = (all_data-np.mean(all_data, axis=0))/np.std(all_data, axis=0)
    myinputs = np.array([[float(i) for i in x] for x in all_data[::3]], dtype=float)
    coords = np.array([[float(i) for i in x] for x in all_data[2::3]], dtype=float)
print('finished opening')
lastcoord = coords[:,-3:] # get last three columns
lastcoord = lastcoord[:,-1:]
#print(len(lastcoord))
X = myinputs
# X = myinputs#/np.amax(myinputs, axis=0)
y = lastcoord
# y = lastcoord#/np.amax(lastcoord, axis=0)


NN = Neural_Network(iSize=6, oSize=1, hSize=3)
NN.learn(X, y, iterations=1000000, interval=100000)


# data = np.loadtxt("shapemap5thpolyhead.txt")
# myinputs = data[::4]
# Xcoeffs = data[1::4]
# Ycoeffs = data[2::4]
# Zcoeffs = data[3::4]

# X = myinputs
# y = Xcoeffs/np.amax(Xcoeffs, axis=0)
# NN = Neural_Network(iSize=6, oSize=6, hSize=6)
# NN.learn(X, y, iterations=1000000, interval=100000)

print("done")
#NN.saveWeights()
#NN.predict()