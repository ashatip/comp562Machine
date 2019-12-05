# comp562Machine

COMP562 (machine learning) UNC 2018 Fall final project.

Team Member: Paul Song, Christopher Alan Hemmer, Eric Xin, Ahmet Hatip

Neural network project to predict the location of a medical Robot based on 6 motors

Table of contents
=================

<!--ts-->
   * [UsingTFUpdated.py](#UsingTFUpdated.py)
   * [20pointnnTF.py](#20pointnnTF.py)
   * [usingTF_triple.py](#usingTF_triple.py)
   * [usingTFLoadModel.py](#usingTFLoadModel.py)
   * [usingTFSaveModel.py](#usingTFSaveModel.py)
<!--te-->

### UsingTFUpdated.py
6 inputs - 3 rotational, 3 translational\
3 outputs - x, y, z of last coordinate\
2 hidden layers - 64 nodes each - ReLU\
loss - MSE\
optimizer - tf.train.RMSPropOptimizer(0.001)\
split - 8/2

### 20pointnnTF.py
7 inputs - t, 3 rotational, 3 translational\
3 outputs - x, y, z of coordinate\
2 hidden layers - 64 nodes each - ReLU\
loss - MSE\
optimizer - tf.train.RMSPropOptimizer(0.001)\
split - 8/2

### usingTF_triple.py
unfinished

### usingTFLoadModel.py
```
model = keras.models.load_model('reversemod.h5')
optimizer = tf.train.RMSPropOptimizer(0.001)
model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
```
### usingTFSaveModel.py
add the following line after model.fit
```
model.save('modelscaling.h5')
```


## Floyd commands

floyd run --env tensorflow-1.11 --data thebowja/datasets/shapemap1000000:data --follow "python dofloydjob.py"
floyd run --env tensorflow-1.11 --gpu --data thebowja/datasets/shapemap1000000:data --follow "python 20pointnnTF.py"

