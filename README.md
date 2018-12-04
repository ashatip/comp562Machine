# comp562Machine



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


## Floyd commands

floyd run --env tensorflow-1.11 --data thebowja/datasets/shapemap1000000:data --follow "python 20pointnnTF.py"
floyd run --env tensorflow-1.11 --gpu --data thebowja/datasets/shapemap1000000:data --follow "python 20pointnnTF.py"

