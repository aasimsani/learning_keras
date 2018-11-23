# A small experiment with the Happy house dataset
An experiment to learn Keras on a bored evening when I wanted to understand how I could work on ML faster


Here's the Keras summary for the rudimentary CNN for the Happy house project

```Train Accuracy: 100%```
```Test Accuracy: 96.67%```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        544
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 16)        2064
_________________________________________________________________
P1 (MaxPooling2D)            (None, 7, 7, 16)          0
_________________________________________________________________
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dropout (Dropout)            (None, 784)               0
_________________________________________________________________
D1 (Dense)                   (None, 256)               200960
_________________________________________________________________
D2 (Dense)                   (None, 2)                 514
=================================================================
Total params: 204,082
Trainable params: 204,082
Non-trainable params: 0
_________________________________________________________________
````
